# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional, Union

import regex as re

from vllm.entrypoints.chat_utils import random_tool_call_id
from vllm.entrypoints.openai.protocol import (ChatCompletionRequest,
                                              DeltaFunctionCall, DeltaMessage,
                                              DeltaToolCall,
                                              ExtractedToolCallInformation,
                                              FunctionCall, ToolCall)
from vllm.entrypoints.openai.tool_parsers.abstract_tool_parser import (
    ToolParser, ToolParserManager)
from vllm.entrypoints.openai.tool_parsers.utils import (
    extract_intermediate_diff)
from vllm.logger import init_logger
from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


@dataclass
class RegexConfig:
    """Configuration for regex-based tool parsing"""
    # Main pattern to extract tool calls section
    tool_call_pattern: str
    # Pattern to extract individual function calls (optional)
    function_pattern: Optional[str] = None
    # Pattern to extract function name (optional)
    function_name_pattern: Optional[str] = None
    # Pattern to extract arguments (optional)
    arguments_pattern: Optional[str] = None
    # Tokens to strip from extracted content
    strip_tokens: Optional[list[str]] = None
    # Streaming configuration
    streaming_chunk_pattern: Optional[str] = None
    streaming_accumulate: bool = True

    def __post_init__(self):
        if self.strip_tokens is None:
            self.strip_tokens = []

        # Validate regex patterns
        try:
            re.compile(self.tool_call_pattern)
            if self.function_pattern:
                re.compile(self.function_pattern)
            if self.function_name_pattern:
                re.compile(self.function_name_pattern)
            if self.arguments_pattern:
                re.compile(self.arguments_pattern)
            if self.streaming_chunk_pattern:
                re.compile(self.streaming_chunk_pattern)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}") from e


@ToolParserManager.register_module("regex")
class RegexToolParser(ToolParser):
    """
    Generic regex-based tool parser for custom model support.
    
    This parser allows users to define custom regex patterns to extract
    tool calls from model outputs, eliminating the need for model-specific
    implementations.
    
    Used when --enable-auto-tool-choice --tool-call-parser regex
    --tool-parser-config '{"tool_call_pattern": "...", ...}' are set.
    """

    def __init__(self, tokenizer: AnyTokenizer, config: Optional[dict] = None):
        super().__init__(tokenizer)

        if config is None:
            raise ValueError("RegexToolParser requires a configuration dict")

        try:
            self.config = RegexConfig(**config)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid configuration for RegexToolParser: {e}") from e

        # Compile regex patterns
        self.tool_call_regex = re.compile(self.config.tool_call_pattern,
                                          re.DOTALL)
        self.function_regex = re.compile(
            self.config.function_pattern,
            re.DOTALL) if self.config.function_pattern else None
        self.function_name_regex = re.compile(
            self.config.function_name_pattern,
            re.DOTALL) if self.config.function_name_pattern else None
        self.arguments_regex = re.compile(
            self.config.arguments_pattern,
            re.DOTALL) if self.config.arguments_pattern else None
        self.streaming_regex = re.compile(
            self.config.streaming_chunk_pattern,
            re.DOTALL) if self.config.streaming_chunk_pattern else None

        # Streaming state
        self.accumulated_text = ""
        self.current_tool_calls: list[dict] = []
        self.last_processed_length = 0

    def _strip_tokens(self, text: str) -> str:
        """Remove configured tokens from text"""
        if self.config.strip_tokens:
            for token in self.config.strip_tokens:
                text = text.replace(token, "")
        return text.strip()

    def _extract_tool_calls_from_text(self, text: str) -> list[dict]:
        """Extract tool calls from text using configured patterns"""
        tool_calls = []

        # Extract tool call section
        tool_matches = self.tool_call_regex.findall(text)
        if not tool_matches:
            return []

        for tool_match in tool_matches:
            tool_content = self._strip_tokens(tool_match)

            # Try to parse as JSON first
            try:
                # If it's valid JSON, parse it directly
                parsed = json.loads(tool_content)
                if isinstance(parsed, list):
                    tool_calls.extend(parsed)
                else:
                    tool_calls.append(parsed)
                continue
            except json.JSONDecodeError:
                pass

            # If not JSON, use regex patterns
            if self.function_regex:
                function_matches = self.function_regex.findall(tool_content)
                for func_match in function_matches:
                    tool_call = {}

                    # Extract function name
                    if self.function_name_regex:
                        name_match = self.function_name_regex.search(
                            func_match)
                        if name_match:
                            tool_call["name"] = name_match.group(
                                1) if name_match.groups(
                                ) else name_match.group(0)
                    else:
                        # Try to extract from the match itself
                        tool_call["name"] = func_match if isinstance(
                            func_match, str) else func_match[0]

                    # Extract arguments
                    if self.arguments_regex:
                        args_match = self.arguments_regex.search(
                            func_match if isinstance(func_match, str
                                                     ) else str(func_match))
                        if args_match:
                            args_str = args_match.group(
                                1) if args_match.groups(
                                ) else args_match.group(0)
                            try:
                                tool_call["arguments"] = json.loads(args_str)
                            except json.JSONDecodeError:
                                tool_call["arguments"] = args_str

                    if "name" in tool_call:
                        tool_calls.append(tool_call)
            else:
                # No function pattern, try to parse the content as a whole
                try:
                    # Try to extract function calls from the content
                    # This is a fallback for simple formats
                    if "{" in tool_content and "}" in tool_content:
                        # Try to parse as JSON object
                        parsed = json.loads(tool_content)
                        if isinstance(parsed, dict) and "name" in parsed:
                            tool_calls.append(parsed)
                except json.JSONDecodeError:
                    logger.debug("Could not parse tool content as JSON: %s",
                                 tool_content[:100])

        return tool_calls

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from a complete model response using regex patterns.
        """
        # Check if tool call pattern matches
        if not self.tool_call_regex.search(model_output):
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

        try:
            # Extract tool calls
            raw_tool_calls = self._extract_tool_calls_from_text(model_output)

            # Convert to ToolCall objects
            tool_calls = []
            for raw_call in raw_tool_calls:
                if not isinstance(raw_call, dict) or "name" not in raw_call:
                    continue

                # Extract arguments
                arguments = raw_call.get("arguments", {})
                if isinstance(arguments, dict):
                    arguments_str = json.dumps(arguments, ensure_ascii=False)
                elif isinstance(arguments, str):
                    # Validate it's valid JSON
                    try:
                        json.loads(arguments)
                        arguments_str = arguments
                    except json.JSONDecodeError:
                        arguments_str = json.dumps({"raw": arguments})
                else:
                    arguments_str = json.dumps({})

                tool_calls.append(
                    ToolCall(type="function",
                             function=FunctionCall(name=raw_call["name"],
                                                   arguments=arguments_str)))

            # Get content before tool calls
            match = self.tool_call_regex.search(model_output)
            content = model_output[:match.start()].strip() if match else None

            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None)

        except Exception:
            logger.exception("Error extracting tool calls with regex parser")
            return ExtractedToolCallInformation(tools_called=False,
                                                tool_calls=[],
                                                content=model_output)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> Union[DeltaMessage, None]:
        """
        Extract tool calls from streaming response using regex patterns.
        """
        # Check if we should accumulate text
        if self.config.streaming_accumulate:
            self.accumulated_text = current_text
        else:
            self.accumulated_text += delta_text

        # Check if tool call pattern matches yet
        if not self.tool_call_regex.search(self.accumulated_text):
            # No tool calls detected yet, return delta as content
            return DeltaMessage(content=delta_text)

        try:
            # Extract current tool calls
            current_tool_calls = self._extract_tool_calls_from_text(
                self.accumulated_text)

            # If no complete tool calls yet, accumulate
            if not current_tool_calls:
                return None

            # Check if we have new tool calls
            if len(current_tool_calls) > len(self.current_tool_calls):
                # New tool call detected
                new_index = len(self.current_tool_calls)
                new_call = current_tool_calls[new_index]

                if "name" in new_call:
                    # Prepare function call with arguments if present
                    function_args = {}
                    function_args["name"] = new_call["name"]

                    if "arguments" in new_call:
                        args = new_call["arguments"]
                        if isinstance(args, dict):
                            args_str = json.dumps(args, ensure_ascii=False)
                        else:
                            args_str = str(args)
                        function_args["arguments"] = args_str

                    # Send new tool call
                    delta_tool_call = DeltaToolCall(
                        index=new_index,
                        type="function",
                        id=random_tool_call_id(),
                        function=DeltaFunctionCall(**function_args).model_dump(
                            exclude_none=True))

                    self.current_tool_calls.append(new_call)
                    return DeltaMessage(tool_calls=[delta_tool_call])

            # Check for updates to existing tool calls
            elif len(current_tool_calls) == len(self.current_tool_calls):
                # Check last tool call for argument updates
                if self.current_tool_calls:
                    last_index = len(self.current_tool_calls) - 1
                    prev_call = self.current_tool_calls[last_index]
                    curr_call = current_tool_calls[last_index]

                    # Check for argument updates
                    prev_args = prev_call.get("arguments", {})
                    curr_args = curr_call.get("arguments", {})

                    if (curr_args != prev_args and isinstance(curr_args, dict)
                            and isinstance(prev_args, dict)):
                        # Calculate diff
                        curr_str = json.dumps(curr_args, ensure_ascii=False)
                        prev_str = json.dumps(prev_args, ensure_ascii=False)
                        diff = extract_intermediate_diff(curr_str, prev_str)

                        if diff:
                            self.current_tool_calls[last_index] = curr_call
                            return DeltaMessage(tool_calls=[
                                DeltaToolCall(index=last_index,
                                              function=DeltaFunctionCall(
                                                  arguments=diff).model_dump(
                                                      exclude_none=True))
                            ])

            return None

        except Exception:
            logger.exception("Error in streaming regex tool extraction")
            return None
