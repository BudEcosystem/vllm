# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from unittest.mock import MagicMock

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction, run_tool_extraction_streaming)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

# Test configurations for different formats
XML_CONFIG = {
    "tool_call_pattern": r"<tools>(.*?)</tools>",
    "function_pattern": r"<function>(.*?)</function>",
    "function_name_pattern": r"<name>(.*?)</name>",
    "arguments_pattern": r"<arguments>(.*?)</arguments>",
    "strip_tokens": ["<tools>", "</tools>"]
}

JSON_CONFIG = {
    "tool_call_pattern": r"\[TOOL_CALLS\](.*?)(?:\[/TOOL_CALLS\]|$)",
    "strip_tokens": ["[TOOL_CALLS]", "[/TOOL_CALLS]"]
}

CUSTOM_CONFIG = {
    "tool_call_pattern": r"@@@TOOLS(.*?)@@@END",
    "function_pattern": r"FUNC:(.*?)(?=FUNC:|$)",
    "function_name_pattern": r"^(\w+)",
    "arguments_pattern": r"\((.*?)\)$",
    "strip_tokens": ["@@@TOOLS", "@@@END"]
}

# Test cases for XML format
XML_SIMPLE_OUTPUT = """<tools>
<function>
<name>get_weather</name>
<arguments>{"city": "San Francisco", "metric": "celsius"}</arguments>
</function>
</tools>"""

XML_MULTIPLE_OUTPUT = """<tools>
<function>
<name>get_weather</name>
<arguments>{"city": "San Francisco"}</arguments>
</function>
<function>
<name>get_time</name>
<arguments>{"timezone": "UTC"}</arguments>
</function>
</tools>"""

# Test cases for JSON format
JSON_SIMPLE_OUTPUT = ('[TOOL_CALLS][{"name": "get_weather", "arguments": '
                      '{"city": "Paris", "metric": "celsius"}}]')

JSON_MULTIPLE_OUTPUT = """[TOOL_CALLS][
{"name": "get_weather", "arguments": {"city": "London"}},
{"name": "calculate", "arguments": {"expression": "2+2"}}
]"""

# Test cases for custom format
CUSTOM_SIMPLE_OUTPUT = """@@@TOOLS
FUNC:get_weather({"city": "Tokyo", "metric": "fahrenheit"})
@@@END"""

CUSTOM_MULTIPLE_OUTPUT = """@@@TOOLS
FUNC:search({"query": "python tutorials"})
FUNC:send_email({"to": "user@example.com", "subject": "Hello"})
@@@END"""


def create_regex_tool_parser(config: dict) -> ToolParser:
    """Helper to create a regex tool parser with given config"""
    mock_tokenizer = MagicMock()
    mock_tokenizer.get_vocab.return_value = {}
    return ToolParserManager.get_tool_parser("regex")(mock_tokenizer,
                                                      config=config)


@pytest.mark.parametrize("streaming", [True, False])
def test_no_tool_call(streaming: bool):
    """Test that regular text without tool calls is handled correctly"""
    tool_parser = create_regex_tool_parser(XML_CONFIG)
    model_output = "How can I help you today?"

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert content == model_output
    assert len(tool_calls) == 0


@pytest.mark.parametrize("streaming", [True, False])
def test_xml_format_single_tool(streaming: bool):
    """Test XML format with single tool call"""
    tool_parser = create_regex_tool_parser(XML_CONFIG)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              XML_SIMPLE_OUTPUT,
                                              streaming=streaming)

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].type == "function"
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {
        "city": "San Francisco",
        "metric": "celsius"
    }


@pytest.mark.parametrize("streaming", [True, False])
def test_xml_format_multiple_tools(streaming: bool):
    """Test XML format with multiple tool calls"""
    tool_parser = create_regex_tool_parser(XML_CONFIG)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              XML_MULTIPLE_OUTPUT,
                                              streaming=streaming)

    assert content == ""
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[1].function.name == "get_time"


@pytest.mark.parametrize("streaming", [True, False])
def test_json_format_single_tool(streaming: bool):
    """Test JSON format with single tool call"""
    tool_parser = create_regex_tool_parser(JSON_CONFIG)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              JSON_SIMPLE_OUTPUT,
                                              streaming=streaming)

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {
        "city": "Paris",
        "metric": "celsius"
    }


@pytest.mark.parametrize("streaming", [True, False])
def test_json_format_multiple_tools(streaming: bool):
    """Test JSON format with multiple tool calls"""
    tool_parser = create_regex_tool_parser(JSON_CONFIG)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              JSON_MULTIPLE_OUTPUT,
                                              streaming=streaming)

    assert content == ""
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[1].function.name == "calculate"


@pytest.mark.parametrize("streaming", [True, False])
def test_custom_format_single_tool(streaming: bool):
    """Test custom format with single tool call"""
    tool_parser = create_regex_tool_parser(CUSTOM_CONFIG)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              CUSTOM_SIMPLE_OUTPUT,
                                              streaming=streaming)

    assert content == ""
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"
    assert json.loads(tool_calls[0].function.arguments) == {
        "city": "Tokyo",
        "metric": "fahrenheit"
    }


@pytest.mark.parametrize("streaming", [True, False])
def test_custom_format_multiple_tools(streaming: bool):
    """Test custom format with multiple tool calls"""
    tool_parser = create_regex_tool_parser(CUSTOM_CONFIG)

    content, tool_calls = run_tool_extraction(tool_parser,
                                              CUSTOM_MULTIPLE_OUTPUT,
                                              streaming=streaming)

    assert content == ""
    assert len(tool_calls) == 2
    assert tool_calls[0].function.name == "search"
    assert tool_calls[1].function.name == "send_email"


@pytest.mark.parametrize("streaming", [True, False])
def test_mixed_content_and_tools(streaming: bool):
    """Test output with both regular content and tool calls"""
    tool_parser = create_regex_tool_parser(XML_CONFIG)
    model_output = "Let me help you with that. " + XML_SIMPLE_OUTPUT

    content, tool_calls = run_tool_extraction(tool_parser,
                                              model_output,
                                              streaming=streaming)

    assert content == "Let me help you with that. "
    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"


def test_invalid_regex_config():
    """Test that invalid regex patterns raise appropriate errors"""
    invalid_config = {
        "tool_call_pattern": r"[invalid regex"  # Invalid regex
    }

    with pytest.raises(ValueError, match="Invalid regex pattern"):
        mock_tokenizer = MagicMock()
        ToolParserManager.get_tool_parser("regex")(mock_tokenizer,
                                                   config=invalid_config)


def test_missing_config():
    """Test that missing config raises appropriate error"""
    mock_tokenizer = MagicMock()

    with pytest.raises(ValueError, match="requires a configuration dict"):
        ToolParserManager.get_tool_parser("regex")(mock_tokenizer)


def test_malformed_tool_output():
    """Test handling of malformed tool output"""
    tool_parser = create_regex_tool_parser(XML_CONFIG)
    malformed_output = "<tools><function><name>broken"  # Incomplete XML

    content, tool_calls = run_tool_extraction(tool_parser,
                                              malformed_output,
                                              streaming=False)

    # Should handle gracefully - either parse what it can or return as content
    assert len(tool_calls) == 0 or (len(tool_calls) == 1 and
                                    tool_calls[0].function.name == "broken")


def test_streaming_with_chunks():
    """Test streaming mode with realistic chunks"""
    tool_parser = create_regex_tool_parser(JSON_CONFIG)

    model_output_chunks = [
        "[TOOL_CALLS][{\"name\": \"get_",
        "weather\", \"arguments\": {\"city\":",
        " \"Berlin\", \"metric\": \"celsius\"}}]"
    ]

    reconstructor = run_tool_extraction_streaming(
        tool_parser, model_output_chunks, assert_one_tool_per_delta=False)

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Berlin",
        "metric": "celsius"
    }


def test_empty_tool_calls():
    """Test handling of empty tool call sections"""
    tool_parser = create_regex_tool_parser(JSON_CONFIG)
    empty_output = "[TOOL_CALLS][]"

    content, tool_calls = run_tool_extraction(tool_parser,
                                              empty_output,
                                              streaming=False)

    assert content == ""
    assert len(tool_calls) == 0


def test_nested_json_arguments():
    """Test handling of nested JSON in arguments"""
    tool_parser = create_regex_tool_parser(JSON_CONFIG)
    nested_output = """[TOOL_CALLS][{
        "name": "create_user",
        "arguments": {
            "user": {
                "name": "John Doe",
                "address": {
                    "city": "New York",
                    "country": "USA"
                }
            },
            "roles": ["admin", "user"]
        }
    }]"""

    content, tool_calls = run_tool_extraction(tool_parser,
                                              nested_output,
                                              streaming=False)

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "create_user"
    args = json.loads(tool_calls[0].function.arguments)
    assert args["user"]["name"] == "John Doe"
    assert args["user"]["address"]["city"] == "New York"
    assert args["roles"] == ["admin", "user"]


# Integration test configurations for real-world scenarios
INTEGRATION_CONFIGS = {
    "xml_style": {
        "tool_call_pattern": r"<tools>(.*?)</tools>",
        "function_pattern": r"<function>(.*?)</function>",
        "function_name_pattern": r"<name>(.*?)</name>",
        "arguments_pattern": r"<arguments>(.*?)</arguments>",
        "strip_tokens": ["<tools>", "</tools>"]
    },
    "json_block": {
        "tool_call_pattern": r"```tools\n(.*?)\n```",
        "strip_tokens": ["```tools", "```"]
    },
    "bracket_style": {
        "tool_call_pattern": r"\[TOOL_CALLS\](.*?)\[/TOOL_CALLS\]",
        "strip_tokens": ["[TOOL_CALLS]", "[/TOOL_CALLS]"]
    },
    "custom_delimiter": {
        "tool_call_pattern": r"<<<TOOLS>>>(.*?)<<<END>>>",
        "function_pattern": r"CALL:(.*?)(?=CALL:|$)",
        "function_name_pattern": r"^(\w+)",
        "arguments_pattern": r"\((.*?)\)$",
        "strip_tokens": ["<<<TOOLS>>>", "<<<END>>>"]
    }
}

# Integration test outputs
INTEGRATION_OUTPUTS = {
    "xml_style":
    """
Let me check the weather for you.

<tools>
<function>
<name>get_weather</name>
<arguments>{"city": "San Francisco", "unit": "celsius"}</arguments>
</function>
</tools>
""",
    "json_block":
    """
I'll help you with that calculation and weather check.

```tools
[
    {
        "name": "calculate",
        "arguments": {"expression": "25 * 4 + 10"}
    },
    {
        "name": "get_weather",
        "arguments": {"city": "Tokyo", "unit": "fahrenheit"}
    }
]
```
""",
    "bracket_style":
    """
Here's what I found:

[TOOL_CALLS]
[{"name": "search_web", "arguments": {"query": "latest AI news", 
  "num_results": 5}}]
[/TOOL_CALLS]
""",
    "custom_delimiter":
    """
Processing your request...

<<<TOOLS>>>
CALL:send_email({"to": "user@example.com", "subject": "Meeting Tomorrow",
  "body": "Don't forget about our meeting at 2 PM."})
CALL:create_calendar_event({"title": "Team Meeting", "date": "2024-01-10",
  "time": "14:00"})
<<<END>>>
"""
}


def test_integration_xml_style():
    """Test XML-style tool call extraction in real-world scenario"""
    tool_parser = create_regex_tool_parser(INTEGRATION_CONFIGS["xml_style"])
    request = ChatCompletionRequest(messages=[], model="test")

    result = tool_parser.extract_tool_calls(INTEGRATION_OUTPUTS["xml_style"],
                                            request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].function.name == "get_weather"
    assert "San Francisco" in result.tool_calls[0].function.arguments
    assert result.content.strip() == "Let me check the weather for you."


def test_integration_json_block():
    """Test JSON block style tool call extraction"""
    tool_parser = create_regex_tool_parser(INTEGRATION_CONFIGS["json_block"])
    request = ChatCompletionRequest(messages=[], model="test")

    result = tool_parser.extract_tool_calls(INTEGRATION_OUTPUTS["json_block"],
                                            request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "calculate"
    assert result.tool_calls[1].function.name == "get_weather"

    # Verify arguments are valid JSON
    args1 = json.loads(result.tool_calls[0].function.arguments)
    assert args1["expression"] == "25 * 4 + 10"

    args2 = json.loads(result.tool_calls[1].function.arguments)
    assert args2["city"] == "Tokyo"


def test_integration_custom_delimiter():
    """Test custom delimiter style tool call extraction"""
    tool_parser = create_regex_tool_parser(
        INTEGRATION_CONFIGS["custom_delimiter"])
    request = ChatCompletionRequest(messages=[], model="test")

    result = tool_parser.extract_tool_calls(
        INTEGRATION_OUTPUTS["custom_delimiter"], request)

    assert result.tools_called is True
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].function.name == "send_email"
    assert result.tool_calls[1].function.name == "create_calendar_event"

    # Verify email arguments
    email_args = json.loads(result.tool_calls[0].function.arguments)
    assert email_args["to"] == "user@example.com"
    assert email_args["subject"] == "Meeting Tomorrow"


def test_integration_streaming_chunks():
    """Test streaming mode with realistic chunks from integration scenario"""
    tool_parser = create_regex_tool_parser(
        INTEGRATION_CONFIGS["bracket_style"])
    request = ChatCompletionRequest(messages=[], model="test")

    # Simulate streaming chunks
    chunks = [
        "Here's what I found:\n\n[TOOL_CALLS]\n[{\"name\": \"search_",
        "web\", \"arguments\": {\"query\": \"latest AI news\",",
        " \"num_results\": 5}}]\n[/TOOL_CALLS]"
    ]

    previous_text = ""
    previous_tokens: list[int] = []
    all_deltas = []

    for chunk in chunks:
        current_text = previous_text + chunk
        delta_message = tool_parser.extract_tool_calls_streaming(
            previous_text, current_text, chunk, previous_tokens,
            previous_tokens, [], request)
        if delta_message:
            all_deltas.append(delta_message)
        previous_text = current_text

    # Verify we got tool call deltas
    assert len(all_deltas) > 0

    # Check if we got the tool call
    tool_calls_found = any(
        hasattr(delta, 'tool_calls') and delta.tool_calls
        for delta in all_deltas)
    assert tool_calls_found
