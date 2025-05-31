"""
Advanced Plugin System for vLLM Monitoring.

This module provides a sophisticated plugin architecture that enables:
- Easy addition of new components without knowledge of system internals
- Automatic dependency resolution and management
- Runtime hot-reloading and injection
- Hardware/environment compatibility checking
- AI-agent friendly interface for automated plugin creation
- Comprehensive error handling and feedback
"""

import os
import sys
import json
import time
import inspect
import importlib
import importlib.util
import threading
import hashlib
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Callable, Type, Tuple, Union
from enum import Enum, auto
from collections import defaultdict, OrderedDict
import logging
import traceback
import weakref
from concurrent.futures import ThreadPoolExecutor, Future
import yaml
import ast
import re

# Import core components
from .core import get_logger, CircularBuffer
from .lifecycle_tracker import LifecycleState, GuardrailPolicy


class PluginType(Enum):
    """Types of plugins supported by the system"""
    COLLECTOR = auto()
    ANALYZER = auto()
    INTERVENTION = auto()
    GUARDRAIL = auto()
    EXPORTER = auto()
    STATE_HANDLER = auto()
    COMPONENT = auto()
    EXTENSION = auto()


class PluginStatus(Enum):
    """Plugin lifecycle status"""
    NOT_LOADED = auto()
    LOADING = auto()
    LOADED = auto()
    ACTIVE = auto()
    INACTIVE = auto()
    ERROR = auto()
    UPDATING = auto()
    UNLOADING = auto()


@dataclass
class PluginMetadata:
    """Complete plugin metadata and configuration"""
    name: str
    version: str
    type: PluginType
    description: str
    author: str = "Unknown"
    dependencies: List[str] = field(default_factory=list)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    environment_requirements: Dict[str, str] = field(default_factory=dict)
    configuration: Dict[str, Any] = field(default_factory=dict)
    entry_points: Dict[str, str] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    priority: int = 100
    auto_enable: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = {
            "name": self.name,
            "version": self.version,
            "type": self.type.name,
            "description": self.description,
            "author": self.author,
            "dependencies": self.dependencies,
            "hardware_requirements": self.hardware_requirements,
            "environment_requirements": self.environment_requirements,
            "configuration": self.configuration,
            "entry_points": self.entry_points,
            "capabilities": self.capabilities,
            "conflicts": self.conflicts,
            "priority": self.priority,
            "auto_enable": self.auto_enable
        }
        return data


@dataclass
class PluginValidationResult:
    """Result of plugin validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_dependencies: List[str] = field(default_factory=list)
    hardware_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


class PluginInterface(ABC):
    """
    Base interface for all plugins.
    
    This abstract base class defines the minimal interface that all plugins
    must implement. It's designed to be simple enough for AI agents to understand
    and implement correctly.
    """
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    def initialize(self, context: Dict[str, Any]) -> bool:
        """
        Initialize the plugin with given context.
        
        Args:
            context: Runtime context including configuration and dependencies
            
        Returns:
            True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the plugin's main functionality.
        
        The signature varies by plugin type:
        - Collectors: execute() -> Dict[str, Any]
        - Analyzers: execute(data: Dict) -> Dict[str, Any]
        - Interventions: execute(context: Dict) -> bool
        - Guardrails: execute(checkpoint: Any) -> Tuple[bool, Optional[str]]
        """
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources before unloading"""
        pass
    
    def validate(self) -> PluginValidationResult:
        """Validate plugin configuration and requirements"""
        return PluginValidationResult(valid=True)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current plugin status"""
        return {"status": "active"}


class SimplePlugin(PluginInterface):
    """
    Simple plugin base class for easy implementation.
    
    This class provides sensible defaults and makes it extremely easy
    to create new plugins, even for AI agents.
    """
    
    def __init__(self, 
                 name: str,
                 plugin_type: PluginType,
                 execute_func: Callable,
                 description: str = "",
                 **metadata_kwargs):
        self.name = name
        self.plugin_type = plugin_type
        self.execute_func = execute_func
        self.description = description
        self.metadata_kwargs = metadata_kwargs
        self._initialized = False
        self._context = {}
    
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        return PluginMetadata(
            name=self.name,
            version=self.metadata_kwargs.get("version", "1.0.0"),
            type=self.plugin_type,
            description=self.description,
            **self.metadata_kwargs
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the plugin"""
        self._context = context
        self._initialized = True
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute the plugin function"""
        if not self._initialized:
            raise RuntimeError(f"Plugin {self.name} not initialized")
        return self.execute_func(*args, **kwargs)
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self._initialized = False
        self._context = {}


class PluginRegistry:
    """
    Central registry for all plugins with dependency management.
    
    Features:
    - Automatic dependency resolution
    - Conflict detection and resolution
    - Hardware compatibility checking
    - Plugin versioning support
    """
    
    def __init__(self):
        self.logger = get_logger()
        self._lock = threading.RLock()
        
        # Plugin storage
        self.plugins: Dict[str, PluginInterface] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        self.status: Dict[str, PluginStatus] = {}
        
        # Dependency graph
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
        # Type-based indexing
        self.plugins_by_type: Dict[PluginType, Set[str]] = defaultdict(set)
        
        # Capability indexing
        self.plugins_by_capability: Dict[str, Set[str]] = defaultdict(set)
        
        # Version tracking
        self.plugin_versions: Dict[str, List[str]] = defaultdict(list)
        
        # Conflict tracking
        self.conflicts: Dict[str, Set[str]] = defaultdict(set)
        
        # Load order for dependency resolution
        self.load_order: List[str] = []
        
        self.logger.info("Plugin registry initialized")
    
    def register(self, plugin: PluginInterface) -> bool:
        """
        Register a new plugin.
        
        Args:
            plugin: Plugin instance to register
            
        Returns:
            True if registration successful
        """
        with self._lock:
            try:
                metadata = plugin.get_metadata()
                
                # Validate plugin
                validation = self._validate_plugin(plugin, metadata)
                if not validation.valid:
                    self._emit_validation_feedback(metadata.name, validation)
                    return False
                
                # Check for conflicts
                if self._check_conflicts(metadata):
                    return False
                
                # Add to registry
                self.plugins[metadata.name] = plugin
                self.metadata[metadata.name] = metadata
                self.status[metadata.name] = PluginStatus.LOADED
                
                # Update indexes
                self.plugins_by_type[metadata.type].add(metadata.name)
                for capability in metadata.capabilities:
                    self.plugins_by_capability[capability].add(metadata.name)
                
                # Update dependency graph
                self._update_dependency_graph(metadata)
                
                # Update version tracking
                self.plugin_versions[metadata.name].append(metadata.version)
                
                self.logger.info(f"Plugin '{metadata.name}' registered successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to register plugin: {e}")
                return False
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name
            
        Returns:
            True if unregistration successful
        """
        with self._lock:
            if name not in self.plugins:
                return False
            
            # Check reverse dependencies
            if self.reverse_dependencies[name]:
                dependent_plugins = list(self.reverse_dependencies[name])
                self.logger.warning(
                    f"Cannot unregister '{name}': required by {dependent_plugins}"
                )
                return False
            
            # Cleanup plugin
            try:
                self.plugins[name].cleanup()
            except Exception as e:
                self.logger.error(f"Error during plugin cleanup: {e}")
            
            # Remove from registry
            metadata = self.metadata[name]
            del self.plugins[name]
            del self.metadata[name]
            del self.status[name]
            
            # Update indexes
            self.plugins_by_type[metadata.type].discard(name)
            for capability in metadata.capabilities:
                self.plugins_by_capability[capability].discard(name)
            
            # Update dependency graph
            for dep in self.dependency_graph[name]:
                self.reverse_dependencies[dep].discard(name)
            del self.dependency_graph[name]
            
            self.logger.info(f"Plugin '{name}' unregistered")
            return True
    
    def get_plugin(self, name: str) -> Optional[PluginInterface]:
        """Get a plugin by name"""
        return self.plugins.get(name)
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[PluginInterface]:
        """Get all plugins of a specific type"""
        with self._lock:
            names = self.plugins_by_type.get(plugin_type, set())
            return [self.plugins[name] for name in names if name in self.plugins]
    
    def get_plugins_by_capability(self, capability: str) -> List[PluginInterface]:
        """Get all plugins with a specific capability"""
        with self._lock:
            names = self.plugins_by_capability.get(capability, set())
            return [self.plugins[name] for name in names if name in self.plugins]
    
    def resolve_dependencies(self, name: str) -> List[str]:
        """
        Resolve plugin dependencies in correct order.
        
        Args:
            name: Plugin name
            
        Returns:
            Ordered list of plugin names to load
        """
        with self._lock:
            if name not in self.metadata:
                return []
            
            # Topological sort for dependency resolution
            visited = set()
            stack = []
            
            def visit(plugin_name: str):
                if plugin_name in visited:
                    return
                visited.add(plugin_name)
                
                # Visit dependencies first
                if plugin_name in self.metadata:
                    for dep in self.metadata[plugin_name].dependencies:
                        if dep in self.metadata:
                            visit(dep)
                
                stack.append(plugin_name)
            
            visit(name)
            return stack
    
    def _validate_plugin(self, 
                        plugin: PluginInterface, 
                        metadata: PluginMetadata) -> PluginValidationResult:
        """Validate a plugin before registration"""
        result = PluginValidationResult(valid=True)
        
        # Check metadata completeness
        if not metadata.name:
            result.errors.append("Plugin name is required")
            result.valid = False
        
        if not metadata.version:
            result.errors.append("Plugin version is required")
            result.valid = False
        
        # Check dependencies
        for dep in metadata.dependencies:
            if dep not in self.plugins:
                result.missing_dependencies.append(dep)
                result.warnings.append(f"Dependency '{dep}' not found")
        
        # Check hardware requirements
        if metadata.hardware_requirements:
            hw_check = self._check_hardware_requirements(metadata.hardware_requirements)
            if not hw_check["satisfied"]:
                result.hardware_issues.extend(hw_check["issues"])
                result.warnings.extend(hw_check["issues"])
        
        # Run plugin's own validation
        try:
            plugin_validation = plugin.validate()
            result.errors.extend(plugin_validation.errors)
            result.warnings.extend(plugin_validation.warnings)
            if plugin_validation.errors:
                result.valid = False
        except Exception as e:
            result.errors.append(f"Plugin validation failed: {str(e)}")
            result.valid = False
        
        return result
    
    def _check_conflicts(self, metadata: PluginMetadata) -> bool:
        """Check for plugin conflicts"""
        for conflict in metadata.conflicts:
            if conflict in self.plugins:
                self.logger.error(
                    f"Plugin '{metadata.name}' conflicts with '{conflict}'"
                )
                return True
        return False
    
    def _update_dependency_graph(self, metadata: PluginMetadata) -> None:
        """Update dependency tracking"""
        for dep in metadata.dependencies:
            self.dependency_graph[metadata.name].add(dep)
            self.reverse_dependencies[dep].add(metadata.name)
    
    def _check_hardware_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Check if hardware requirements are satisfied"""
        result = {"satisfied": True, "issues": []}
        
        # Example hardware checks
        if "gpu" in requirements:
            try:
                import torch
                if requirements["gpu"] and not torch.cuda.is_available():
                    result["satisfied"] = False
                    result["issues"].append("GPU required but not available")
            except ImportError:
                if requirements["gpu"]:
                    result["satisfied"] = False
                    result["issues"].append("PyTorch not installed for GPU check")
        
        if "min_memory_gb" in requirements:
            try:
                import psutil
                available_gb = psutil.virtual_memory().available / (1024**3)
                if available_gb < requirements["min_memory_gb"]:
                    result["satisfied"] = False
                    result["issues"].append(
                        f"Insufficient memory: {available_gb:.1f}GB < "
                        f"{requirements['min_memory_gb']}GB required"
                    )
            except ImportError:
                result["issues"].append("Cannot check memory requirements")
        
        return result
    
    def _emit_validation_feedback(self, 
                                 plugin_name: str, 
                                 validation: PluginValidationResult) -> None:
        """Emit detailed validation feedback"""
        if validation.errors:
            self.logger.error(f"Plugin '{plugin_name}' validation errors: {validation.errors}")
        if validation.warnings:
            self.logger.warning(f"Plugin '{plugin_name}' warnings: {validation.warnings}")
        if validation.missing_dependencies:
            self.logger.info(
                f"Plugin '{plugin_name}' missing dependencies: "
                f"{validation.missing_dependencies}"
            )
        if validation.suggestions:
            self.logger.info(f"Suggestions for '{plugin_name}': {validation.suggestions}")


class PluginLoader:
    """
    Dynamic plugin loader with hot-reload support.
    
    Features:
    - Load plugins from files, directories, or code strings
    - Hot-reload support with file watching
    - Safe sandboxed execution
    - AI-agent friendly interface
    """
    
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self.logger = get_logger()
        self._lock = threading.Lock()
        
        # Plugin sources
        self.plugin_dirs: List[Path] = []
        self.loaded_modules: Dict[str, Any] = {}
        self.file_checksums: Dict[str, str] = {}
        
        # File watcher
        self._watcher_thread: Optional[threading.Thread] = None
        self._watching = False
        self.watch_interval = 1.0  # seconds
        
        # Execution sandbox
        self.sandbox_globals = self._create_sandbox_globals()
    
    def add_plugin_directory(self, directory: Union[str, Path]) -> None:
        """Add a directory to scan for plugins"""
        path = Path(directory)
        if path.exists() and path.is_dir():
            self.plugin_dirs.append(path)
            self.logger.info(f"Added plugin directory: {path}")
    
    def load_plugin_from_file(self, filepath: Union[str, Path]) -> bool:
        """
        Load a plugin from a Python file.
        
        Args:
            filepath: Path to the plugin file
            
        Returns:
            True if loading successful
        """
        filepath = Path(filepath)
        if not filepath.exists():
            self.logger.error(f"Plugin file not found: {filepath}")
            return False
        
        try:
            # Calculate checksum for change detection
            checksum = self._calculate_checksum(filepath)
            self.file_checksums[str(filepath)] = checksum
            
            # Load module
            spec = importlib.util.spec_from_file_location(
                f"plugin_{filepath.stem}", 
                filepath
            )
            if not spec or not spec.loader:
                return False
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find and register plugins
            plugins_found = self._extract_plugins_from_module(module)
            
            if plugins_found:
                self.loaded_modules[str(filepath)] = module
                self.logger.info(f"Loaded {len(plugins_found)} plugins from {filepath}")
                return True
            else:
                self.logger.warning(f"No plugins found in {filepath}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin from {filepath}: {e}")
            return False
    
    def load_plugin_from_code(self, 
                             code: str, 
                             plugin_name: str = "dynamic_plugin") -> bool:
        """
        Load a plugin from a code string.
        
        This is particularly useful for AI agents that generate plugin code.
        
        Args:
            code: Python code defining the plugin
            plugin_name: Name for the dynamic plugin
            
        Returns:
            True if loading successful
        """
        try:
            # Create a safe execution environment
            local_vars = {}
            exec(code, self.sandbox_globals, local_vars)
            
            # Find plugin classes or functions
            plugins_found = []
            for name, obj in local_vars.items():
                if self._is_plugin_object(obj):
                    plugin = self._create_plugin_wrapper(obj, plugin_name)
                    if plugin and self.registry.register(plugin):
                        plugins_found.append(plugin)
            
            if plugins_found:
                self.logger.info(
                    f"Loaded {len(plugins_found)} plugins from code string"
                )
                return True
            else:
                self.logger.warning("No valid plugins found in code")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load plugin from code: {e}")
            self.logger.debug(f"Code that failed:\n{code}")
            return False
    
    def load_plugin_from_config(self, config: Dict[str, Any]) -> bool:
        """
        Load a plugin from a configuration dictionary.
        
        This allows creating simple plugins without writing full classes.
        
        Args:
            config: Plugin configuration
            
        Returns:
            True if loading successful
        """
        try:
            # Validate required fields
            required = ["name", "type", "execute"]
            if not all(field in config for field in required):
                self.logger.error(f"Missing required fields: {required}")
                return False
            
            # Create plugin from config
            plugin_type = PluginType[config["type"].upper()]
            
            # Create execution function
            if isinstance(config["execute"], str):
                # Code string
                exec_func = self._create_function_from_string(
                    config["execute"],
                    config.get("execute_args", [])
                )
            else:
                # Assume it's already a callable
                exec_func = config["execute"]
            
            # Create plugin
            plugin = SimplePlugin(
                name=config["name"],
                plugin_type=plugin_type,
                execute_func=exec_func,
                description=config.get("description", ""),
                version=config.get("version", "1.0.0"),
                dependencies=config.get("dependencies", []),
                hardware_requirements=config.get("hardware_requirements", {}),
                capabilities=config.get("capabilities", [])
            )
            
            return self.registry.register(plugin)
            
        except Exception as e:
            self.logger.error(f"Failed to load plugin from config: {e}")
            return False
    
    def scan_and_load_plugins(self) -> int:
        """
        Scan all plugin directories and load plugins.
        
        Returns:
            Number of plugins loaded
        """
        loaded_count = 0
        
        for plugin_dir in self.plugin_dirs:
            for file_path in plugin_dir.glob("*.py"):
                if file_path.name.startswith("_"):
                    continue  # Skip private modules
                    
                if self.load_plugin_from_file(file_path):
                    loaded_count += 1
            
            # Also check for YAML/JSON configs
            for config_path in plugin_dir.glob("*.yaml"):
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    if self.load_plugin_from_config(config):
                        loaded_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to load config {config_path}: {e}")
        
        return loaded_count
    
    def start_watching(self) -> None:
        """Start watching plugin directories for changes"""
        if self._watching:
            return
            
        self._watching = True
        self._watcher_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True
        )
        self._watcher_thread.start()
        self.logger.info("Started plugin file watcher")
    
    def stop_watching(self) -> None:
        """Stop watching plugin directories"""
        self._watching = False
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5)
        self.logger.info("Stopped plugin file watcher")
    
    def _watch_loop(self) -> None:
        """File watcher loop"""
        while self._watching:
            try:
                for filepath, old_checksum in list(self.file_checksums.items()):
                    path = Path(filepath)
                    if not path.exists():
                        continue
                        
                    new_checksum = self._calculate_checksum(path)
                    if new_checksum != old_checksum:
                        self.logger.info(f"Detected change in {filepath}")
                        self._reload_plugin_file(path)
                        
            except Exception as e:
                self.logger.error(f"Error in file watcher: {e}")
            
            time.sleep(self.watch_interval)
    
    def _reload_plugin_file(self, filepath: Path) -> None:
        """Reload a plugin file"""
        # First unregister old plugins from this file
        if str(filepath) in self.loaded_modules:
            module = self.loaded_modules[str(filepath)]
            # Find and unregister plugins from this module
            for name, obj in inspect.getmembers(module):
                if hasattr(obj, "__plugin_name__"):
                    self.registry.unregister(obj.__plugin_name__)
        
        # Reload the file
        self.load_plugin_from_file(filepath)
    
    def _extract_plugins_from_module(self, module: Any) -> List[PluginInterface]:
        """Extract plugin objects from a module"""
        plugins = []
        
        for name, obj in inspect.getmembers(module):
            if self._is_plugin_object(obj):
                # Create instance if it's a class
                if inspect.isclass(obj):
                    try:
                        instance = obj()
                        if self.registry.register(instance):
                            plugins.append(instance)
                            # Mark for tracking
                            instance.__plugin_name__ = instance.get_metadata().name
                    except Exception as e:
                        self.logger.error(f"Failed to instantiate {name}: {e}")
                elif hasattr(obj, 'get_metadata'):
                    # Already an instance
                    if self.registry.register(obj):
                        plugins.append(obj)
                        obj.__plugin_name__ = obj.get_metadata().name
        
        return plugins
    
    def _is_plugin_object(self, obj: Any) -> bool:
        """Check if an object is a valid plugin"""
        if inspect.isclass(obj):
            return issubclass(obj, PluginInterface) and obj != PluginInterface
        else:
            return isinstance(obj, PluginInterface)
    
    def _create_plugin_wrapper(self, obj: Any, name: str) -> Optional[PluginInterface]:
        """Create a plugin wrapper for various object types"""
        if callable(obj) and not inspect.isclass(obj):
            # Wrap a function as a plugin
            return SimplePlugin(
                name=name,
                plugin_type=PluginType.COMPONENT,
                execute_func=obj,
                description=obj.__doc__ or "Dynamic plugin"
            )
        return None
    
    def _create_function_from_string(self, 
                                   code: str, 
                                   args: List[str]) -> Callable:
        """Create a function from a code string"""
        # Wrap code in a function if needed
        if not code.strip().startswith("def "):
            func_code = f"def execute({', '.join(args)}):\n"
            func_code += "\n".join(f"    {line}" for line in code.split("\n"))
        else:
            func_code = code
        
        local_vars = {}
        exec(func_code, self.sandbox_globals, local_vars)
        
        # Find the function
        for name, obj in local_vars.items():
            if callable(obj):
                return obj
        
        raise ValueError("No function found in code")
    
    def _create_sandbox_globals(self) -> Dict[str, Any]:
        """Create a safe global environment for plugin execution"""
        return {
            "__builtins__": {
                # Safe built-ins only
                "len": len,
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "sum": sum,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round,
                "all": all,
                "any": any,
                "dict": dict,
                "list": list,
                "tuple": tuple,
                "set": set,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "print": print,
                "isinstance": isinstance,
                "hasattr": hasattr,
                "getattr": getattr,
                "setattr": setattr,
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
            },
            # Import common modules
            "time": time,
            "json": json,
            "re": re,
            "math": __import__("math"),
            "collections": __import__("collections"),
            "itertools": __import__("itertools"),
            # Plugin system imports
            "PluginInterface": PluginInterface,
            "SimplePlugin": SimplePlugin,
            "PluginType": PluginType,
            "PluginMetadata": PluginMetadata,
            "GuardrailPolicy": GuardrailPolicy,
            "LifecycleState": LifecycleState,
        }
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate file checksum for change detection"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()


class PluginManager:
    """
    High-level plugin manager with simplified interface.
    
    This class provides an easy-to-use interface for plugin management,
    designed to be simple enough for AI agents to use effectively.
    """
    
    def __init__(self):
        self.logger = get_logger()
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)
        
        # Message bus for inter-plugin communication
        self.message_bus = MessageBus()
        
        # Plugin execution context
        self.context = {
            "registry": self.registry,
            "message_bus": self.message_bus,
            "logger": self.logger
        }
        
        # Feedback handlers
        self.feedback_handlers: List[Callable] = []
        
        # Plugin templates for AI agents
        self.templates = PluginTemplates()
        
        # Initialize default plugin directory
        default_dir = Path(__file__).parent / "plugins"
        if not default_dir.exists():
            default_dir.mkdir(exist_ok=True)
        self.loader.add_plugin_directory(default_dir)
    
    def create_plugin(self,
                     name: str,
                     plugin_type: str,
                     execute_code: str,
                     description: str = "",
                     **kwargs) -> bool:
        """
        Create a new plugin from code.
        
        This is the simplest interface for AI agents to create plugins.
        
        Args:
            name: Plugin name
            plugin_type: Type of plugin (collector, analyzer, etc.)
            execute_code: Python code for the execute function
            description: Plugin description
            **kwargs: Additional metadata
            
        Returns:
            True if plugin created successfully
        """
        try:
            config = {
                "name": name,
                "type": plugin_type,
                "execute": execute_code,
                "description": description,
                **kwargs
            }
            
            success = self.loader.load_plugin_from_config(config)
            
            if success:
                self._emit_feedback(
                    "success",
                    f"Plugin '{name}' created successfully",
                    {"type": plugin_type}
                )
            else:
                self._emit_feedback(
                    "error",
                    f"Failed to create plugin '{name}'",
                    {"type": plugin_type}
                )
            
            return success
            
        except Exception as e:
            self._emit_feedback(
                "error",
                f"Exception creating plugin '{name}': {str(e)}",
                {"type": plugin_type, "error": str(e)}
            )
            return False
    
    def create_guardrail(self,
                        name: str,
                        description: str,
                        condition_code: str,
                        intervention: Optional[str] = None,
                        severity: str = "warning") -> bool:
        """
        Create a new guardrail plugin.
        
        Args:
            name: Guardrail name
            description: What the guardrail checks
            condition_code: Python code that returns True if guardrail triggered
            intervention: Name of intervention to execute
            severity: warning, error, or critical
            
        Returns:
            True if guardrail created successfully
        """
        # Create guardrail function
        guardrail_code = f"""
def check_condition(checkpoint):
    {condition_code}
"""
        
        try:
            # Create guardrail plugin
            config = {
                "name": f"guardrail_{name}",
                "type": "guardrail",
                "execute": guardrail_code,
                "execute_args": ["checkpoint"],
                "description": description,
                "capabilities": ["guardrail", name],
                "configuration": {
                    "intervention": intervention,
                    "severity": severity
                }
            }
            
            return self.loader.load_plugin_from_config(config)
            
        except Exception as e:
            self.logger.error(f"Failed to create guardrail '{name}': {e}")
            return False
    
    def create_intervention(self,
                           name: str,
                           description: str,
                           action_code: str,
                           required_context: List[str] = None) -> bool:
        """
        Create a new intervention plugin.
        
        Args:
            name: Intervention name
            description: What the intervention does
            action_code: Python code to execute the intervention
            required_context: List of required context keys
            
        Returns:
            True if intervention created successfully
        """
        intervention_code = f"""
def execute_intervention(context):
    # Validate required context
    required = {required_context or []}
    for key in required:
        if key not in context:
            raise ValueError(f"Missing required context: {{key}}")
    
    {action_code}
    
    return True  # Return True if successful
"""
        
        config = {
            "name": f"intervention_{name}",
            "type": "intervention",
            "execute": intervention_code,
            "execute_args": ["context"],
            "description": description,
            "capabilities": ["intervention", name]
        }
        
        return self.loader.load_plugin_from_config(config)
    
    def list_plugins(self, plugin_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all registered plugins.
        
        Args:
            plugin_type: Filter by type (optional)
            
        Returns:
            List of plugin information dictionaries
        """
        plugins = []
        
        if plugin_type:
            try:
                ptype = PluginType[plugin_type.upper()]
                plugin_list = self.registry.get_plugins_by_type(ptype)
            except KeyError:
                self.logger.warning(f"Unknown plugin type: {plugin_type}")
                return []
        else:
            plugin_list = list(self.registry.plugins.values())
        
        for plugin in plugin_list:
            metadata = plugin.get_metadata()
            status = self.registry.status.get(metadata.name, PluginStatus.UNKNOWN)
            
            plugins.append({
                "name": metadata.name,
                "type": metadata.type.name,
                "version": metadata.version,
                "description": metadata.description,
                "status": status.name,
                "capabilities": metadata.capabilities,
                "dependencies": metadata.dependencies
            })
        
        return plugins
    
    def execute_plugin(self, 
                      name: str, 
                      *args, 
                      **kwargs) -> Tuple[bool, Any]:
        """
        Execute a plugin by name.
        
        Args:
            name: Plugin name
            *args, **kwargs: Arguments for the plugin
            
        Returns:
            Tuple of (success, result)
        """
        try:
            plugin = self.registry.get_plugin(name)
            if not plugin:
                return False, f"Plugin '{name}' not found"
            
            # Initialize if needed
            if self.registry.status.get(name) != PluginStatus.ACTIVE:
                if not plugin.initialize(self.context):
                    return False, f"Plugin '{name}' initialization failed"
                self.registry.status[name] = PluginStatus.ACTIVE
            
            # Execute
            result = plugin.execute(*args, **kwargs)
            return True, result
            
        except Exception as e:
            self.logger.error(f"Plugin '{name}' execution failed: {e}")
            return False, str(e)
    
    def get_plugin_template(self, plugin_type: str) -> str:
        """
        Get a template for creating a new plugin.
        
        This helps AI agents understand the structure needed.
        
        Args:
            plugin_type: Type of plugin
            
        Returns:
            Template code string
        """
        return self.templates.get_template(plugin_type)
    
    def validate_plugin_code(self, code: str) -> PluginValidationResult:
        """
        Validate plugin code before loading.
        
        Args:
            code: Python code to validate
            
        Returns:
            Validation result
        """
        result = PluginValidationResult(valid=True)
        
        try:
            # Parse the code
            ast.parse(code)
        except SyntaxError as e:
            result.valid = False
            result.errors.append(f"Syntax error: {e}")
            return result
        
        # Check for dangerous operations
        dangerous_patterns = [
            r'\b__import__\b',
            r'\beval\b',
            r'\bexec\b',
            r'\bopen\b',
            r'\bfile\b',
            r'\bsubprocess\b',
            r'\bos\.',
            r'\bsys\.',
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                result.warnings.append(
                    f"Potentially dangerous operation detected: {pattern}"
                )
        
        return result
    
    def add_feedback_handler(self, handler: Callable) -> None:
        """Add a feedback handler"""
        self.feedback_handlers.append(handler)
    
    def _emit_feedback(self, level: str, message: str, context: Dict[str, Any]) -> None:
        """Emit feedback to all handlers"""
        for handler in self.feedback_handlers:
            try:
                handler(level, message, context)
            except Exception as e:
                self.logger.error(f"Feedback handler error: {e}")


class MessageBus:
    """Simple message bus for inter-plugin communication"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def subscribe(self, topic: str, callback: Callable) -> None:
        """Subscribe to a topic"""
        with self._lock:
            self._subscribers[topic].append(callback)
    
    def publish(self, topic: str, message: Any) -> None:
        """Publish a message to a topic"""
        with self._lock:
            for callback in self._subscribers[topic]:
                try:
                    callback(message)
                except Exception as e:
                    logging.error(f"Message handler error: {e}")


class PluginTemplates:
    """Templates for different plugin types"""
    
    def get_template(self, plugin_type: str) -> str:
        """Get template code for a plugin type"""
        templates = {
            "collector": """
# Collector plugin template
# Collectors gather data from various sources

def collect_data():
    '''Collect and return data'''
    data = {
        'timestamp': time.time(),
        'metric1': 42,
        'metric2': 'value'
    }
    return data

# The function should return a dictionary of collected data
""",
            "analyzer": """
# Analyzer plugin template
# Analyzers process data and detect patterns

def analyze_data(data):
    '''Analyze data and return insights'''
    insights = {
        'anomalies': [],
        'trends': [],
        'warnings': []
    }
    
    # Add your analysis logic here
    if data.get('metric1', 0) > 100:
        insights['warnings'].append('Metric1 exceeds threshold')
    
    return insights

# The function receives data and returns analysis results
""",
            "intervention": """
# Intervention plugin template
# Interventions take corrective actions

def execute_intervention(context):
    '''Execute intervention based on context'''
    # Access context data
    component = context.get('component')
    issue = context.get('issue')
    
    # Take corrective action
    # Example: restart component, adjust settings, etc.
    
    # Return True if successful, False otherwise
    return True

# The function receives context and returns success status
""",
            "guardrail": """
# Guardrail plugin template
# Guardrails monitor conditions and trigger interventions

def check_condition(checkpoint):
    '''Check if guardrail condition is met'''
    # Access checkpoint data
    metrics = checkpoint.metrics
    state = checkpoint.state
    
    # Define your condition
    # Return True if guardrail should trigger
    if metrics.get('error_rate', 0) > 0.1:
        return True
    
    return False

# The function receives a checkpoint and returns True/False
""",
            "exporter": """
# Exporter plugin template
# Exporters send data to external systems

def export_data(data, config):
    '''Export data to external system'''
    # Access configuration
    endpoint = config.get('endpoint')
    format = config.get('format', 'json')
    
    # Format and send data
    # Example: HTTP POST, write to file, etc.
    
    # Return export status
    return {
        'success': True,
        'records_exported': len(data),
        'destination': endpoint
    }

# The function receives data and config, returns status
"""
        }
        
        return templates.get(plugin_type, "# Unknown plugin type")


# Example usage for AI agents
"""
# Creating a simple memory monitoring guardrail
manager = PluginManager()

# Create guardrail
manager.create_guardrail(
    name="high_memory_usage",
    description="Triggers when memory usage exceeds 90%",
    condition_code="return checkpoint.metrics.get('memory_percent', 0) > 90",
    intervention="cleanup_memory",
    severity="warning"
)

# Create intervention
manager.create_intervention(
    name="cleanup_memory",
    description="Perform garbage collection to free memory",
    action_code='''
import gc
gc.collect()
print("Memory cleanup performed")
''',
    required_context=[]
)

# List all plugins
plugins = manager.list_plugins()
for plugin in plugins:
    print(f"{plugin['name']} ({plugin['type']}): {plugin['description']}")
"""