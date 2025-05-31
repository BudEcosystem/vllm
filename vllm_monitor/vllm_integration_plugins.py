"""
Comprehensive vLLM Integration Plugins for All States and Exceptions.

This module provides exhaustive monitoring and intervention plugins that
integrate deeply with vLLM's core components to track every state,
catch every exception, and provide mitigation strategies.
"""

import os
import sys
import psutil
import subprocess
import time
import json
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass
import logging

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

from .plugin_system import PluginInterface, PluginMetadata, PluginType, SimplePlugin
from .lifecycle_tracker import LifecycleState, StateTransition, GuardrailPolicy
from .predictive_failure_detection import MitigationStrategy, MitigationOutcome
from .continuous_learning import MitigationAttempt


# ============================================================================
# Pre-Startup Validation Plugins
# ============================================================================

class PreStartupValidator(PluginInterface):
    """Comprehensive pre-startup validation plugin"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="pre_startup_validator",
            version="1.0.0",
            type=PluginType.COMPONENT,
            description="Validates system before vLLM startup",
            capabilities=["pre_startup", "validation", "configuration"],
            auto_enable=True
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        self.logger = context.get("logger", logging.getLogger())
        self.context = context
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Execute comprehensive pre-startup validation"""
        results = {
            "timestamp": time.time(),
            "validations": {},
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # 1. OS and Driver Validation
        os_validation = self._validate_os_and_drivers()
        results["validations"]["os_and_drivers"] = os_validation
        
        # 2. Hardware Validation
        hw_validation = self._validate_hardware()
        results["validations"]["hardware"] = hw_validation
        
        # 3. CUDA/ROCm Validation
        gpu_validation = self._validate_gpu_environment()
        results["validations"]["gpu_environment"] = gpu_validation
        
        # 4. Memory Validation
        mem_validation = self._validate_memory_requirements()
        results["validations"]["memory"] = mem_validation
        
        # 5. Network Validation
        net_validation = self._validate_network_config()
        results["validations"]["network"] = net_validation
        
        # 6. Environment Variables
        env_validation = self._validate_environment_variables()
        results["validations"]["environment"] = env_validation
        
        # 7. Dependencies Validation
        dep_validation = self._validate_dependencies()
        results["validations"]["dependencies"] = dep_validation
        
        # 8. File System Validation
        fs_validation = self._validate_filesystem()
        results["validations"]["filesystem"] = fs_validation
        
        # Aggregate results
        for validation in results["validations"].values():
            results["errors"].extend(validation.get("errors", []))
            results["warnings"].extend(validation.get("warnings", []))
            results["recommendations"].extend(validation.get("recommendations", []))
        
        results["passed"] = len(results["errors"]) == 0
        
        return results
    
    def cleanup(self):
        pass
    
    def _validate_os_and_drivers(self) -> Dict[str, Any]:
        """Validate OS and driver configurations"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check OS
        os_info = {
            "system": os.uname().system,
            "release": os.uname().release,
            "version": os.uname().version
        }
        
        # Check for recommended OS
        if os_info["system"] != "Linux":
            result["warnings"].append(f"Non-Linux OS detected: {os_info['system']}")
            result["recommendations"].append("vLLM performs best on Linux systems")
        
        # Check NVIDIA driver
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                driver_version = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True
                ).strip()
                
                # Check driver version
                major_version = int(driver_version.split('.')[0])
                if major_version < 525:  # Minimum for H100
                    result["warnings"].append(f"Outdated NVIDIA driver: {driver_version}")
                    result["recommendations"].append("Update to NVIDIA driver 525.60 or newer")
                
            except subprocess.CalledProcessError:
                result["errors"].append("Failed to query NVIDIA driver version")
        
        # Check kernel parameters
        kernel_params_to_check = {
            "vm.overcommit_memory": "1",  # Allow memory overcommit
            "net.ipv4.tcp_retries2": "5",  # Reduce TCP retries for faster failure detection
        }
        
        for param, expected in kernel_params_to_check.items():
            try:
                actual = subprocess.check_output(
                    ["sysctl", "-n", param],
                    text=True
                ).strip()
                
                if actual != expected:
                    result["warnings"].append(
                        f"Kernel parameter {param}={actual}, recommended: {expected}"
                    )
                    result["recommendations"].append(
                        f"Set {param}={expected} for optimal performance"
                    )
            except:
                pass
        
        result["passed"] = len(result["errors"]) == 0
        return result
    
    def _validate_hardware(self) -> Dict[str, Any]:
        """Validate hardware configuration"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": [],
            "hardware_info": {}
        }
        
        # CPU validation
        cpu_count = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        result["hardware_info"]["cpu"] = {
            "physical_cores": cpu_count,
            "logical_cores": cpu_threads
        }
        
        if cpu_count < 4:
            result["warnings"].append(f"Low CPU core count: {cpu_count}")
            result["recommendations"].append("Consider using a system with at least 8 CPU cores")
        
        # GPU validation
        if TORCH_AVAILABLE and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            result["hardware_info"]["gpu_count"] = gpu_count
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                gpu_info = {
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "memory_gb": props.total_memory / (1024**3),
                    "sm_count": props.multi_processor_count
                }
                result["hardware_info"][f"gpu_{i}"] = gpu_info
                
                # Check compute capability
                if props.major < 7:
                    result["warnings"].append(
                        f"GPU {i} has compute capability {props.major}.{props.minor}, "
                        "which may not support all optimizations"
                    )
                
                # Check memory
                if gpu_info["memory_gb"] < 16:
                    result["warnings"].append(
                        f"GPU {i} has limited memory: {gpu_info['memory_gb']:.1f}GB"
                    )
        else:
            result["errors"].append("No CUDA-capable GPU detected")
        
        # Check PCIe/NVLink
        if gpu_count > 1:
            # Check for NVLink
            try:
                nvlink_status = subprocess.check_output(
                    ["nvidia-smi", "nvlink", "-s"],
                    text=True
                )
                if "Link" in nvlink_status:
                    result["hardware_info"]["interconnect"] = "NVLink"
                else:
                    result["hardware_info"]["interconnect"] = "PCIe"
                    result["recommendations"].append(
                        "Consider using NVLink for better multi-GPU performance"
                    )
            except:
                pass
        
        result["passed"] = len(result["errors"]) == 0
        return result
    
    def _validate_gpu_environment(self) -> Dict[str, Any]:
        """Validate GPU environment (CUDA/ROCm)"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # CUDA validation
        if TORCH_AVAILABLE:
            # Check CUDA version
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                result["cuda_version"] = cuda_version
                
                # Check for compatible CUDA version
                major, minor = map(int, cuda_version.split('.')[:2])
                if major < 11 or (major == 11 and minor < 7):
                    result["warnings"].append(f"CUDA {cuda_version} is outdated")
                    result["recommendations"].append("Update to CUDA 11.7 or newer")
                
                # Check cuDNN
                if hasattr(torch.backends.cudnn, 'version'):
                    cudnn_version = torch.backends.cudnn.version()
                    result["cudnn_version"] = cudnn_version
                    
                    if cudnn_version < 8400:  # 8.4.0
                        result["warnings"].append("cuDNN version is outdated")
                        result["recommendations"].append("Update to cuDNN 8.4.0 or newer")
            else:
                result["errors"].append("CUDA not available despite PyTorch being installed")
        
        # Check GPU memory state
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(i)
                free_gb = mem_info[0] / (1024**3)
                total_gb = mem_info[1] / (1024**3)
                used_percent = ((total_gb - free_gb) / total_gb) * 100
                
                if used_percent > 10:
                    result["warnings"].append(
                        f"GPU {i} already has {used_percent:.1f}% memory in use"
                    )
                    result["recommendations"].append(
                        f"Consider freeing GPU {i} memory before starting vLLM"
                    )
        
        result["passed"] = len(result["errors"]) == 0
        return result
    
    def _validate_memory_requirements(self) -> Dict[str, Any]:
        """Validate system memory requirements"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Get memory info
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        result["memory_info"] = {
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
            "used_percent": mem.percent,
            "swap_total_gb": swap.total / (1024**3),
            "swap_used_percent": swap.percent
        }
        
        # Check available memory
        if mem.available < 16 * 1024**3:  # 16GB minimum
            result["errors"].append(
                f"Insufficient system memory: {mem.available / (1024**3):.1f}GB available"
            )
            result["recommendations"].append("At least 16GB of free RAM is recommended")
        
        # Check memory fragmentation
        if hasattr(mem, 'buffers'):
            buffers_cached = (mem.buffers + mem.cached) / (1024**3)
            if buffers_cached > mem.total * 0.5:
                result["warnings"].append(
                    f"High memory usage by buffers/cache: {buffers_cached:.1f}GB"
                )
                result["recommendations"].append(
                    "Consider dropping caches: echo 3 > /proc/sys/vm/drop_caches"
                )
        
        # Check swap
        if swap.total == 0:
            result["warnings"].append("No swap space configured")
            result["recommendations"].append(
                "Consider adding swap space for memory overflow situations"
            )
        
        # Check huge pages
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "HugePages_Total" in line:
                        huge_pages = int(line.split()[1])
                        if huge_pages > 0:
                            result["warnings"].append(f"Huge pages enabled: {huge_pages}")
                            result["recommendations"].append(
                                "Huge pages may interfere with memory allocation"
                            )
        except:
            pass
        
        result["passed"] = len(result["errors"]) == 0
        return result
    
    def _validate_network_config(self) -> Dict[str, Any]:
        """Validate network configuration for distributed setup"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check network interfaces
        net_if_stats = psutil.net_if_stats()
        active_interfaces = [iface for iface, stats in net_if_stats.items() 
                           if stats.isup and iface != 'lo']
        
        if not active_interfaces:
            result["errors"].append("No active network interfaces found")
        
        # Check for InfiniBand
        ib_present = any('ib' in iface for iface in active_interfaces)
        if not ib_present and len(active_interfaces) > 0:
            # Check for high-speed ethernet
            for iface in active_interfaces:
                stats = net_if_stats[iface]
                if hasattr(stats, 'speed') and stats.speed < 10000:  # Less than 10Gbps
                    result["warnings"].append(
                        f"Network interface {iface} has low speed: {stats.speed}Mbps"
                    )
                    result["recommendations"].append(
                        "Use 10Gbps+ network for distributed training"
                    )
        
        # Check NCCL environment
        nccl_env_vars = {
            "NCCL_DEBUG": "WARN",  # Recommended for debugging
            "NCCL_SOCKET_IFNAME": None,  # Should be set for multi-NIC systems
        }
        
        for var, recommended in nccl_env_vars.items():
            current = os.environ.get(var)
            if recommended and current != recommended:
                result["recommendations"].append(
                    f"Set {var}={recommended} for better NCCL performance"
                )
            elif var == "NCCL_SOCKET_IFNAME" and not current and len(active_interfaces) > 1:
                result["recommendations"].append(
                    f"Set NCCL_SOCKET_IFNAME to specify which network interface to use"
                )
        
        result["passed"] = len(result["errors"]) == 0
        return result
    
    def _validate_environment_variables(self) -> Dict[str, Any]:
        """Validate critical environment variables"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Critical environment variables
        critical_vars = {
            "CUDA_VISIBLE_DEVICES": {
                "check": lambda v: v is not None,
                "error": "CUDA_VISIBLE_DEVICES not set",
                "recommendation": "Set CUDA_VISIBLE_DEVICES to control GPU visibility"
            },
            "OMP_NUM_THREADS": {
                "check": lambda v: v is not None and int(v) > 0,
                "error": "OMP_NUM_THREADS not set",
                "recommendation": "Set OMP_NUM_THREADS to number of CPU cores"
            }
        }
        
        # Performance-related variables
        perf_vars = {
            "CUDA_LAUNCH_BLOCKING": {
                "check": lambda v: v != "1",
                "warning": "CUDA_LAUNCH_BLOCKING=1 will hurt performance",
                "recommendation": "Unset CUDA_LAUNCH_BLOCKING for production"
            },
            "PYTORCH_CUDA_ALLOC_CONF": {
                "check": lambda v: v is not None,
                "recommendation": "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            }
        }
        
        # Check critical variables
        for var, config in critical_vars.items():
            value = os.environ.get(var)
            try:
                if not config["check"](value):
                    if "error" in config:
                        result["errors"].append(config["error"])
                    if "recommendation" in config:
                        result["recommendations"].append(config["recommendation"])
            except:
                pass
        
        # Check performance variables
        for var, config in perf_vars.items():
            value = os.environ.get(var)
            try:
                if not config["check"](value):
                    if "warning" in config:
                        result["warnings"].append(config["warning"])
                    if "recommendation" in config:
                        result["recommendations"].append(config["recommendation"])
            except:
                pass
        
        result["passed"] = len(result["errors"]) == 0
        return result
    
    def _validate_dependencies(self) -> Dict[str, Any]:
        """Validate Python dependencies"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check critical dependencies
        critical_deps = {
            "torch": {
                "min_version": "2.0.0",
                "check": lambda: torch.__version__
            },
            "transformers": {
                "min_version": "4.34.0",
                "check": lambda: __import__("transformers").__version__
            },
            "numpy": {
                "min_version": "1.21.0",
                "check": lambda: __import__("numpy").__version__
            }
        }
        
        for package, config in critical_deps.items():
            try:
                version = config["check"]()
                # Simple version comparison (would be more robust in production)
                if version < config["min_version"]:
                    result["warnings"].append(
                        f"{package} version {version} is older than recommended {config['min_version']}"
                    )
            except ImportError:
                result["errors"].append(f"Required package '{package}' not installed")
            except Exception as e:
                result["warnings"].append(f"Failed to check {package}: {str(e)}")
        
        # Check for conflicting packages
        try:
            import tensorflow
            result["warnings"].append("TensorFlow detected - may cause GPU memory conflicts")
            result["recommendations"].append(
                "Consider running in a separate environment without TensorFlow"
            )
        except ImportError:
            pass
        
        result["passed"] = len(result["errors"]) == 0
        return result
    
    def _validate_filesystem(self) -> Dict[str, Any]:
        """Validate filesystem requirements"""
        result = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Check temp directory space
        temp_dir = Path("/tmp")
        if temp_dir.exists():
            temp_stats = psutil.disk_usage(str(temp_dir))
            temp_free_gb = temp_stats.free / (1024**3)
            
            if temp_free_gb < 10:
                result["warnings"].append(
                    f"Low temp space: {temp_free_gb:.1f}GB free in /tmp"
                )
                result["recommendations"].append(
                    "Ensure at least 10GB free space in temp directory"
                )
        
        # Check model cache directory
        cache_dir = Path.home() / ".cache" / "huggingface"
        if cache_dir.exists():
            cache_stats = psutil.disk_usage(str(cache_dir.parent))
            cache_free_gb = cache_stats.free / (1024**3)
            
            if cache_free_gb < 50:
                result["warnings"].append(
                    f"Low cache space: {cache_free_gb:.1f}GB free for model cache"
                )
                result["recommendations"].append(
                    "Ensure sufficient space for model downloads"
                )
        
        # Check file descriptor limits
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < 65536:
                result["warnings"].append(f"Low file descriptor limit: {soft}")
                result["recommendations"].append(
                    "Increase file descriptor limit: ulimit -n 65536"
                )
        except:
            pass
        
        result["passed"] = len(result["errors"]) == 0
        return result


# ============================================================================
# State Tracking Plugins
# ============================================================================

class VLLMStateTracker(PluginInterface):
    """Comprehensive state tracking plugin for all vLLM states"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="vllm_state_tracker",
            version="1.0.0",
            type=PluginType.COLLECTOR,
            description="Tracks all vLLM lifecycle states and transitions",
            capabilities=["state_tracking", "lifecycle", "transitions"],
            auto_enable=True
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        self.context = context
        self.lifecycle_tracker = context.get("lifecycle_tracker")
        self.state_hooks = {}
        self._register_state_hooks()
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Collect current state information"""
        if not self.lifecycle_tracker:
            return {"error": "No lifecycle tracker available"}
        
        current_state = self.lifecycle_tracker.current_state
        state_info = {
            "timestamp": time.time(),
            "current_state": current_state.name,
            "state_duration": time.time() - self.lifecycle_tracker._last_checkpoint_time,
            "total_checkpoints": len(self.lifecycle_tracker.checkpoints),
            "state_health": self._assess_state_health(current_state),
            "pending_transitions": self._get_pending_transitions(current_state),
            "state_specific_metrics": self._collect_state_specific_metrics(current_state)
        }
        
        return state_info
    
    def cleanup(self):
        pass
    
    def _register_state_hooks(self):
        """Register hooks for each state"""
        self.state_hooks = {
            LifecycleState.INITIALIZING: self._track_initialization,
            LifecycleState.LOADING_MODEL: self._track_model_loading,
            LifecycleState.ALLOCATING_MEMORY: self._track_memory_allocation,
            LifecycleState.COMPILING_KERNELS: self._track_kernel_compilation,
            LifecycleState.SETTING_UP_WORKERS: self._track_worker_setup,
            LifecycleState.SERVING: self._track_serving,
            LifecycleState.ERROR: self._track_error_state,
            LifecycleState.SHUTTING_DOWN: self._track_shutdown
        }
    
    def _assess_state_health(self, state: LifecycleState) -> str:
        """Assess health of current state"""
        if state in [LifecycleState.ERROR, LifecycleState.CRITICAL_ERROR]:
            return "unhealthy"
        elif state in [LifecycleState.RECOVERING, LifecycleState.PAUSED]:
            return "degraded"
        else:
            return "healthy"
    
    def _get_pending_transitions(self, state: LifecycleState) -> List[str]:
        """Get possible transitions from current state"""
        transitions = {
            LifecycleState.INITIALIZING: ["LOADING_MODEL", "ERROR"],
            LifecycleState.LOADING_MODEL: ["ALLOCATING_MEMORY", "ERROR"],
            LifecycleState.READY: ["SERVING", "PAUSED", "SHUTTING_DOWN"],
            LifecycleState.SERVING: ["PROCESSING_REQUESTS", "PAUSED", "ERROR"],
            LifecycleState.ERROR: ["RECOVERING", "CRITICAL_ERROR", "SHUTTING_DOWN"]
        }
        
        return transitions.get(state, [])
    
    def _collect_state_specific_metrics(self, state: LifecycleState) -> Dict[str, Any]:
        """Collect metrics specific to current state"""
        if state in self.state_hooks:
            return self.state_hooks[state]()
        return {}
    
    def _track_initialization(self) -> Dict[str, Any]:
        """Track initialization state"""
        metrics = {
            "config_loaded": self._check_config_loaded(),
            "dependencies_ready": self._check_dependencies(),
            "cuda_initialized": torch.cuda.is_initialized() if TORCH_AVAILABLE else False
        }
        return metrics
    
    def _track_model_loading(self) -> Dict[str, Any]:
        """Track model loading state"""
        metrics = {
            "model_size_gb": self._estimate_model_size(),
            "download_progress": self._get_download_progress(),
            "available_memory_gb": self._get_available_memory()
        }
        return metrics
    
    def _track_memory_allocation(self) -> Dict[str, Any]:
        """Track memory allocation state"""
        metrics = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                mem_info = torch.cuda.mem_get_info(i)
                metrics[f"gpu_{i}_free_gb"] = mem_info[0] / (1024**3)
                metrics[f"gpu_{i}_allocated_gb"] = (mem_info[1] - mem_info[0]) / (1024**3)
        
        return metrics
    
    def _track_kernel_compilation(self) -> Dict[str, Any]:
        """Track kernel compilation state"""
        return {
            "compilation_time": time.time() - self.lifecycle_tracker._last_checkpoint_time,
            "cuda_graphs_enabled": os.environ.get("VLLM_USE_CUDA_GRAPH", "1") == "1"
        }
    
    def _track_worker_setup(self) -> Dict[str, Any]:
        """Track worker setup state"""
        metrics = {
            "worker_count": self._get_worker_count(),
            "distributed_initialized": dist.is_initialized() if TORCH_AVAILABLE else False
        }
        
        if RAY_AVAILABLE and ray.is_initialized():
            metrics["ray_nodes"] = len(ray.nodes())
            metrics["ray_resources"] = ray.available_resources()
        
        return metrics
    
    def _track_serving(self) -> Dict[str, Any]:
        """Track serving state"""
        return {
            "requests_per_second": self._get_request_rate(),
            "active_requests": self._get_active_requests(),
            "queue_size": self._get_queue_size(),
            "average_latency_ms": self._get_average_latency()
        }
    
    def _track_error_state(self) -> Dict[str, Any]:
        """Track error state"""
        return {
            "error_count": len(self.lifecycle_tracker.checkpoints[-1].error_context.get("errors", [])),
            "error_types": self._classify_errors(),
            "recovery_attempts": self._get_recovery_attempts()
        }
    
    def _track_shutdown(self) -> Dict[str, Any]:
        """Track shutdown state"""
        return {
            "cleanup_progress": self._get_cleanup_progress(),
            "resources_released": self._check_resources_released()
        }
    
    # Helper methods
    def _check_config_loaded(self) -> bool:
        return self.context.get("config") is not None
    
    def _check_dependencies(self) -> bool:
        return True  # Simplified
    
    def _estimate_model_size(self) -> float:
        # Would get from model config
        return 7.0  # Default 7B model
    
    def _get_download_progress(self) -> float:
        return 100.0  # Simplified
    
    def _get_available_memory(self) -> float:
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.mem_get_info()[0] / (1024**3)
        return 0.0
    
    def _get_worker_count(self) -> int:
        return self.context.get("worker_count", 1)
    
    def _get_request_rate(self) -> float:
        return 0.0  # Would get from request tracker
    
    def _get_active_requests(self) -> int:
        return 0  # Would get from request tracker
    
    def _get_queue_size(self) -> int:
        return 0  # Would get from scheduler
    
    def _get_average_latency(self) -> float:
        return 0.0  # Would get from metrics
    
    def _classify_errors(self) -> List[str]:
        return []  # Would analyze error types
    
    def _get_recovery_attempts(self) -> int:
        return 0  # Would track recovery attempts
    
    def _get_cleanup_progress(self) -> float:
        return 0.0  # Would track cleanup
    
    def _check_resources_released(self) -> bool:
        return False  # Would check resource state


# ============================================================================
# Exception Monitoring Plugins
# ============================================================================

class VLLMExceptionMonitor(PluginInterface):
    """Comprehensive exception monitoring for vLLM"""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="vllm_exception_monitor",
            version="1.0.0",
            type=PluginType.COLLECTOR,
            description="Monitors and classifies all vLLM exceptions",
            capabilities=["exception_monitoring", "error_classification"],
            auto_enable=True
        )
    
    def initialize(self, context: Dict[str, Any]) -> bool:
        self.context = context
        self.exception_handlers = self._create_exception_handlers()
        self.exception_history = []
        self._install_exception_hooks()
        return True
    
    def execute(self) -> Dict[str, Any]:
        """Get exception monitoring data"""
        return {
            "timestamp": time.time(),
            "total_exceptions": len(self.exception_history),
            "recent_exceptions": self.exception_history[-10:],
            "exception_summary": self._summarize_exceptions(),
            "error_patterns": self._identify_error_patterns()
        }
    
    def cleanup(self):
        self._uninstall_exception_hooks()
    
    def _create_exception_handlers(self) -> Dict[type, Callable]:
        """Create handlers for specific exception types"""
        return {
            # CUDA/GPU exceptions
            RuntimeError: self._handle_runtime_error,
            torch.cuda.OutOfMemoryError: self._handle_cuda_oom,
            
            # Model loading exceptions
            ValueError: self._handle_value_error,
            KeyError: self._handle_key_error,
            
            # Network/distributed exceptions
            ConnectionError: self._handle_connection_error,
            TimeoutError: self._handle_timeout_error,
            
            # Resource exceptions
            MemoryError: self._handle_memory_error,
            OSError: self._handle_os_error,
            
            # Generic exceptions
            Exception: self._handle_generic_exception
        }
    
    def _install_exception_hooks(self):
        """Install global exception hooks"""
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._global_exception_handler
    
    def _uninstall_exception_hooks(self):
        """Restore original exception hooks"""
        if hasattr(self, '_original_excepthook'):
            sys.excepthook = self._original_excepthook
    
    def _global_exception_handler(self, exc_type, exc_value, exc_traceback):
        """Global exception handler"""
        # Log exception
        self._log_exception(exc_type, exc_value, exc_traceback)
        
        # Call appropriate handler
        handler = self.exception_handlers.get(exc_type, self._handle_generic_exception)
        handler(exc_value, exc_traceback)
        
        # Call original handler
        self._original_excepthook(exc_type, exc_value, exc_traceback)
    
    def _log_exception(self, exc_type, exc_value, exc_traceback):
        """Log exception details"""
        exception_info = {
            "timestamp": time.time(),
            "type": exc_type.__name__,
            "message": str(exc_value),
            "traceback": traceback.format_tb(exc_traceback),
            "state": self.context.get("current_state", "unknown"),
            "metrics": self._collect_error_metrics()
        }
        
        self.exception_history.append(exception_info)
    
    def _handle_runtime_error(self, exc_value, exc_traceback):
        """Handle PyTorch runtime errors"""
        error_str = str(exc_value)
        
        if "out of memory" in error_str.lower():
            self._handle_cuda_oom(exc_value, exc_traceback)
        elif "nccl" in error_str.lower():
            self._handle_nccl_error(exc_value, exc_traceback)
        elif "cudnn" in error_str.lower():
            self._handle_cudnn_error(exc_value, exc_traceback)
    
    def _handle_cuda_oom(self, exc_value, exc_traceback):
        """Handle CUDA out of memory errors"""
        # Extract allocation info
        error_str = str(exc_value)
        
        mitigation = {
            "error": "CUDA_OOM",
            "immediate_actions": [
                "clear_cuda_cache",
                "reduce_batch_size",
                "enable_cpu_offload"
            ],
            "recommendations": [
                "Reduce max_model_len",
                "Lower gpu_memory_utilization",
                "Enable quantization"
            ]
        }
        
        self.context["last_oom_error"] = mitigation
    
    def _handle_nccl_error(self, exc_value, exc_traceback):
        """Handle NCCL communication errors"""
        mitigation = {
            "error": "NCCL_ERROR",
            "immediate_actions": [
                "check_network_connectivity",
                "restart_distributed_workers",
                "fallback_to_gloo"
            ],
            "recommendations": [
                "Check NCCL_DEBUG=INFO logs",
                "Verify all GPUs are accessible",
                "Check for network timeouts"
            ]
        }
        
        self.context["last_nccl_error"] = mitigation
    
    def _handle_cudnn_error(self, exc_value, exc_traceback):
        """Handle cuDNN errors"""
        mitigation = {
            "error": "CUDNN_ERROR",
            "immediate_actions": [
                "disable_cudnn_benchmarking",
                "clear_cudnn_cache",
                "reduce_sequence_length"
            ],
            "recommendations": [
                "Update cuDNN version",
                "Check GPU compute capability",
                "Verify tensor dimensions"
            ]
        }
        
        self.context["last_cudnn_error"] = mitigation
    
    def _handle_value_error(self, exc_value, exc_traceback):
        """Handle value errors (often from model loading)"""
        error_str = str(exc_value)
        
        if "model" in error_str.lower():
            mitigation = {
                "error": "MODEL_LOADING_ERROR",
                "immediate_actions": [
                    "verify_model_files",
                    "check_model_config",
                    "clear_model_cache"
                ],
                "recommendations": [
                    "Verify model compatibility",
                    "Check tokenizer configuration",
                    "Ensure model files are complete"
                ]
            }
        else:
            mitigation = {
                "error": "INVALID_CONFIGURATION",
                "immediate_actions": [
                    "validate_parameters",
                    "reset_to_defaults"
                ]
            }
        
        self.context["last_value_error"] = mitigation
    
    def _handle_key_error(self, exc_value, exc_traceback):
        """Handle key errors (often from config issues)"""
        mitigation = {
            "error": "CONFIGURATION_KEY_ERROR",
            "immediate_actions": [
                "check_config_schema",
                "provide_default_values"
            ],
            "recommendations": [
                "Review configuration documentation",
                "Use validated configuration loader"
            ]
        }
        
        self.context["last_key_error"] = mitigation
    
    def _handle_connection_error(self, exc_value, exc_traceback):
        """Handle network connection errors"""
        mitigation = {
            "error": "NETWORK_CONNECTION_ERROR",
            "immediate_actions": [
                "retry_connection",
                "check_firewall_rules",
                "verify_endpoints"
            ],
            "recommendations": [
                "Check network connectivity",
                "Verify service availability",
                "Review timeout settings"
            ]
        }
        
        self.context["last_connection_error"] = mitigation
    
    def _handle_timeout_error(self, exc_value, exc_traceback):
        """Handle timeout errors"""
        mitigation = {
            "error": "OPERATION_TIMEOUT",
            "immediate_actions": [
                "increase_timeout_duration",
                "check_resource_availability",
                "reduce_operation_scope"
            ]
        }
        
        self.context["last_timeout_error"] = mitigation
    
    def _handle_memory_error(self, exc_value, exc_traceback):
        """Handle system memory errors"""
        mitigation = {
            "error": "SYSTEM_MEMORY_ERROR",
            "immediate_actions": [
                "free_system_memory",
                "enable_swap",
                "reduce_model_size"
            ],
            "critical": True
        }
        
        self.context["last_memory_error"] = mitigation
    
    def _handle_os_error(self, exc_value, exc_traceback):
        """Handle OS-level errors"""
        error_str = str(exc_value)
        
        if "space" in error_str.lower():
            mitigation = {
                "error": "DISK_SPACE_ERROR",
                "immediate_actions": [
                    "clear_temp_files",
                    "clean_model_cache"
                ]
            }
        else:
            mitigation = {
                "error": "OS_ERROR",
                "immediate_actions": [
                    "check_permissions",
                    "verify_paths"
                ]
            }
        
        self.context["last_os_error"] = mitigation
    
    def _handle_generic_exception(self, exc_value, exc_traceback):
        """Handle any other exception"""
        mitigation = {
            "error": "GENERIC_ERROR",
            "immediate_actions": [
                "log_full_traceback",
                "capture_system_state",
                "attempt_graceful_recovery"
            ]
        }
        
        self.context["last_generic_error"] = mitigation
    
    def _collect_error_metrics(self) -> Dict[str, Any]:
        """Collect metrics at time of error"""
        metrics = {}
        
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics[f"gpu_{i}_memory_used"] = torch.cuda.memory_allocated(i) / (1024**3)
        
        metrics["cpu_percent"] = psutil.cpu_percent()
        metrics["memory_percent"] = psutil.virtual_memory().percent
        
        return metrics
    
    def _summarize_exceptions(self) -> Dict[str, int]:
        """Summarize exceptions by type"""
        summary = {}
        for exc in self.exception_history:
            exc_type = exc["type"]
            summary[exc_type] = summary.get(exc_type, 0) + 1
        return summary
    
    def _identify_error_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in errors"""
        patterns = []
        
        # OOM pattern
        oom_errors = [e for e in self.exception_history 
                     if "out of memory" in e.get("message", "").lower()]
        if len(oom_errors) > 2:
            patterns.append({
                "pattern": "frequent_oom",
                "count": len(oom_errors),
                "recommendation": "Reduce memory usage systematically"
            })
        
        # NCCL pattern
        nccl_errors = [e for e in self.exception_history 
                      if "nccl" in e.get("message", "").lower()]
        if len(nccl_errors) > 1:
            patterns.append({
                "pattern": "nccl_instability",
                "count": len(nccl_errors),
                "recommendation": "Check distributed setup"
            })
        
        return patterns


# ============================================================================
# Guardrail Generation Functions
# ============================================================================

def create_memory_guardrails() -> List[GuardrailPolicy]:
    """Create memory-related guardrails"""
    guardrails = []
    
    # GPU memory guardrails
    guardrails.append(GuardrailPolicy(
        name="gpu_memory_critical",
        description="Prevent GPU OOM by monitoring memory usage",
        condition=lambda cp: any(
            cp.metrics.get(f"gpu_{i}_memory_percent", 0) > 95
            for i in range(torch.cuda.device_count() if TORCH_AVAILABLE else 0)
        ),
        intervention="emergency_gpu_memory_cleanup",
        severity="critical"
    ))
    
    guardrails.append(GuardrailPolicy(
        name="gpu_memory_warning",
        description="Warn on high GPU memory usage",
        condition=lambda cp: any(
            cp.metrics.get(f"gpu_{i}_memory_percent", 0) > 85
            for i in range(torch.cuda.device_count() if TORCH_AVAILABLE else 0)
        ),
        intervention="reduce_batch_size",
        severity="warning"
    ))
    
    # System memory guardrails
    guardrails.append(GuardrailPolicy(
        name="system_memory_critical",
        description="Prevent system OOM",
        condition=lambda cp: cp.metrics.get("memory_percent", 0) > 95,
        intervention="emergency_system_memory_cleanup",
        severity="critical"
    ))
    
    # Memory growth guardrail
    guardrails.append(GuardrailPolicy(
        name="memory_leak_detection",
        description="Detect potential memory leaks",
        condition=lambda cp: cp.metrics.get("memory_growth_rate", 0) > 0.1,  # 10% per minute
        intervention="investigate_memory_leak",
        severity="warning"
    ))
    
    return guardrails


def create_performance_guardrails() -> List[GuardrailPolicy]:
    """Create performance-related guardrails"""
    guardrails = []
    
    # Latency guardrails
    guardrails.append(GuardrailPolicy(
        name="high_latency",
        description="Detect high inference latency",
        condition=lambda cp: cp.metrics.get("avg_latency_ms", 0) > 1000,
        intervention="optimize_inference",
        severity="warning"
    ))
    
    # Throughput guardrails
    guardrails.append(GuardrailPolicy(
        name="low_throughput",
        description="Detect low throughput",
        condition=lambda cp: cp.metrics.get("tokens_per_second", float('inf')) < 100,
        intervention="increase_batch_size",
        severity="warning"
    ))
    
    # GPU utilization guardrails
    guardrails.append(GuardrailPolicy(
        name="low_gpu_utilization",
        description="Detect underutilized GPU",
        condition=lambda cp: cp.metrics.get("gpu_utilization", 100) < 50,
        intervention="optimize_gpu_usage",
        severity="warning"
    ))
    
    # Queue buildup guardrail
    guardrails.append(GuardrailPolicy(
        name="request_queue_buildup",
        description="Detect request queue overflow",
        condition=lambda cp: cp.metrics.get("queue_size", 0) > 1000,
        intervention="scale_workers",
        severity="error"
    ))
    
    return guardrails


def create_stability_guardrails() -> List[GuardrailPolicy]:
    """Create stability-related guardrails"""
    guardrails = []
    
    # Error rate guardrails
    guardrails.append(GuardrailPolicy(
        name="high_error_rate",
        description="Detect high error rate",
        condition=lambda cp: cp.metrics.get("error_rate", 0) > 0.05,
        intervention="investigate_errors",
        severity="error"
    ))
    
    # Worker health guardrails
    guardrails.append(GuardrailPolicy(
        name="worker_failure",
        description="Detect worker failures",
        condition=lambda cp: cp.metrics.get("failed_workers", 0) > 0,
        intervention="restart_failed_workers",
        severity="error"
    ))
    
    # Deadlock detection
    guardrails.append(GuardrailPolicy(
        name="potential_deadlock",
        description="Detect potential deadlock",
        condition=lambda cp: cp.metrics.get("requests_stuck", 0) > 0,
        intervention="break_deadlock",
        severity="critical"
    ))
    
    # Temperature guardrails
    guardrails.append(GuardrailPolicy(
        name="gpu_overheating",
        description="Detect GPU overheating",
        condition=lambda cp: any(
            cp.metrics.get(f"gpu_{i}_temperature", 0) > 85
            for i in range(torch.cuda.device_count() if TORCH_AVAILABLE else 0)
        ),
        intervention="throttle_gpu",
        severity="warning"
    ))
    
    return guardrails


def create_all_guardrails() -> List[GuardrailPolicy]:
    """Create all guardrails for vLLM"""
    all_guardrails = []
    all_guardrails.extend(create_memory_guardrails())
    all_guardrails.extend(create_performance_guardrails())
    all_guardrails.extend(create_stability_guardrails())
    return all_guardrails


# ============================================================================
# Plugin Registration Function
# ============================================================================

def register_all_vllm_plugins(plugin_manager):
    """Register all vLLM integration plugins"""
    # Register pre-startup validator
    plugin_manager.registry.register(PreStartupValidator())
    
    # Register state tracker
    plugin_manager.registry.register(VLLMStateTracker())
    
    # Register exception monitor
    plugin_manager.registry.register(VLLMExceptionMonitor())
    
    # Create and register guardrails
    for guardrail in create_all_guardrails():
        plugin_manager.lifecycle_tracker.register_guardrail(guardrail)
    
    # Register mitigation strategies (covered in next file)
    
    return True