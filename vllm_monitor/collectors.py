"""
High-performance data collectors for vLLM monitoring.

These collectors are designed to extract state information from vLLM components
with sub-microsecond overhead and minimal memory allocation.
"""

import gc
import inspect
import time
import threading
import weakref
from typing import Any, Dict, List, Optional, Set, Union
import psutil
import numpy as np

try:
    import GPUtil
    HAS_GPUTIL = True
except ImportError:
    HAS_GPUTIL = False

from .core import (
    ComponentState, ComponentType, StateType, Collector, 
    PerformanceTimer, AlertLevel
)

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import nvml_py3 as nvml
    nvml.nvmlInit()
    HAS_NVML = True
except ImportError:
    HAS_NVML = False


class BaseCollector:
    """Base class for all collectors with common functionality."""
    
    def __init__(self, enabled: bool = True, sampling_rate: float = 1.0):
        self.enabled = enabled
        self.sampling_rate = sampling_rate
        self._collection_count = 0
        self._total_time_us = 0.0
        self._lock = threading.RLock()
    
    def set_sampling_rate(self, rate: float) -> None:
        """Set sampling rate for adaptive collection."""
        with self._lock:
            self.sampling_rate = max(0.0, min(1.0, rate))
    
    def should_collect(self) -> bool:
        """Check if we should collect data based on sampling rate."""
        if not self.enabled:
            return False
        if self.sampling_rate >= 1.0:
            return True
        return np.random.random() < self.sampling_rate
    
    def get_overhead_us(self) -> float:
        """Get average collection overhead in microseconds."""
        with self._lock:
            if self._collection_count == 0:
                return 0.0
            return self._total_time_us / self._collection_count
    
    def _record_collection_time(self, time_us: float) -> None:
        """Record collection time for overhead tracking."""
        with self._lock:
            self._collection_count += 1
            self._total_time_us += time_us


class StateCollector(BaseCollector):
    """Collector for vLLM component states with minimal introspection overhead."""
    
    def __init__(self, monitor_registry: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.monitor_registry = monitor_registry
        self._last_collection = {}
        
        # Pre-compile attribute paths for faster access
        self._state_extractors = {
            ComponentType.ENGINE: self._extract_engine_state,
            ComponentType.SCHEDULER: self._extract_scheduler_state,
            ComponentType.WORKER: self._extract_worker_state,
            ComponentType.MODEL_RUNNER: self._extract_model_runner_state,
            ComponentType.CACHE_ENGINE: self._extract_cache_state,
            ComponentType.BLOCK_MANAGER: self._extract_block_manager_state,
        }
    
    def collect(self) -> List[ComponentState]:
        """Collect states from all registered components."""
        if not self.should_collect():
            return []
        
        with PerformanceTimer() as timer:
            states = []
            current_time = time.time()
            
            for component_id, component_info in self.monitor_registry.items():
                component_ref = component_info['ref']
                component = component_ref()
                
                if component is None:
                    continue  # Component was garbage collected
                
                component_type = component_info['type']
                extractor = self._state_extractors.get(component_type)
                
                if extractor:
                    try:
                        state = extractor(component_id, component, current_time)
                        if state:
                            states.append(state)
                    except Exception as e:
                        # Create error state
                        error_state = ComponentState(
                            component_id=component_id,
                            component_type=component_type,
                            state_type=StateType.ERROR,
                            timestamp=current_time,
                            is_healthy=False,
                            last_error=str(e),
                            error_count=1
                        )
                        states.append(error_state)
        
        self._record_collection_time(timer.elapsed_us)
        return states
    
    def _extract_engine_state(self, component_id: str, engine: Any, timestamp: float) -> Optional[ComponentState]:
        """Extract state from LLM Engine with minimal overhead."""
        try:
            data = {}
            
            # Use getattr with defaults to avoid exceptions
            if hasattr(engine, 'engine_config'):
                config = engine.engine_config
                data['model_config'] = {
                    'model': getattr(config, 'model', 'unknown'),
                    'max_model_len': getattr(config, 'max_model_len', 0),
                    'dtype': str(getattr(config, 'dtype', 'unknown')),
                }
            
            if hasattr(engine, 'scheduler'):
                scheduler = engine.scheduler
                if hasattr(scheduler, 'get_num_unfinished_seq_groups'):
                    data['pending_requests'] = scheduler.get_num_unfinished_seq_groups()
            
            # Check if engine is running
            is_running = getattr(engine, '_is_running', True)
            
            return ComponentState(
                component_id=component_id,
                component_type=ComponentType.ENGINE,
                state_type=StateType.OPERATIONAL,
                timestamp=timestamp,
                data=data,
                is_healthy=is_running
            )
            
        except Exception:
            return None
    
    def _extract_scheduler_state(self, component_id: str, scheduler: Any, timestamp: float) -> Optional[ComponentState]:
        """Extract state from scheduler with minimal overhead."""
        try:
            data = {}
            
            # Get queue information
            if hasattr(scheduler, 'waiting'):
                data['waiting_requests'] = len(scheduler.waiting)
            if hasattr(scheduler, 'running'):
                data['running_requests'] = len(scheduler.running)
            if hasattr(scheduler, 'swapped'):
                data['swapped_requests'] = len(scheduler.swapped)
            
            # Get capacity information
            if hasattr(scheduler, 'block_manager'):
                block_manager = scheduler.block_manager
                if hasattr(block_manager, 'get_num_free_gpu_blocks'):
                    data['free_gpu_blocks'] = block_manager.get_num_free_gpu_blocks()
                if hasattr(block_manager, 'get_num_free_cpu_blocks'):
                    data['free_cpu_blocks'] = block_manager.get_num_free_cpu_blocks()
            
            # Calculate health based on queue sizes
            total_requests = sum([
                data.get('waiting_requests', 0),
                data.get('running_requests', 0),
                data.get('swapped_requests', 0)
            ])
            
            # Health score based on utilization
            health_score = 1.0
            if total_requests > 100:  # Threshold for high load
                health_score = max(0.1, 1.0 - (total_requests - 100) / 1000)
            
            return ComponentState(
                component_id=component_id,
                component_type=ComponentType.SCHEDULER,
                state_type=StateType.OPERATIONAL,
                timestamp=timestamp,
                data=data,
                health_score=health_score,
                is_healthy=health_score > 0.5
            )
            
        except Exception:
            return None
    
    def _extract_worker_state(self, component_id: str, worker: Any, timestamp: float) -> Optional[ComponentState]:
        """Extract state from worker with minimal overhead."""
        try:
            data = {}
            
            # Get worker ID and rank
            if hasattr(worker, 'rank'):
                data['rank'] = worker.rank
            if hasattr(worker, 'local_rank'):
                data['local_rank'] = worker.local_rank
            
            # Get model information
            if hasattr(worker, 'model_runner'):
                model_runner = worker.model_runner
                if hasattr(model_runner, 'model'):
                    data['model_loaded'] = model_runner.model is not None
            
            # Get cache information
            if hasattr(worker, 'cache_engine'):
                cache_engine = worker.cache_engine
                if hasattr(cache_engine, 'get_num_free_gpu_blocks'):
                    data['free_cache_blocks'] = cache_engine.get_num_free_gpu_blocks()
            
            return ComponentState(
                component_id=component_id,
                component_type=ComponentType.WORKER,
                state_type=StateType.OPERATIONAL,
                timestamp=timestamp,
                data=data,
                is_healthy=True
            )
            
        except Exception:
            return None
    
    def _extract_model_runner_state(self, component_id: str, model_runner: Any, timestamp: float) -> Optional[ComponentState]:
        """Extract state from model runner with minimal overhead."""
        try:
            data = {}
            
            # Model information
            if hasattr(model_runner, 'model'):
                model = model_runner.model
                data['model_loaded'] = model is not None
                if model and hasattr(model, 'config'):
                    config = model.config
                    data['model_type'] = getattr(config, 'model_type', 'unknown')
            
            # Execution statistics
            if hasattr(model_runner, '_last_sampled_token_ids'):
                data['last_execution'] = getattr(model_runner, '_last_execution_time', 0)
            
            return ComponentState(
                component_id=component_id,
                component_type=ComponentType.MODEL_RUNNER,
                state_type=StateType.OPERATIONAL,
                timestamp=timestamp,
                data=data,
                is_healthy=data.get('model_loaded', False)
            )
            
        except Exception:
            return None
    
    def _extract_cache_state(self, component_id: str, cache_engine: Any, timestamp: float) -> Optional[ComponentState]:
        """Extract state from cache engine with minimal overhead."""
        try:
            data = {}
            
            # Cache block information
            if hasattr(cache_engine, 'gpu_cache'):
                data['has_gpu_cache'] = cache_engine.gpu_cache is not None
            if hasattr(cache_engine, 'cpu_cache'):
                data['has_cpu_cache'] = cache_engine.cpu_cache is not None
            
            # Memory usage if available
            if hasattr(cache_engine, 'get_num_free_gpu_blocks'):
                data['free_gpu_blocks'] = cache_engine.get_num_free_gpu_blocks()
            if hasattr(cache_engine, 'get_num_free_cpu_blocks'):
                data['free_cpu_blocks'] = cache_engine.get_num_free_cpu_blocks()
            
            # Health based on available blocks
            health_score = 1.0
            free_blocks = data.get('free_gpu_blocks', 0)
            if free_blocks == 0:
                health_score = 0.1  # Critical: no free blocks
            elif free_blocks < 10:
                health_score = 0.3  # Low: very few free blocks
            elif free_blocks < 50:
                health_score = 0.7  # Medium: some free blocks
            
            return ComponentState(
                component_id=component_id,
                component_type=ComponentType.CACHE_ENGINE,
                state_type=StateType.RESOURCE,
                timestamp=timestamp,
                data=data,
                health_score=health_score,
                is_healthy=health_score > 0.5
            )
            
        except Exception:
            return None
    
    def _extract_block_manager_state(self, component_id: str, block_manager: Any, timestamp: float) -> Optional[ComponentState]:
        """Extract state from block manager with minimal overhead."""
        try:
            data = {}
            
            # Block allocation information
            if hasattr(block_manager, 'block_size'):
                data['block_size'] = block_manager.block_size
            
            if hasattr(block_manager, 'num_gpu_blocks'):
                data['total_gpu_blocks'] = block_manager.num_gpu_blocks
            if hasattr(block_manager, 'num_cpu_blocks'):
                data['total_cpu_blocks'] = block_manager.num_cpu_blocks
            
            # Free blocks
            if hasattr(block_manager, 'get_num_free_gpu_blocks'):
                free_gpu = block_manager.get_num_free_gpu_blocks()
                data['free_gpu_blocks'] = free_gpu
                data['gpu_utilization'] = 1.0 - (free_gpu / max(data.get('total_gpu_blocks', 1), 1))
            
            if hasattr(block_manager, 'get_num_free_cpu_blocks'):
                free_cpu = block_manager.get_num_free_cpu_blocks()
                data['free_cpu_blocks'] = free_cpu
                data['cpu_utilization'] = 1.0 - (free_cpu / max(data.get('total_cpu_blocks', 1), 1))
            
            # Calculate health score based on utilization
            gpu_util = data.get('gpu_utilization', 0)
            if gpu_util > 0.95:
                health_score = 0.1
            elif gpu_util > 0.85:
                health_score = 0.5
            elif gpu_util > 0.7:
                health_score = 0.8
            else:
                health_score = 1.0
            
            return ComponentState(
                component_id=component_id,
                component_type=ComponentType.BLOCK_MANAGER,
                state_type=StateType.RESOURCE,
                timestamp=timestamp,
                data=data,
                health_score=health_score,
                is_healthy=health_score > 0.5
            )
            
        except Exception:
            return None


class PerformanceCollector(BaseCollector):
    """Collector for performance metrics with minimal overhead."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._last_cpu_times = {}
        self._last_network_io = None
        
    def collect(self) -> List[ComponentState]:
        """Collect system-wide performance metrics."""
        if not self.should_collect():
            return []
        
        with PerformanceTimer() as timer:
            states = []
            current_time = time.time()
            
            # CPU metrics
            cpu_state = self._collect_cpu_metrics(current_time)
            if cpu_state:
                states.append(cpu_state)
            
            # Memory metrics
            memory_state = self._collect_memory_metrics(current_time)
            if memory_state:
                states.append(memory_state)
            
            # GPU metrics
            gpu_states = self._collect_gpu_metrics(current_time)
            states.extend(gpu_states)
            
            # Network metrics
            network_state = self._collect_network_metrics(current_time)
            if network_state:
                states.append(network_state)
        
        self._record_collection_time(timer.elapsed_us)
        return states
    
    def _collect_cpu_metrics(self, timestamp: float) -> Optional[ComponentState]:
        """Collect CPU performance metrics."""
        try:
            # Use non-blocking CPU percent
            cpu_percent = psutil.cpu_percent(interval=None)
            cpu_count = psutil.cpu_count()
            
            # Load average
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            data = {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_avg_1m': load_avg[0],
                'load_avg_5m': load_avg[1],
                'load_avg_15m': load_avg[2],
            }
            
            # Health score based on CPU usage
            health_score = 1.0 - min(cpu_percent / 100.0, 1.0)
            
            return ComponentState(
                component_id="system_cpu",
                component_type=ComponentType.WORKER,  # Closest match
                state_type=StateType.PERFORMANCE,
                timestamp=timestamp,
                data=data,
                cpu_usage=cpu_percent,
                health_score=health_score,
                is_healthy=cpu_percent < 90.0
            )
            
        except Exception:
            return None
    
    def _collect_memory_metrics(self, timestamp: float) -> Optional[ComponentState]:
        """Collect memory performance metrics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            data = {
                'total_memory_gb': memory.total / (1024**3),
                'available_memory_gb': memory.available / (1024**3),
                'used_memory_gb': memory.used / (1024**3),
                'memory_percent': memory.percent,
                'swap_total_gb': swap.total / (1024**3),
                'swap_used_gb': swap.used / (1024**3),
                'swap_percent': swap.percent,
            }
            
            # Health score based on memory usage
            health_score = 1.0 - (memory.percent / 100.0)
            
            return ComponentState(
                component_id="system_memory",
                component_type=ComponentType.WORKER,
                state_type=StateType.PERFORMANCE,
                timestamp=timestamp,
                data=data,
                memory_usage=memory.percent,
                health_score=health_score,
                is_healthy=memory.percent < 85.0
            )
            
        except Exception:
            return None
    
    def _collect_gpu_metrics(self, timestamp: float) -> List[ComponentState]:
        """Collect GPU performance metrics."""
        states = []
        
        if not HAS_TORCH and not HAS_NVML and not HAS_GPUTIL:
            return states
        
        try:
            if HAS_NVML:
                # Use NVML for detailed GPU metrics
                device_count = nvml.nvmlDeviceGetCount()
                
                for i in range(device_count):
                    handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    
                    # GPU utilization
                    util = nvml.nvmlDeviceGetUtilizationRates(handle)
                    
                    # Memory info
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(handle)
                    
                    # Temperature
                    temp = nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU)
                    
                    # Power
                    power = nvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    
                    data = {
                        'gpu_utilization': util.gpu,
                        'memory_utilization': util.memory,
                        'memory_total_gb': mem_info.total / (1024**3),
                        'memory_used_gb': mem_info.used / (1024**3),
                        'memory_free_gb': mem_info.free / (1024**3),
                        'memory_percent': (mem_info.used / mem_info.total) * 100,
                        'temperature_c': temp,
                        'power_watts': power,
                    }
                    
                    # Health score based on multiple factors
                    health_factors = [
                        1.0 - (util.gpu / 100.0),  # Lower is better for utilization
                        1.0 - (mem_info.used / mem_info.total),  # Lower memory usage is better
                        1.0 - max(0, (temp - 60) / 40.0),  # Penalty for high temperature
                    ]
                    health_score = np.mean(health_factors)
                    
                    state = ComponentState(
                        component_id=f"gpu_{i}",
                        component_type=ComponentType.WORKER,
                        state_type=StateType.PERFORMANCE,
                        timestamp=timestamp,
                        data=data,
                        gpu_usage=util.gpu,
                        memory_usage=data['memory_percent'],
                        health_score=health_score,
                        is_healthy=temp < 80 and util.gpu < 95
                    )
                    states.append(state)
            
            elif HAS_TORCH:
                # Use PyTorch for basic GPU metrics
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    
                    for i in range(device_count):
                        # Memory info
                        memory_allocated = torch.cuda.memory_allocated(i)
                        memory_reserved = torch.cuda.memory_reserved(i)
                        memory_total = torch.cuda.get_device_properties(i).total_memory
                        
                        data = {
                            'memory_allocated_gb': memory_allocated / (1024**3),
                            'memory_reserved_gb': memory_reserved / (1024**3),
                            'memory_total_gb': memory_total / (1024**3),
                            'memory_percent': (memory_allocated / memory_total) * 100,
                        }
                        
                        health_score = 1.0 - (memory_allocated / memory_total)
                        
                        state = ComponentState(
                            component_id=f"gpu_{i}",
                            component_type=ComponentType.WORKER,
                            state_type=StateType.PERFORMANCE,
                            timestamp=timestamp,
                            data=data,
                            memory_usage=data['memory_percent'],
                            health_score=health_score,
                            is_healthy=data['memory_percent'] < 90.0
                        )
                        states.append(state)
        
        except Exception:
            pass
        
        return states
    
    def _collect_network_metrics(self, timestamp: float) -> Optional[ComponentState]:
        """Collect network performance metrics."""
        try:
            net_io = psutil.net_io_counters()
            
            data = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'errin': net_io.errin,
                'errout': net_io.errout,
                'dropin': net_io.dropin,
                'dropout': net_io.dropout,
            }
            
            # Calculate rates if we have previous data
            if self._last_network_io:
                time_delta = timestamp - self._last_network_io['timestamp']
                if time_delta > 0:
                    data['bytes_sent_rate'] = (net_io.bytes_sent - self._last_network_io['bytes_sent']) / time_delta
                    data['bytes_recv_rate'] = (net_io.bytes_recv - self._last_network_io['bytes_recv']) / time_delta
            
            self._last_network_io = {
                'timestamp': timestamp,
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
            }
            
            # Health score based on error rates
            total_packets = net_io.packets_sent + net_io.packets_recv
            total_errors = net_io.errin + net_io.errout + net_io.dropin + net_io.dropout
            error_rate = total_errors / max(total_packets, 1)
            health_score = 1.0 - min(error_rate * 100, 1.0)
            
            return ComponentState(
                component_id="system_network",
                component_type=ComponentType.API_SERVER,
                state_type=StateType.PERFORMANCE,
                timestamp=timestamp,
                data=data,
                health_score=health_score,
                is_healthy=error_rate < 0.01
            )
            
        except Exception:
            return None


class ResourceCollector(BaseCollector):
    """Collector for resource utilization with minimal overhead."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._process = psutil.Process()
        
    def collect(self) -> List[ComponentState]:
        """Collect resource utilization metrics."""
        if not self.should_collect():
            return []
        
        with PerformanceTimer() as timer:
            states = []
            current_time = time.time()
            
            # Process-specific metrics
            process_state = self._collect_process_metrics(current_time)
            if process_state:
                states.append(process_state)
            
            # File descriptor metrics
            fd_state = self._collect_fd_metrics(current_time)
            if fd_state:
                states.append(fd_state)
            
            # Disk metrics
            disk_state = self._collect_disk_metrics(current_time)
            if disk_state:
                states.append(disk_state)
        
        self._record_collection_time(timer.elapsed_us)
        return states
    
    def _collect_process_metrics(self, timestamp: float) -> Optional[ComponentState]:
        """Collect metrics for the current process."""
        try:
            # CPU and memory usage
            cpu_percent = self._process.cpu_percent()
            memory_info = self._process.memory_info()
            memory_percent = self._process.memory_percent()
            
            # Thread count
            num_threads = self._process.num_threads()
            
            # File descriptors
            num_fds = self._process.num_fds() if hasattr(self._process, 'num_fds') else 0
            
            data = {
                'pid': self._process.pid,
                'cpu_percent': cpu_percent,
                'memory_rss_mb': memory_info.rss / (1024**2),
                'memory_vms_mb': memory_info.vms / (1024**2),
                'memory_percent': memory_percent,
                'num_threads': num_threads,
                'num_fds': num_fds,
            }
            
            # Health score based on resource usage
            health_factors = [
                1.0 - min(cpu_percent / 100.0, 1.0),
                1.0 - min(memory_percent / 100.0, 1.0),
                1.0 - min(num_threads / 1000.0, 1.0),  # Assume 1000 threads is concerning
            ]
            health_score = np.mean(health_factors)
            
            return ComponentState(
                component_id="vllm_process",
                component_type=ComponentType.ENGINE,
                state_type=StateType.RESOURCE,
                timestamp=timestamp,
                data=data,
                cpu_usage=cpu_percent,
                memory_usage=memory_percent,
                health_score=health_score,
                is_healthy=cpu_percent < 95.0 and memory_percent < 90.0
            )
            
        except Exception:
            return None
    
    def _collect_fd_metrics(self, timestamp: float) -> Optional[ComponentState]:
        """Collect file descriptor metrics."""
        try:
            # Get file descriptor count
            num_fds = self._process.num_fds() if hasattr(self._process, 'num_fds') else 0
            
            # Get system limits
            import resource
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            
            data = {
                'num_fds': num_fds,
                'fd_soft_limit': soft_limit,
                'fd_hard_limit': hard_limit,
                'fd_usage_percent': (num_fds / soft_limit) * 100 if soft_limit > 0 else 0,
            }
            
            # Health score based on FD usage
            fd_usage = num_fds / max(soft_limit, 1)
            health_score = 1.0 - min(fd_usage, 1.0)
            
            return ComponentState(
                component_id="file_descriptors",
                component_type=ComponentType.ENGINE,
                state_type=StateType.RESOURCE,
                timestamp=timestamp,
                data=data,
                health_score=health_score,
                is_healthy=fd_usage < 0.8
            )
            
        except Exception:
            return None
    
    def _collect_disk_metrics(self, timestamp: float) -> Optional[ComponentState]:
        """Collect disk utilization metrics."""
        try:
            # Get disk usage for current directory
            disk_usage = psutil.disk_usage('.')
            
            # Get I/O statistics
            disk_io = psutil.disk_io_counters()
            
            data = {
                'disk_total_gb': disk_usage.total / (1024**3),
                'disk_used_gb': disk_usage.used / (1024**3),
                'disk_free_gb': disk_usage.free / (1024**3),
                'disk_percent': (disk_usage.used / disk_usage.total) * 100,
            }
            
            if disk_io:
                data.update({
                    'read_bytes': disk_io.read_bytes,
                    'write_bytes': disk_io.write_bytes,
                    'read_count': disk_io.read_count,
                    'write_count': disk_io.write_count,
                })
            
            # Health score based on disk usage
            disk_percent = data['disk_percent']
            if disk_percent > 95:
                health_score = 0.1
            elif disk_percent > 85:
                health_score = 0.5
            elif disk_percent > 70:
                health_score = 0.8
            else:
                health_score = 1.0
            
            return ComponentState(
                component_id="disk_usage",
                component_type=ComponentType.ENGINE,
                state_type=StateType.RESOURCE,
                timestamp=timestamp,
                data=data,
                health_score=health_score,
                is_healthy=disk_percent < 90.0
            )
            
        except Exception:
            return None


class ErrorCollector(BaseCollector):
    """Collector for errors and exceptions with stack trace analysis."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._error_counts = {}
        self._last_errors = {}
        
        # Hook into Python's exception handling
        self._original_excepthook = None
        self._install_exception_hook()
    
    def _install_exception_hook(self):
        """Install exception hook to capture errors."""
        import sys
        
        self._original_excepthook = sys.excepthook
        
        def exception_hook(exc_type, exc_value, exc_traceback):
            # Record the exception
            self._record_exception(exc_type, exc_value, exc_traceback)
            # Call original hook
            if self._original_excepthook:
                self._original_excepthook(exc_type, exc_value, exc_traceback)
        
        sys.excepthook = exception_hook
    
    def _record_exception(self, exc_type, exc_value, exc_traceback):
        """Record an exception for later collection."""
        import traceback
        
        error_key = f"{exc_type.__name__}:{str(exc_value)[:100]}"
        current_time = time.time()
        
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        self._last_errors[error_key] = {
            'timestamp': current_time,
            'type': exc_type.__name__,
            'message': str(exc_value),
            'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback),
        }
    
    def collect(self) -> List[ComponentState]:
        """Collect error and exception information."""
        if not self.should_collect():
            return []
        
        with PerformanceTimer() as timer:
            states = []
            current_time = time.time()
            
            # Create error state if we have errors
            if self._error_counts:
                total_errors = sum(self._error_counts.values())
                recent_errors = {
                    k: v for k, v in self._last_errors.items()
                    if current_time - v['timestamp'] < 300  # Last 5 minutes
                }
                
                data = {
                    'total_error_count': total_errors,
                    'recent_error_count': len(recent_errors),
                    'error_types': list(self._error_counts.keys()),
                    'recent_errors': list(recent_errors.values())[:10],  # Last 10 errors
                }
                
                # Health score based on recent error rate
                error_rate = len(recent_errors) / 300.0  # Errors per second
                health_score = max(0.0, 1.0 - min(error_rate * 100, 1.0))
                
                state = ComponentState(
                    component_id="error_tracker",
                    component_type=ComponentType.ENGINE,
                    state_type=StateType.ERROR,
                    timestamp=current_time,
                    data=data,
                    health_score=health_score,
                    is_healthy=len(recent_errors) == 0,
                    error_count=total_errors,
                    last_error=list(recent_errors.values())[0]['message'] if recent_errors else None
                )
                states.append(state)
        
        self._record_collection_time(timer.elapsed_us)
        return states
    
    def cleanup(self):
        """Restore original exception hook."""
        import sys
        if self._original_excepthook:
            sys.excepthook = self._original_excepthook


class RequestCollector(BaseCollector):
    """Collector for request lifecycle and performance metrics."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._request_stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'active_requests': 0,
            'total_latency': 0.0,
            'total_tokens': 0,
        }
        self._request_history = []
        self._lock = threading.Lock()
    
    def record_request_start(self, request_id: str, request_data: Dict[str, Any]):
        """Record the start of a request."""
        with self._lock:
            self._request_stats['total_requests'] += 1
            self._request_stats['active_requests'] += 1
            
            request_info = {
                'request_id': request_id,
                'start_time': time.time(),
                'data': request_data,
                'status': 'active'
            }
            self._request_history.append(request_info)
    
    def record_request_completion(self, request_id: str, tokens_generated: int, success: bool):
        """Record the completion of a request."""
        with self._lock:
            current_time = time.time()
            self._request_stats['active_requests'] = max(0, self._request_stats['active_requests'] - 1)
            
            if success:
                self._request_stats['completed_requests'] += 1
                self._request_stats['total_tokens'] += tokens_generated
            else:
                self._request_stats['failed_requests'] += 1
            
            # Find and update request in history
            for req in reversed(self._request_history):
                if req['request_id'] == request_id and req['status'] == 'active':
                    req['status'] = 'completed' if success else 'failed'
                    req['end_time'] = current_time
                    req['latency'] = current_time - req['start_time']
                    req['tokens_generated'] = tokens_generated
                    self._request_stats['total_latency'] += req['latency']
                    break
    
    def collect(self) -> List[ComponentState]:
        """Collect request performance metrics."""
        if not self.should_collect():
            return []
        
        with PerformanceTimer() as timer:
            current_time = time.time()
            
            with self._lock:
                stats = self._request_stats.copy()
                recent_requests = [
                    req for req in self._request_history
                    if current_time - req['start_time'] < 300  # Last 5 minutes
                ]
            
            # Calculate metrics
            total_requests = stats['total_requests']
            success_rate = (stats['completed_requests'] / max(total_requests, 1)) * 100
            
            avg_latency = 0.0
            if stats['completed_requests'] > 0:
                avg_latency = stats['total_latency'] / stats['completed_requests']
            
            tokens_per_second = 0.0
            if stats['total_latency'] > 0:
                tokens_per_second = stats['total_tokens'] / stats['total_latency']
            
            data = {
                'total_requests': total_requests,
                'completed_requests': stats['completed_requests'],
                'failed_requests': stats['failed_requests'],
                'active_requests': stats['active_requests'],
                'success_rate_percent': success_rate,
                'average_latency_s': avg_latency,
                'tokens_per_second': tokens_per_second,
                'recent_requests_5min': len(recent_requests),
            }
            
            # Health score based on success rate and active requests
            health_factors = [
                success_rate / 100.0,  # Higher success rate is better
                1.0 - min(stats['active_requests'] / 100.0, 1.0),  # Lower active requests is better
                1.0 - min(avg_latency / 60.0, 1.0),  # Lower latency is better
            ]
            health_score = np.mean(health_factors)
            
            state = ComponentState(
                component_id="request_handler",
                component_type=ComponentType.REQUEST_HANDLER,
                state_type=StateType.REQUEST,
                timestamp=current_time,
                data=data,
                health_score=health_score,
                is_healthy=success_rate > 95.0 and stats['active_requests'] < 50,
                requests_processed=total_requests,
                average_latency_ms=avg_latency * 1000,
                throughput_rps=tokens_per_second
            )
            
        self._record_collection_time(timer.elapsed_us)
        return [state]