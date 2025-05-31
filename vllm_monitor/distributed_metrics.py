"""
Distributed Metrics Collection for vLLM Monitoring.

This module provides comprehensive metrics collection for distributed vLLM deployments,
including multi-worker scenarios, heterogeneous hardware, and various parallelism configurations.
"""

import time
import json
import socket
import threading
import weakref
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from enum import Enum, auto
from collections import defaultdict, deque
import logging
import psutil
import os
import subprocess

try:
    import torch
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

from .core import CircularBuffer, get_logger
from .plugin_system import PluginInterface, PluginMetadata, PluginType, SimplePlugin


class WorkerRole(Enum):
    """Roles in distributed setup"""
    MASTER = auto()
    WORKER = auto()
    PREFILL = auto()
    DECODE = auto()
    ROUTER = auto()
    COORDINATOR = auto()


class ParallelismType(Enum):
    """Types of parallelism in vLLM"""
    TENSOR_PARALLEL = auto()
    PIPELINE_PARALLEL = auto()
    DATA_PARALLEL = auto()
    SEQUENCE_PARALLEL = auto()
    EXPERT_PARALLEL = auto()


class CommunicationType(Enum):
    """Types of communication in distributed setup"""
    NCCL = auto()
    GLOO = auto()
    MPI = auto()
    CUSTOM_ALLREDUCE = auto()
    SHM = auto()  # Shared memory


@dataclass
class WorkerMetrics:
    """Metrics for a single worker"""
    worker_id: str
    role: WorkerRole
    hostname: str
    pid: int
    # Resource metrics
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    gpu_id: Optional[int] = None
    gpu_utilization: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    gpu_temperature: float = 0.0
    # Performance metrics
    throughput: float = 0.0
    latency_ms: float = 0.0
    queue_size: int = 0
    active_requests: int = 0
    # Communication metrics
    bytes_sent: int = 0
    bytes_received: int = 0
    comm_latency_ms: float = 0.0
    # Error metrics
    error_count: int = 0
    last_error: Optional[str] = None
    # Timestamps
    last_update: float = field(default_factory=time.time)
    uptime: float = 0.0


@dataclass
class CommunicationMetrics:
    """Metrics for inter-worker communication"""
    src_worker: str
    dst_worker: str
    comm_type: CommunicationType
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss_rate: float = 0.0
    messages_sent: int = 0
    bytes_sent: int = 0
    errors: int = 0
    last_update: float = field(default_factory=time.time)


@dataclass
class ParallelismMetrics:
    """Metrics for different parallelism types"""
    parallelism_type: ParallelismType
    world_size: int
    local_rank: int
    global_rank: int
    # Performance
    sync_time_ms: float = 0.0
    idle_time_ms: float = 0.0
    compute_time_ms: float = 0.0
    efficiency: float = 0.0
    # Load balancing
    load_imbalance: float = 0.0
    stragglers: List[str] = field(default_factory=list)
    # Communication
    allreduce_time_ms: float = 0.0
    broadcast_time_ms: float = 0.0
    p2p_time_ms: float = 0.0


@dataclass
class HardwareTopology:
    """Hardware topology information"""
    node_id: str
    hostname: str
    cpu_info: Dict[str, Any]
    gpu_info: List[Dict[str, Any]]
    network_info: Dict[str, Any]
    numa_nodes: List[Dict[str, Any]]
    pcie_topology: Dict[str, Any]
    interconnect_type: Optional[str] = None  # NVLink, PCIe, etc.


class DistributedMetricsCollector:
    """
    Comprehensive metrics collector for distributed vLLM deployments.
    
    Features:
    - Multi-worker metrics collection
    - Hardware topology awareness
    - Communication pattern analysis
    - Parallelism efficiency tracking
    - Heterogeneous hardware support
    """
    
    def __init__(self, 
                 node_id: str = None,
                 role: WorkerRole = WorkerRole.WORKER):
        self.logger = get_logger()
        self._lock = threading.RLock()
        
        # Identity
        self.node_id = node_id or socket.gethostname()
        self.role = role
        self.pid = os.getpid()
        
        # Worker tracking
        self.local_workers: Dict[str, WorkerMetrics] = {}
        self.remote_workers: Dict[str, WorkerMetrics] = {}
        self.worker_history = CircularBuffer(10000)
        
        # Communication tracking
        self.comm_metrics: Dict[Tuple[str, str], CommunicationMetrics] = {}
        self.comm_history = CircularBuffer(5000)
        
        # Parallelism tracking
        self.parallelism_metrics: Dict[ParallelismType, ParallelismMetrics] = {}
        self.sync_events = CircularBuffer(1000)
        
        # Hardware topology
        self.hardware_topology: Optional[HardwareTopology] = None
        self.heterogeneous_config: Dict[str, Any] = {}
        
        # Network monitoring
        self.network_interfaces: Dict[str, Dict[str, Any]] = {}
        self.network_stats_baseline: Dict[str, Any] = {}
        
        # GPU monitoring
        self.gpu_handles: Dict[int, Any] = {}
        self._init_gpu_monitoring()
        
        # Collection threads
        self._collectors: Dict[str, threading.Thread] = {}
        self._running = False
        
        # Metrics aggregation
        self.aggregated_metrics = CircularBuffer(1000)
        self.anomaly_buffer = CircularBuffer(500)
        
        # Initialize hardware topology
        self._detect_hardware_topology()
        
        self.logger.info(f"Distributed metrics collector initialized on {self.node_id}")

    def _init_gpu_monitoring(self):
        """Initialize GPU monitoring if available"""
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                for i in range(device_count):
                    self.gpu_handles[i] = pynvml.nvmlDeviceGetHandleByIndex(i)
                self.logger.info(f"Initialized GPU monitoring for {device_count} devices")
            except Exception as e:
                self.logger.warning(f"Failed to initialize NVML: {e}")

    def _detect_hardware_topology(self):
        """Detect hardware topology of the current node"""
        try:
            cpu_info = {
                "count": psutil.cpu_count(logical=False),
                "logical_count": psutil.cpu_count(logical=True),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else {},
                "architecture": os.uname().machine
            }
            
            gpu_info = []
            if TORCH_AVAILABLE and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append({
                        "index": i,
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "total_memory_mb": props.total_memory / (1024**2),
                        "multi_processor_count": props.multi_processor_count
                    })
            
            network_info = {}
            for iface, addrs in psutil.net_if_addrs().items():
                if iface != 'lo':  # Skip loopback
                    network_info[iface] = {
                        "addresses": [addr.address for addr in addrs],
                        "is_up": iface in psutil.net_if_stats()
                    }
            
            # NUMA node detection
            numa_nodes = []
            try:
                numa_path = "/sys/devices/system/node/"
                if os.path.exists(numa_path):
                    for node in os.listdir(numa_path):
                        if node.startswith("node"):
                            numa_nodes.append({
                                "id": int(node[4:]),
                                "cpus": self._get_numa_cpus(node)
                            })
            except:
                pass
            
            # Detect interconnect type
            interconnect = None
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Check for NVLink
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "nvlink", "-s"],
                        capture_output=True,
                        text=True
                    )
                    if "Link" in result.stdout:
                        interconnect = "NVLink"
                except:
                    pass
            
            self.hardware_topology = HardwareTopology(
                node_id=self.node_id,
                hostname=socket.gethostname(),
                cpu_info=cpu_info,
                gpu_info=gpu_info,
                network_info=network_info,
                numa_nodes=numa_nodes,
                pcie_topology={},  # Would need lspci parsing
                interconnect_type=interconnect
            )
            
        except Exception as e:
            self.logger.error(f"Failed to detect hardware topology: {e}")

    def _get_numa_cpus(self, node: str) -> List[int]:
        """Get CPU list for a NUMA node"""
        try:
            with open(f"/sys/devices/system/node/{node}/cpulist", 'r') as f:
                cpulist = f.read().strip()
                # Parse CPU ranges like "0-7,16-23"
                cpus = []
                for part in cpulist.split(','):
                    if '-' in part:
                        start, end = map(int, part.split('-'))
                        cpus.extend(range(start, end + 1))
                    else:
                        cpus.append(int(part))
                return cpus
        except:
            return []

    def register_worker(self, 
                       worker_id: str,
                       role: WorkerRole,
                       hostname: str,
                       pid: int,
                       gpu_id: Optional[int] = None) -> None:
        """Register a worker for monitoring"""
        with self._lock:
            metrics = WorkerMetrics(
                worker_id=worker_id,
                role=role,
                hostname=hostname,
                pid=pid,
                gpu_id=gpu_id
            )
            
            if hostname == self.node_id:
                self.local_workers[worker_id] = metrics
            else:
                self.remote_workers[worker_id] = metrics
            
            self.logger.info(f"Registered worker {worker_id} ({role.name}) on {hostname}")

    def update_worker_metrics(self,
                            worker_id: str,
                            metrics: Dict[str, Any]) -> None:
        """Update metrics for a worker"""
        with self._lock:
            worker = self.local_workers.get(worker_id) or self.remote_workers.get(worker_id)
            if not worker:
                self.logger.warning(f"Unknown worker: {worker_id}")
                return
            
            # Update metrics
            for key, value in metrics.items():
                if hasattr(worker, key):
                    setattr(worker, key, value)
            
            worker.last_update = time.time()
            
            # Store in history
            self.worker_history.append({
                "timestamp": time.time(),
                "worker_id": worker_id,
                "metrics": asdict(worker)
            })

    def collect_local_metrics(self) -> Dict[str, WorkerMetrics]:
        """Collect metrics for local workers"""
        metrics = {}
        
        with self._lock:
            for worker_id, worker in self.local_workers.items():
                try:
                    # CPU and memory metrics
                    if worker.pid:
                        try:
                            process = psutil.Process(worker.pid)
                            worker.cpu_percent = process.cpu_percent()
                            worker.memory_percent = process.memory_percent()
                            worker.memory_used_mb = process.memory_info().rss / (1024**2)
                        except psutil.NoSuchProcess:
                            self.logger.warning(f"Process {worker.pid} not found")
                    
                    # GPU metrics
                    if worker.gpu_id is not None and worker.gpu_id in self.gpu_handles:
                        gpu_metrics = self._collect_gpu_metrics(worker.gpu_id)
                        worker.gpu_utilization = gpu_metrics["utilization"]
                        worker.gpu_memory_percent = gpu_metrics["memory_percent"]
                        worker.gpu_memory_used_mb = gpu_metrics["memory_used_mb"]
                        worker.gpu_temperature = gpu_metrics["temperature"]
                    
                    # Update timestamp
                    worker.last_update = time.time()
                    worker.uptime = time.time() - worker.last_update
                    
                    metrics[worker_id] = worker
                    
                except Exception as e:
                    self.logger.error(f"Failed to collect metrics for {worker_id}: {e}")
        
        return metrics

    def _collect_gpu_metrics(self, gpu_id: int) -> Dict[str, float]:
        """Collect GPU metrics using NVML"""
        metrics = {
            "utilization": 0.0,
            "memory_percent": 0.0,
            "memory_used_mb": 0.0,
            "temperature": 0.0
        }
        
        if gpu_id not in self.gpu_handles:
            return metrics
        
        try:
            handle = self.gpu_handles[gpu_id]
            
            # Utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            metrics["utilization"] = util.gpu
            
            # Memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics["memory_percent"] = (mem_info.used / mem_info.total) * 100
            metrics["memory_used_mb"] = mem_info.used / (1024**2)
            
            # Temperature
            metrics["temperature"] = pynvml.nvmlDeviceGetTemperature(
                handle, 
                pynvml.NVML_TEMPERATURE_GPU
            )
            
        except Exception as e:
            self.logger.debug(f"Failed to get GPU metrics: {e}")
        
        return metrics

    def collect_communication_metrics(self,
                                    src_worker: str,
                                    dst_worker: str,
                                    comm_type: CommunicationType) -> CommunicationMetrics:
        """Collect communication metrics between workers"""
        key = (src_worker, dst_worker)
        
        with self._lock:
            if key not in self.comm_metrics:
                self.comm_metrics[key] = CommunicationMetrics(
                    src_worker=src_worker,
                    dst_worker=dst_worker,
                    comm_type=comm_type
                )
            
            metrics = self.comm_metrics[key]
            
            # Update metrics based on actual communication
            # This would be integrated with vLLM's communication layer
            
            metrics.last_update = time.time()
            
            # Store in history
            self.comm_history.append({
                "timestamp": time.time(),
                "src": src_worker,
                "dst": dst_worker,
                "metrics": asdict(metrics)
            })
            
            return metrics

    def update_parallelism_metrics(self,
                                 parallelism_type: ParallelismType,
                                 metrics: Dict[str, Any]) -> None:
        """Update parallelism-specific metrics"""
        with self._lock:
            if parallelism_type not in self.parallelism_metrics:
                self.parallelism_metrics[parallelism_type] = ParallelismMetrics(
                    parallelism_type=parallelism_type,
                    world_size=metrics.get("world_size", 1),
                    local_rank=metrics.get("local_rank", 0),
                    global_rank=metrics.get("global_rank", 0)
                )
            
            pm = self.parallelism_metrics[parallelism_type]
            
            # Update metrics
            for key, value in metrics.items():
                if hasattr(pm, key):
                    setattr(pm, key, value)
            
            # Calculate efficiency
            if pm.compute_time_ms > 0:
                total_time = pm.compute_time_ms + pm.sync_time_ms + pm.idle_time_ms
                pm.efficiency = pm.compute_time_ms / total_time if total_time > 0 else 0
            
            # Detect stragglers
            if "worker_times" in metrics:
                avg_time = sum(metrics["worker_times"].values()) / len(metrics["worker_times"])
                threshold = avg_time * 1.2  # 20% slower than average
                pm.stragglers = [
                    worker for worker, time in metrics["worker_times"].items()
                    if time > threshold
                ]

    def collect_network_metrics(self) -> Dict[str, Any]:
        """Collect network interface metrics"""
        metrics = {}
        
        try:
            # Get current stats
            net_io = psutil.net_io_counters(pernic=True)
            
            for iface, stats in net_io.items():
                if iface == 'lo':  # Skip loopback
                    continue
                
                current = {
                    "bytes_sent": stats.bytes_sent,
                    "bytes_recv": stats.bytes_recv,
                    "packets_sent": stats.packets_sent,
                    "packets_recv": stats.packets_recv,
                    "errors_in": stats.errin,
                    "errors_out": stats.errout,
                    "drop_in": stats.dropin,
                    "drop_out": stats.dropout
                }
                
                # Calculate rates if we have baseline
                if iface in self.network_stats_baseline:
                    baseline = self.network_stats_baseline[iface]
                    time_delta = time.time() - baseline["timestamp"]
                    
                    if time_delta > 0:
                        rates = {
                            "send_rate_mbps": (current["bytes_sent"] - baseline["bytes_sent"]) * 8 / (time_delta * 1e6),
                            "recv_rate_mbps": (current["bytes_recv"] - baseline["bytes_recv"]) * 8 / (time_delta * 1e6),
                            "packet_loss_rate": (current["drop_in"] + current["drop_out"] - 
                                               baseline["drop_in"] - baseline["drop_out"]) / time_delta
                        }
                        current.update(rates)
                
                # Update baseline
                current["timestamp"] = time.time()
                self.network_stats_baseline[iface] = current.copy()
                
                metrics[iface] = current
                
        except Exception as e:
            self.logger.error(f"Failed to collect network metrics: {e}")
        
        return metrics

    def detect_hardware_heterogeneity(self) -> Dict[str, Any]:
        """Detect and report hardware heterogeneity across workers"""
        heterogeneity = {
            "is_heterogeneous": False,
            "gpu_types": set(),
            "gpu_memory_sizes": set(),
            "cpu_counts": set(),
            "memory_sizes": set(),
            "variations": []
        }
        
        with self._lock:
            all_workers = list(self.local_workers.values()) + list(self.remote_workers.values())
            
            if not all_workers:
                return heterogeneity
            
            # Collect unique hardware configurations
            for worker in all_workers:
                if worker.gpu_id is not None:
                    # Get GPU info from topology
                    if self.hardware_topology and worker.gpu_id < len(self.hardware_topology.gpu_info):
                        gpu = self.hardware_topology.gpu_info[worker.gpu_id]
                        heterogeneity["gpu_types"].add(gpu["name"])
                        heterogeneity["gpu_memory_sizes"].add(gpu["total_memory_mb"])
            
            # Check for heterogeneity
            if len(heterogeneity["gpu_types"]) > 1:
                heterogeneity["is_heterogeneous"] = True
                heterogeneity["variations"].append("Multiple GPU types detected")
            
            if len(heterogeneity["gpu_memory_sizes"]) > 1:
                heterogeneity["is_heterogeneous"] = True
                heterogeneity["variations"].append("Different GPU memory sizes detected")
        
        return heterogeneity

    def get_distributed_summary(self) -> Dict[str, Any]:
        """Get comprehensive distributed metrics summary"""
        with self._lock:
            local_metrics = self.collect_local_metrics()
            network_metrics = self.collect_network_metrics()
            heterogeneity = self.detect_hardware_heterogeneity()
            
            # Aggregate worker metrics
            total_workers = len(self.local_workers) + len(self.remote_workers)
            active_workers = sum(
                1 for w in list(self.local_workers.values()) + list(self.remote_workers.values())
                if time.time() - w.last_update < 60  # Active in last minute
            )
            
            # Aggregate GPU metrics
            total_gpu_util = sum(w.gpu_utilization for w in local_metrics.values() if w.gpu_id is not None)
            avg_gpu_util = total_gpu_util / len([w for w in local_metrics.values() if w.gpu_id is not None]) if local_metrics else 0
            
            # Communication summary
            total_comm_bandwidth = sum(
                cm.bandwidth_mbps for cm in self.comm_metrics.values()
            )
            avg_comm_latency = sum(
                cm.latency_ms for cm in self.comm_metrics.values()
            ) / len(self.comm_metrics) if self.comm_metrics else 0
            
            # Parallelism summary
            parallelism_summary = {}
            for ptype, metrics in self.parallelism_metrics.items():
                parallelism_summary[ptype.name] = {
                    "world_size": metrics.world_size,
                    "efficiency": metrics.efficiency,
                    "sync_overhead_ms": metrics.sync_time_ms,
                    "stragglers": len(metrics.stragglers)
                }
            
            return {
                "node_id": self.node_id,
                "timestamp": time.time(),
                "hardware_topology": asdict(self.hardware_topology) if self.hardware_topology else None,
                "workers": {
                    "total": total_workers,
                    "active": active_workers,
                    "local": len(self.local_workers),
                    "remote": len(self.remote_workers),
                    "by_role": self._count_workers_by_role()
                },
                "resources": {
                    "avg_gpu_utilization": avg_gpu_util,
                    "total_gpu_memory_used_gb": sum(
                        w.gpu_memory_used_mb / 1024 for w in local_metrics.values() 
                        if w.gpu_id is not None
                    ),
                    "avg_cpu_percent": sum(w.cpu_percent for w in local_metrics.values()) / len(local_metrics) if local_metrics else 0,
                    "total_memory_used_gb": sum(w.memory_used_mb / 1024 for w in local_metrics.values())
                },
                "communication": {
                    "total_bandwidth_mbps": total_comm_bandwidth,
                    "avg_latency_ms": avg_comm_latency,
                    "active_connections": len(self.comm_metrics),
                    "network_interfaces": network_metrics
                },
                "parallelism": parallelism_summary,
                "heterogeneity": heterogeneity,
                "health": {
                    "total_errors": sum(w.error_count for w in local_metrics.values()),
                    "workers_with_errors": sum(1 for w in local_metrics.values() if w.error_count > 0)
                }
            }

    def _count_workers_by_role(self) -> Dict[str, int]:
        """Count workers by role"""
        counts = defaultdict(int)
        with self._lock:
            for worker in list(self.local_workers.values()) + list(self.remote_workers.values()):
                counts[worker.role.name] += 1
        return dict(counts)

    def start_collection(self, interval: float = 1.0) -> None:
        """Start metrics collection threads"""
        if self._running:
            return
        
        self._running = True
        
        # Start local metrics collector
        self._collectors["local"] = threading.Thread(
            target=self._collection_loop,
            args=(self.collect_local_metrics, interval),
            daemon=True
        )
        self._collectors["local"].start()
        
        # Start network metrics collector
        self._collectors["network"] = threading.Thread(
            target=self._collection_loop,
            args=(self.collect_network_metrics, interval * 5),  # Less frequent
            daemon=True
        )
        self._collectors["network"].start()
        
        self.logger.info("Started distributed metrics collection")

    def stop_collection(self) -> None:
        """Stop metrics collection"""
        self._running = False
        for thread in self._collectors.values():
            thread.join(timeout=5)
        self._collectors.clear()
        self.logger.info("Stopped distributed metrics collection")

    def _collection_loop(self, collect_func: Callable, interval: float) -> None:
        """Generic collection loop"""
        while self._running:
            try:
                collect_func()
            except Exception as e:
                self.logger.error(f"Collection error: {e}")
            time.sleep(interval)

    def create_plugin(self) -> PluginInterface:
        """Create a plugin interface for this collector"""
        def collect():
            return self.get_distributed_summary()
        
        return SimplePlugin(
            name="distributed_metrics_collector",
            plugin_type=PluginType.COLLECTOR,
            execute_func=collect,
            description="Collects comprehensive metrics for distributed vLLM deployments",
            capabilities=["distributed", "multi-gpu", "network", "parallelism"],
            hardware_requirements={"gpu": False}  # Can run without GPU
        )


# Specialized collectors for different scenarios

class TensorParallelMetricsCollector:
    """Specialized collector for tensor parallel setups"""
    
    def __init__(self, tp_size: int, tp_rank: int):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.logger = get_logger()
        
        # TP-specific metrics
        self.allreduce_times = CircularBuffer(1000)
        self.layer_sync_times = defaultdict(list)
        self.communication_volume = CircularBuffer(1000)
        
    def record_allreduce(self, size_bytes: int, time_ms: float) -> None:
        """Record an allreduce operation"""
        self.allreduce_times.append({
            "timestamp": time.time(),
            "size_bytes": size_bytes,
            "time_ms": time_ms,
            "bandwidth_gbps": (size_bytes * 8) / (time_ms * 1e6)
        })
    
    def record_layer_sync(self, layer_name: str, time_ms: float) -> None:
        """Record layer synchronization time"""
        self.layer_sync_times[layer_name].append(time_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get TP metrics summary"""
        allreduce_data = list(self.allreduce_times.buffer)
        if not allreduce_data:
            return {}
        
        return {
            "tp_size": self.tp_size,
            "tp_rank": self.tp_rank,
            "allreduce": {
                "count": len(allreduce_data),
                "avg_time_ms": sum(d["time_ms"] for d in allreduce_data) / len(allreduce_data),
                "avg_bandwidth_gbps": sum(d["bandwidth_gbps"] for d in allreduce_data) / len(allreduce_data),
                "total_volume_gb": sum(d["size_bytes"] for d in allreduce_data) / 1e9
            },
            "layer_sync": {
                layer: {
                    "avg_time_ms": sum(times) / len(times),
                    "max_time_ms": max(times),
                    "count": len(times)
                }
                for layer, times in self.layer_sync_times.items()
            }
        }


class PipelineParallelMetricsCollector:
    """Specialized collector for pipeline parallel setups"""
    
    def __init__(self, pp_size: int, pp_rank: int, num_stages: int):
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.num_stages = num_stages
        self.logger = get_logger()
        
        # PP-specific metrics
        self.pipeline_bubbles = CircularBuffer(1000)
        self.stage_times = defaultdict(list)
        self.micro_batch_times = CircularBuffer(1000)
        
    def record_pipeline_bubble(self, bubble_time_ms: float) -> None:
        """Record pipeline bubble time"""
        self.pipeline_bubbles.append({
            "timestamp": time.time(),
            "bubble_time_ms": bubble_time_ms,
            "stage": self.pp_rank
        })
    
    def record_stage_time(self, stage: int, compute_time_ms: float) -> None:
        """Record stage computation time"""
        self.stage_times[stage].append(compute_time_ms)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get PP metrics summary"""
        bubble_data = list(self.pipeline_bubbles.buffer)
        
        return {
            "pp_size": self.pp_size,
            "pp_rank": self.pp_rank,
            "num_stages": self.num_stages,
            "pipeline_efficiency": {
                "bubble_time_ms": sum(d["bubble_time_ms"] for d in bubble_data) / len(bubble_data) if bubble_data else 0,
                "stage_balance": self._calculate_stage_balance()
            },
            "stage_times": {
                stage: {
                    "avg_time_ms": sum(times) / len(times),
                    "variance": self._calculate_variance(times)
                }
                for stage, times in self.stage_times.items()
            }
        }
    
    def _calculate_stage_balance(self) -> float:
        """Calculate load balance across stages"""
        if not self.stage_times:
            return 1.0
        
        avg_times = [sum(times) / len(times) for times in self.stage_times.values()]
        if not avg_times:
            return 1.0
        
        mean = sum(avg_times) / len(avg_times)
        variance = sum((t - mean) ** 2 for t in avg_times) / len(avg_times)
        
        # Return balance score (1.0 is perfect balance)
        return 1.0 / (1.0 + variance / mean if mean > 0 else 1.0)
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)


# Example usage
"""
# Create distributed metrics collector
collector = DistributedMetricsCollector(node_id="node1", role=WorkerRole.MASTER)

# Register workers
collector.register_worker("worker1", WorkerRole.PREFILL, "node1", 12345, gpu_id=0)
collector.register_worker("worker2", WorkerRole.DECODE, "node1", 12346, gpu_id=1)
collector.register_worker("worker3", WorkerRole.DECODE, "node2", 12347, gpu_id=0)

# Start collection
collector.start_collection(interval=1.0)

# Update worker metrics
collector.update_worker_metrics("worker1", {
    "throughput": 150.5,
    "queue_size": 10,
    "active_requests": 5
})

# Update parallelism metrics
collector.update_parallelism_metrics(
    ParallelismType.TENSOR_PARALLEL,
    {
        "world_size": 4,
        "sync_time_ms": 2.5,
        "compute_time_ms": 50.0,
        "worker_times": {"worker1": 50, "worker2": 52, "worker3": 48, "worker4": 55}
    }
)

# Get summary
summary = collector.get_distributed_summary()
print(json.dumps(summary, indent=2))

# Create plugin for integration
plugin = collector.create_plugin()
"""