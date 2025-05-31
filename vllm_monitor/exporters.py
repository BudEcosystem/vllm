"""
Export and alerting system for vLLM monitoring data.

This module provides comprehensive export capabilities for metrics, logs, and alerts
with support for multiple backends and formats.
"""

import json
import time
import threading
import asyncio
import gzip
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
import logging
import os

from .core import (
    ComponentState, ComponentType, StateType, AlertLevel,
    MonitorConfig, PerformanceTimer
)

logger = logging.getLogger("vllm_monitor.exporters")


@dataclass
class ExportConfig:
    """Configuration for export operations."""
    export_interval_seconds: float = 60.0
    max_batch_size: int = 1000
    enable_compression: bool = True
    retention_hours: int = 24
    export_format: str = "json"  # json, csv, prometheus
    include_metadata: bool = True


@dataclass
class AlertConfig:
    """Configuration for alert generation."""
    enabled: bool = True
    min_alert_level: AlertLevel = AlertLevel.WARNING
    alert_cooldown_seconds: float = 300.0  # 5 minutes
    max_alerts_per_hour: int = 50
    webhook_url: Optional[str] = None
    email_recipients: List[str] = None
    slack_webhook: Optional[str] = None


class BaseExporter(ABC):
    """Base class for all data exporters."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self._export_count = 0
        self._failed_exports = 0
        self._last_export_time = 0.0
        self._lock = threading.RLock()
    
    @abstractmethod
    async def export_data(self, data: List[Dict[str, Any]]) -> bool:
        """Export data to the target system."""
        pass
    
    @abstractmethod
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics."""
        pass
    
    def should_export(self) -> bool:
        """Check if we should perform export based on interval."""
        current_time = time.time()
        return (current_time - self._last_export_time) >= self.config.export_interval_seconds
    
    def _record_export(self, success: bool) -> None:
        """Record export attempt for statistics."""
        with self._lock:
            self._export_count += 1
            if not success:
                self._failed_exports += 1
            self._last_export_time = time.time()


class MetricsExporter(BaseExporter):
    """Exporter for metrics data with support for multiple formats."""
    
    def __init__(self, 
                 config: ExportConfig,
                 output_directory: str = "./vllm_metrics"):
        super().__init__(config)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Metrics aggregation
        self._metrics_buffer: deque = deque(maxlen=10000)
        self._aggregated_metrics: Dict[str, Any] = defaultdict(list)
        
    async def export_data(self, data: List[Dict[str, Any]]) -> bool:
        """Export metrics data to files."""
        if not data:
            return True
        
        try:
            # Add data to buffer
            self._metrics_buffer.extend(data)
            
            # Aggregate metrics
            self._aggregate_metrics(data)
            
            # Export based on format
            if self.config.export_format == "json":
                success = await self._export_json(data)
            elif self.config.export_format == "csv":
                success = await self._export_csv(data)
            elif self.config.export_format == "prometheus":
                success = await self._export_prometheus(data)
            else:
                logger.error(f"Unsupported export format: {self.config.export_format}")
                success = False
            
            self._record_export(success)
            return success
            
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            self._record_export(False)
            return False
    
    def _aggregate_metrics(self, data: List[Dict[str, Any]]) -> None:
        """Aggregate metrics for summary reporting."""
        current_time = time.time()
        
        for item in data:
            if isinstance(item, dict) and 'component_id' in item:
                component_id = item['component_id']
                
                # Aggregate by component
                self._aggregated_metrics[component_id].append({
                    'timestamp': current_time,
                    'health_score': item.get('health_score', 1.0),
                    'cpu_usage': item.get('cpu_usage'),
                    'memory_usage': item.get('memory_usage'),
                    'error_count': item.get('error_count', 0)
                })
    
    async def _export_json(self, data: List[Dict[str, Any]]) -> bool:
        """Export data in JSON format."""
        try:
            timestamp = int(time.time())
            filename = f"metrics_{timestamp}.json"
            
            if self.config.enable_compression:
                filename += ".gz"
            
            filepath = self.output_directory / filename
            
            export_data = {
                'timestamp': timestamp,
                'export_config': asdict(self.config),
                'metrics_count': len(data),
                'metrics': data
            }
            
            if self.config.enable_compression:
                with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.debug(f"Exported {len(data)} metrics to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
            return False
    
    async def _export_csv(self, data: List[Dict[str, Any]]) -> bool:
        """Export data in CSV format."""
        try:
            import csv
            
            timestamp = int(time.time())
            filename = f"metrics_{timestamp}.csv"
            
            if self.config.enable_compression:
                filename += ".gz"
            
            filepath = self.output_directory / filename
            
            if not data:
                return True
            
            # Get all possible fieldnames
            fieldnames = set()
            for item in data:
                if isinstance(item, dict):
                    fieldnames.update(item.keys())
            
            fieldnames = sorted(fieldnames)
            
            if self.config.enable_compression:
                with gzip.open(filepath, 'wt', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in data:
                        if isinstance(item, dict):
                            writer.writerow(item)
            else:
                with open(filepath, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for item in data:
                        if isinstance(item, dict):
                            writer.writerow(item)
            
            logger.debug(f"Exported {len(data)} metrics to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
            return False
    
    async def _export_prometheus(self, data: List[Dict[str, Any]]) -> bool:
        """Export data in Prometheus format."""
        try:
            timestamp = int(time.time())
            filename = f"metrics_{timestamp}.prom"
            filepath = self.output_directory / filename
            
            lines = []
            lines.append(f"# HELP vllm_monitor_metrics vLLM monitoring metrics")
            lines.append(f"# TYPE vllm_monitor_metrics gauge")
            
            for item in data:
                if not isinstance(item, dict):
                    continue
                
                component_id = item.get('component_id', 'unknown')
                component_type = item.get('component_type', 'unknown')
                timestamp_ms = int(item.get('timestamp', time.time()) * 1000)
                
                # Export numeric metrics
                for key, value in item.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        metric_name = f"vllm_{key}"
                        labels = f'component_id="{component_id}",component_type="{component_type}"'
                        lines.append(f"{metric_name}{{{labels}}} {value} {timestamp_ms}")
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            logger.debug(f"Exported {len(data)} metrics to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Prometheus export failed: {e}")
            return False
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics."""
        with self._lock:
            success_rate = (
                (self._export_count - self._failed_exports) / max(self._export_count, 1) * 100.0
            )
            
            return {
                'total_exports': self._export_count,
                'failed_exports': self._failed_exports,
                'success_rate_percent': success_rate,
                'last_export_time': self._last_export_time,
                'buffer_size': len(self._metrics_buffer),
                'output_directory': str(self.output_directory),
                'export_format': self.config.export_format
            }
    
    def get_aggregated_metrics(self, component_id: Optional[str] = None) -> Dict[str, Any]:
        """Get aggregated metrics summary."""
        if component_id:
            return {
                component_id: self._aggregated_metrics.get(component_id, [])
            }
        else:
            return dict(self._aggregated_metrics)


class LogExporter(BaseExporter):
    """Exporter for log data with structured logging support."""
    
    def __init__(self, 
                 config: ExportConfig,
                 log_directory: str = "./vllm_logs"):
        super().__init__(config)
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Setup structured logger
        self._setup_structured_logger()
        
        # Log buffer
        self._log_buffer: deque = deque(maxlen=10000)
        
    def _setup_structured_logger(self) -> None:
        """Setup structured logging for export."""
        self.structured_logger = logging.getLogger("vllm_monitor.structured")
        
        # Create handler for structured logs
        log_file = self.log_directory / "vllm_monitor.log"
        handler = logging.FileHandler(log_file)
        
        # JSON formatter for structured logs
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
        handler.setFormatter(formatter)
        
        self.structured_logger.addHandler(handler)
        self.structured_logger.setLevel(logging.INFO)
    
    async def export_data(self, data: List[Dict[str, Any]]) -> bool:
        """Export log data to structured logs."""
        if not data:
            return True
        
        try:
            # Add to buffer
            self._log_buffer.extend(data)
            
            # Write structured logs
            for item in data:
                if isinstance(item, dict):
                    log_level = item.get('level', 'INFO')
                    message = json.dumps(item, default=str)
                    
                    if log_level == 'DEBUG':
                        self.structured_logger.debug(message)
                    elif log_level == 'INFO':
                        self.structured_logger.info(message)
                    elif log_level == 'WARNING':
                        self.structured_logger.warning(message)
                    elif log_level == 'ERROR':
                        self.structured_logger.error(message)
                    elif log_level == 'CRITICAL':
                        self.structured_logger.critical(message)
                    else:
                        self.structured_logger.info(message)
            
            # Export to file if needed
            if self.config.export_format == "json":
                await self._export_log_file(data)
            
            self._record_export(True)
            return True
            
        except Exception as e:
            logger.error(f"Log export failed: {e}")
            self._record_export(False)
            return False
    
    async def _export_log_file(self, data: List[Dict[str, Any]]) -> None:
        """Export logs to timestamped file."""
        timestamp = int(time.time())
        filename = f"logs_{timestamp}.json"
        
        if self.config.enable_compression:
            filename += ".gz"
        
        filepath = self.log_directory / filename
        
        export_data = {
            'timestamp': timestamp,
            'log_count': len(data),
            'logs': data
        }
        
        if self.config.enable_compression:
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, default=str)
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get log export statistics."""
        with self._lock:
            success_rate = (
                (self._export_count - self._failed_exports) / max(self._export_count, 1) * 100.0
            )
            
            return {
                'total_exports': self._export_count,
                'failed_exports': self._failed_exports,
                'success_rate_percent': success_rate,
                'last_export_time': self._last_export_time,
                'buffer_size': len(self._log_buffer),
                'log_directory': str(self.log_directory)
            }


class AlertManager:
    """Manager for generating and sending alerts."""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self._alert_history: deque = deque(maxlen=1000)
        self._alert_cooldowns: Dict[str, float] = {}
        self._alerts_sent = 0
        self._failed_alerts = 0
        self._lock = threading.RLock()
        
        # Initialize notification backends
        self._notification_backends: List[Callable] = []
        self._setup_notification_backends()
    
    def _setup_notification_backends(self) -> None:
        """Setup notification backends based on configuration."""
        if self.config.webhook_url:
            self._notification_backends.append(self._send_webhook_alert)
        
        if self.config.email_recipients:
            self._notification_backends.append(self._send_email_alert)
        
        if self.config.slack_webhook:
            self._notification_backends.append(self._send_slack_alert)
    
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send an alert through configured channels."""
        if not self.config.enabled:
            return False
        
        # Check alert level threshold
        alert_level = AlertLevel(alert.get('level', AlertLevel.INFO.value))
        if alert_level.value < self.config.min_alert_level.value:
            return False
        
        # Generate alert key for cooldown tracking
        alert_key = f"{alert.get('component', 'unknown')}_{alert.get('type', 'unknown')}"
        
        with self._lock:
            current_time = time.time()
            
            # Check cooldown
            if alert_key in self._alert_cooldowns:
                if current_time - self._alert_cooldowns[alert_key] < self.config.alert_cooldown_seconds:
                    logger.debug(f"Alert {alert_key} skipped due to cooldown")
                    return False
            
            # Check rate limit
            recent_alerts = [
                a for a in self._alert_history
                if current_time - a['timestamp'] < 3600  # Last hour
            ]
            if len(recent_alerts) >= self.config.max_alerts_per_hour:
                logger.warning("Alert rate limit exceeded")
                return False
            
            # Update cooldown
            self._alert_cooldowns[alert_key] = current_time
        
        # Send alert through all backends
        success = True
        for backend in self._notification_backends:
            try:
                backend_success = await backend(alert)
                success = success and backend_success
            except Exception as e:
                logger.error(f"Alert backend failed: {e}")
                success = False
        
        # Record alert
        with self._lock:
            self._alert_history.append({
                'timestamp': current_time,
                'alert': alert,
                'success': success
            })
            
            if success:
                self._alerts_sent += 1
            else:
                self._failed_alerts += 1
        
        return success
    
    async def _send_webhook_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert via webhook."""
        try:
            import aiohttp
            
            payload = {
                'timestamp': time.time(),
                'source': 'vllm_monitor',
                'alert': alert
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    success = response.status < 400
                    if not success:
                        logger.error(f"Webhook alert failed: {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Webhook alert failed: {e}")
            return False
    
    async def _send_email_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert via email."""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # This is a simplified email implementation
            # In production, you'd want proper SMTP configuration
            
            subject = f"vLLM Monitor Alert: {alert.get('type', 'Unknown')}"
            
            body = f"""
            Alert Details:
            - Component: {alert.get('component', 'Unknown')}
            - Level: {AlertLevel(alert.get('level', AlertLevel.INFO.value)).name}
            - Message: {alert.get('message', 'No message')}
            - Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.get('timestamp', time.time())))}
            
            Data: {json.dumps(alert.get('data', {}), indent=2)}
            """
            
            # Note: This is a placeholder - actual email sending would require
            # SMTP server configuration
            logger.info(f"Email alert would be sent to {self.config.email_recipients}: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Email alert failed: {e}")
            return False
    
    async def _send_slack_alert(self, alert: Dict[str, Any]) -> bool:
        """Send alert via Slack webhook."""
        try:
            import aiohttp
            
            level_colors = {
                AlertLevel.INFO: "#36a64f",      # Green
                AlertLevel.WARNING: "#ff9800",   # Orange
                AlertLevel.ERROR: "#f44336",     # Red
                AlertLevel.CRITICAL: "#9c27b0",  # Purple
                AlertLevel.FATAL: "#000000"      # Black
            }
            
            alert_level = AlertLevel(alert.get('level', AlertLevel.INFO.value))
            color = level_colors.get(alert_level, "#808080")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"vLLM Monitor Alert - {alert_level.name}",
                        "text": alert.get('message', 'No message'),
                        "fields": [
                            {
                                "title": "Component",
                                "value": alert.get('component', 'Unknown'),
                                "short": True
                            },
                            {
                                "title": "Timestamp",
                                "value": time.strftime(
                                    '%Y-%m-%d %H:%M:%S',
                                    time.localtime(alert.get('timestamp', time.time()))
                                ),
                                "short": True
                            }
                        ],
                        "footer": "vLLM Monitor",
                        "ts": int(alert.get('timestamp', time.time()))
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    success = response.status < 400
                    if not success:
                        logger.error(f"Slack alert failed: {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Slack alert failed: {e}")
            return False
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alert statistics."""
        with self._lock:
            success_rate = (
                self._alerts_sent / max(self._alerts_sent + self._failed_alerts, 1) * 100.0
            )
            
            # Recent activity
            current_time = time.time()
            recent_alerts = [
                a for a in self._alert_history
                if current_time - a['timestamp'] < 3600  # Last hour
            ]
            
            return {
                'enabled': self.config.enabled,
                'total_alerts_sent': self._alerts_sent,
                'failed_alerts': self._failed_alerts,
                'success_rate_percent': success_rate,
                'recent_alerts_count': len(recent_alerts),
                'active_cooldowns': len(self._alert_cooldowns),
                'notification_backends': len(self._notification_backends),
                'recent_alerts': [a['alert'] for a in recent_alerts[-5:]]  # Last 5
            }


class ExportManager:
    """Manager for coordinating all export operations."""
    
    def __init__(self, 
                 monitor_config: Optional[MonitorConfig] = None,
                 export_config: Optional[ExportConfig] = None,
                 alert_config: Optional[AlertConfig] = None):
        self.monitor_config = monitor_config or MonitorConfig()
        self.export_config = export_config or ExportConfig()
        self.alert_config = alert_config or AlertConfig()
        
        # Initialize exporters
        self.metrics_exporter = MetricsExporter(self.export_config)
        self.log_exporter = LogExporter(self.export_config)
        self.alert_manager = AlertManager(self.alert_config)
        
        # Export queues
        self._metrics_queue: deque = deque(maxlen=10000)
        self._log_queue: deque = deque(maxlen=10000)
        self._alert_queue: deque = deque(maxlen=1000)
        
        # State tracking
        self._is_running = False
        self._export_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
    async def start(self) -> None:
        """Start the export manager."""
        if self._is_running:
            return
        
        with self._lock:
            self._is_running = True
        
        # Start export loop
        self._export_task = asyncio.create_task(self._export_loop())
        
        logger.info("Export manager started")
    
    async def stop(self) -> None:
        """Stop the export manager."""
        if not self._is_running:
            return
        
        with self._lock:
            self._is_running = False
        
        # Cancel export task
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
        
        # Final export of remaining data
        await self._flush_exports()
        
        logger.info("Export manager stopped")
    
    def queue_metrics(self, metrics: List[ComponentState]) -> None:
        """Queue metrics for export."""
        if not self.monitor_config.enable_metrics_export:
            return
        
        # Convert ComponentState objects to dictionaries
        metric_dicts = [metric.to_dict() for metric in metrics]
        
        with self._lock:
            self._metrics_queue.extend(metric_dicts)
    
    def queue_logs(self, logs: List[Dict[str, Any]]) -> None:
        """Queue logs for export."""
        if not self.monitor_config.enable_logging:
            return
        
        with self._lock:
            self._log_queue.extend(logs)
    
    async def queue_alert(self, alert: Dict[str, Any]) -> None:
        """Queue alert for sending."""
        if not self.monitor_config.enable_alerts:
            return
        
        with self._lock:
            self._alert_queue.append(alert)
        
        # Send alerts immediately for high priority
        alert_level = AlertLevel(alert.get('level', AlertLevel.INFO.value))
        if alert_level.value >= AlertLevel.ERROR.value:
            await self.alert_manager.send_alert(alert)
    
    async def _export_loop(self) -> None:
        """Main export loop."""
        while self._is_running:
            try:
                await self._process_exports()
                await asyncio.sleep(self.export_config.export_interval_seconds)
                
            except Exception as e:
                logger.error(f"Export loop error: {e}")
                await asyncio.sleep(5.0)  # Back off on error
    
    async def _process_exports(self) -> None:
        """Process queued exports."""
        # Export metrics
        if self._metrics_queue and self.metrics_exporter.should_export():
            with self._lock:
                metrics_batch = list(self._metrics_queue)
                self._metrics_queue.clear()
            
            if metrics_batch:
                await self.metrics_exporter.export_data(metrics_batch)
        
        # Export logs
        if self._log_queue and self.log_exporter.should_export():
            with self._lock:
                logs_batch = list(self._log_queue)
                self._log_queue.clear()
            
            if logs_batch:
                await self.log_exporter.export_data(logs_batch)
        
        # Send queued alerts
        with self._lock:
            alerts_to_send = list(self._alert_queue)
            self._alert_queue.clear()
        
        for alert in alerts_to_send:
            await self.alert_manager.send_alert(alert)
    
    async def _flush_exports(self) -> None:
        """Flush remaining export data."""
        # Force export of remaining data
        with self._lock:
            if self._metrics_queue:
                metrics_batch = list(self._metrics_queue)
                self._metrics_queue.clear()
                await self.metrics_exporter.export_data(metrics_batch)
            
            if self._log_queue:
                logs_batch = list(self._log_queue)
                self._log_queue.clear()
                await self.log_exporter.export_data(logs_batch)
            
            # Send remaining alerts
            alerts_to_send = list(self._alert_queue)
            self._alert_queue.clear()
            
            for alert in alerts_to_send:
                await self.alert_manager.send_alert(alert)
    
    def get_export_status(self) -> Dict[str, Any]:
        """Get comprehensive export status."""
        with self._lock:
            return {
                'is_running': self._is_running,
                'queue_sizes': {
                    'metrics': len(self._metrics_queue),
                    'logs': len(self._log_queue),
                    'alerts': len(self._alert_queue)
                },
                'metrics_exporter': self.metrics_exporter.get_export_stats(),
                'log_exporter': self.log_exporter.get_export_stats(),
                'alert_manager': self.alert_manager.get_alert_stats(),
                'export_config': asdict(self.export_config),
                'alert_config': asdict(self.alert_config)
            }