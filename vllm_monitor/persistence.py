"""
Persistence layer for vLLM monitoring system.

Provides persistent storage for:
- Metrics collection history
- Predictive strategies and patterns
- Dynamically added modules and plugins
- Guardrails and policies
- Mitigation strategies and outcomes
- Happy path recordings
- Learning data and models
"""

import os
import json
import pickle
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from contextlib import contextmanager
import zlib
import base64
from dataclasses import dataclass, asdict, field
from enum import Enum

import numpy as np
from collections import deque, defaultdict

from .lifecycle_tracker import LifecycleState, StateTransition, StateCheckpoint, GuardrailPolicy
from .continuous_learning import MitigationAttempt, MitigationOutcome
from .predictive_failure_detection import FailurePrediction, FailurePattern


class StorageBackend(Enum):
    """Supported storage backends."""
    SQLITE = "sqlite"
    JSON = "json"
    HYBRID = "hybrid"  # SQLite for structured data, JSON for configs


@dataclass
class PersistenceConfig:
    """Configuration for persistence system."""
    storage_dir: Path = field(default_factory=lambda: Path.home() / ".vllm_monitor")
    backend: StorageBackend = StorageBackend.HYBRID
    
    # Retention policies
    metrics_retention_days: int = 30
    checkpoint_retention_days: int = 90
    prediction_retention_days: int = 60
    
    # Performance settings
    batch_size: int = 1000
    flush_interval_seconds: int = 60
    compression_enabled: bool = True
    
    # File paths
    db_file: str = "monitor.db"
    plugins_file: str = "plugins.json"
    guardrails_file: str = "guardrails.json"
    strategies_file: str = "strategies.json"
    happy_paths_file: str = "happy_paths.json"
    models_dir: str = "models"


class PersistenceManager:
    """
    Manages persistent storage for all monitoring components.
    
    Features:
    - Automatic schema management
    - Batch writing for performance
    - Data compression
    - Retention policies
    - Export/import capabilities
    """
    
    def __init__(self, config: Optional[PersistenceConfig] = None):
        self.config = config or PersistenceConfig()
        self._ensure_storage_dir()
        
        # Thread safety
        self._lock = threading.RLock()
        self._write_buffer: Dict[str, List[Any]] = defaultdict(list)
        self._last_flush = time.time()
        
        # Initialize storage
        self._init_storage()
        
        # Start background flush thread
        self._flush_thread = threading.Thread(target=self._flush_worker, daemon=True)
        self._flush_thread.start()
    
    def _ensure_storage_dir(self):
        """Ensure storage directory exists."""
        self.config.storage_dir.mkdir(parents=True, exist_ok=True)
        (self.config.storage_dir / self.config.models_dir).mkdir(exist_ok=True)
    
    def _init_storage(self):
        """Initialize storage backends."""
        if self.config.backend in [StorageBackend.SQLITE, StorageBackend.HYBRID]:
            self._init_sqlite()
    
    def _init_sqlite(self):
        """Initialize SQLite database with schema."""
        db_path = self.config.storage_dir / self.config.db_file
        
        with self._get_db() as conn:
            # Metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metric_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    value REAL,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # State checkpoints table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    state TEXT NOT NULL,
                    previous_state TEXT,
                    transition_type TEXT,
                    arguments TEXT,
                    environment TEXT,
                    hardware_state TEXT,
                    metrics TEXT,
                    guardrail_violations TEXT,
                    interventions TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    checkpoint_id INTEGER,
                    failure_type TEXT NOT NULL,
                    probability REAL NOT NULL,
                    confidence REAL NOT NULL,
                    time_to_failure REAL,
                    contributing_factors TEXT,
                    preventive_actions TEXT,
                    pattern_matches TEXT,
                    actual_outcome TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(id)
                )
            """)
            
            # Mitigation attempts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS mitigation_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    attempt_id TEXT UNIQUE NOT NULL,
                    timestamp REAL NOT NULL,
                    initial_state TEXT NOT NULL,
                    initial_metrics TEXT,
                    error_context TEXT,
                    strategy_name TEXT NOT NULL,
                    interventions_executed TEXT,
                    execution_time REAL,
                    outcome TEXT NOT NULL,
                    final_state TEXT,
                    final_metrics TEXT,
                    side_effects TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Happy paths table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS happy_paths (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    start_state TEXT NOT NULL,
                    end_state TEXT NOT NULL,
                    checkpoints TEXT NOT NULL,
                    metrics_profile TEXT,
                    guardrails TEXT,
                    duration REAL,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    last_success TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Learning data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    learning_type TEXT NOT NULL,
                    state_context TEXT,
                    action_taken TEXT,
                    reward REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON metrics(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_type ON metrics(metric_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_timestamp ON checkpoints(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_checkpoints_state ON checkpoints(state)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_predictions_type ON predictions(failure_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_timestamp ON mitigation_attempts(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_strategy ON mitigation_attempts(strategy_name)")
            
            conn.commit()
    
    @contextmanager
    def _get_db(self):
        """Get database connection context manager."""
        db_path = self.config.storage_dir / self.config.db_file
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def _compress(self, data: Any) -> str:
        """Compress and encode data."""
        if not self.config.compression_enabled:
            return json.dumps(data)
        
        json_str = json.dumps(data)
        compressed = zlib.compress(json_str.encode('utf-8'))
        return base64.b64encode(compressed).decode('ascii')
    
    def _decompress(self, data: str) -> Any:
        """Decompress and decode data."""
        if not self.config.compression_enabled:
            return json.loads(data)
        
        try:
            compressed = base64.b64decode(data.encode('ascii'))
            json_str = zlib.decompress(compressed).decode('utf-8')
            return json.loads(json_str)
        except:
            # Fallback for uncompressed data
            return json.loads(data)
    
    # Metrics Persistence
    
    def save_metric(self, metric_type: str, metric_name: str, value: float,
                   tags: Optional[Dict[str, str]] = None,
                   metadata: Optional[Dict[str, Any]] = None):
        """Save a metric to persistent storage."""
        with self._lock:
            self._write_buffer['metrics'].append({
                'timestamp': time.time(),
                'metric_type': metric_type,
                'metric_name': metric_name,
                'value': value,
                'tags': json.dumps(tags or {}),
                'metadata': self._compress(metadata or {})
            })
            
            if len(self._write_buffer['metrics']) >= self.config.batch_size:
                self._flush_metrics()
    
    def get_metrics(self, metric_type: Optional[str] = None,
                   metric_name: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve metrics from storage."""
        with self._get_db() as conn:
            query = "SELECT * FROM metrics WHERE 1=1"
            params = []
            
            if metric_type:
                query += " AND metric_type = ?"
                params.append(metric_type)
            
            if metric_name:
                query += " AND metric_name = ?"
                params.append(metric_name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor:
                results.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'metric_type': row['metric_type'],
                    'metric_name': row['metric_name'],
                    'value': row['value'],
                    'tags': json.loads(row['tags']),
                    'metadata': self._decompress(row['metadata'])
                })
            
            return results
    
    # Checkpoint Persistence
    
    def save_checkpoint(self, checkpoint: StateCheckpoint) -> int:
        """Save a state checkpoint."""
        with self._lock:
            self._write_buffer['checkpoints'].append({
                'timestamp': checkpoint.timestamp,
                'state': checkpoint.state.name,
                'previous_state': checkpoint.previous_state.name if checkpoint.previous_state else None,
                'transition_type': checkpoint.transition_type.name if checkpoint.transition_type else None,
                'arguments': self._compress(checkpoint.arguments),
                'environment': self._compress(checkpoint.environment),
                'hardware_state': self._compress(checkpoint.hardware_state),
                'metrics': self._compress(checkpoint.metrics),
                'guardrail_violations': self._compress(
                    checkpoint.guardrails_enforced
                ),
                'interventions': self._compress(
                    checkpoint.interventions_triggered
                )
            })
            
            if len(self._write_buffer['checkpoints']) >= self.config.batch_size:
                self._flush_checkpoints()
            
            # Return estimated ID (will be accurate after flush)
            return -1
    
    def get_checkpoints(self, state: Optional[LifecycleState] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       limit: int = 100) -> List[StateCheckpoint]:
        """Retrieve checkpoints from storage."""
        with self._get_db() as conn:
            query = "SELECT * FROM checkpoints WHERE 1=1"
            params = []
            
            if state:
                query += " AND state = ?"
                params.append(state.name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor:
                # Reconstruct checkpoint
                checkpoint = StateCheckpoint(
                    timestamp=row['timestamp'],
                    state=LifecycleState[row['state']],
                    previous_state=LifecycleState[row['previous_state']] if row['previous_state'] else None,
                    transition_type=StateTransition[row['transition_type']] if row['transition_type'] else None,
                    arguments=self._decompress(row['arguments']),
                    environment=self._decompress(row['environment']),
                    hardware_state=self._decompress(row['hardware_state']),
                    metrics=self._decompress(row['metrics']),
                    guardrail_violations=[],  # TODO: Reconstruct from dict
                    interventions=[]  # TODO: Reconstruct from dict
                )
                results.append(checkpoint)
            
            return results
    
    # Prediction Persistence
    
    def save_prediction(self, prediction: FailurePrediction,
                       checkpoint_id: Optional[int] = None,
                       actual_outcome: Optional[str] = None):
        """Save a failure prediction."""
        with self._lock:
            self._write_buffer['predictions'].append({
                'timestamp': time.time(),
                'checkpoint_id': checkpoint_id,
                'failure_type': prediction.failure_type,
                'probability': prediction.probability,
                'confidence': prediction.confidence.value if hasattr(prediction.confidence, 'value') else prediction.confidence,
                'time_to_failure': prediction.time_to_failure,
                'contributing_factors': json.dumps(prediction.contributing_factors),
                'preventive_actions': json.dumps([m.interventions for m in prediction.recommended_mitigations] if hasattr(prediction, 'recommended_mitigations') else []),
                'pattern_matches': json.dumps([]),
                'actual_outcome': actual_outcome
            })
            
            if len(self._write_buffer['predictions']) >= self.config.batch_size:
                self._flush_predictions()
    
    def update_prediction_outcome(self, prediction_id: int, actual_outcome: str):
        """Update a prediction with its actual outcome."""
        with self._get_db() as conn:
            conn.execute(
                "UPDATE predictions SET actual_outcome = ? WHERE id = ?",
                (actual_outcome, prediction_id)
            )
            conn.commit()
    
    def get_predictions(self, failure_type: Optional[str] = None,
                       start_time: Optional[float] = None,
                       end_time: Optional[float] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve predictions from storage."""
        with self._get_db() as conn:
            query = "SELECT * FROM predictions WHERE 1=1"
            params = []
            
            if failure_type:
                query += " AND failure_type = ?"
                params.append(failure_type)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor:
                results.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'checkpoint_id': row['checkpoint_id'],
                    'failure_type': row['failure_type'],
                    'probability': row['probability'],
                    'confidence': row['confidence'],
                    'time_to_failure': row['time_to_failure'],
                    'contributing_factors': json.loads(row['contributing_factors']),
                    'preventive_actions': json.loads(row['preventive_actions']),
                    'pattern_matches': json.loads(row['pattern_matches']),
                    'actual_outcome': row['actual_outcome']
                })
            
            return results
    
    # Mitigation Persistence
    
    def save_mitigation_attempt(self, attempt: MitigationAttempt):
        """Save a mitigation attempt."""
        with self._lock:
            self._write_buffer['attempts'].append({
                'attempt_id': attempt.attempt_id,
                'timestamp': attempt.timestamp,
                'initial_state': attempt.initial_state.name,
                'initial_metrics': self._compress(attempt.initial_metrics),
                'error_context': json.dumps(attempt.error_context),
                'strategy_name': attempt.strategy_name,
                'interventions_executed': json.dumps(attempt.interventions_executed),
                'execution_time': attempt.execution_time,
                'outcome': attempt.outcome.name,
                'final_state': attempt.final_state.name if attempt.final_state else None,
                'final_metrics': self._compress(attempt.final_metrics),
                'side_effects': json.dumps(attempt.side_effects)
            })
            
            if len(self._write_buffer['attempts']) >= self.config.batch_size:
                self._flush_attempts()
    
    def get_mitigation_history(self, strategy_name: Optional[str] = None,
                              outcome: Optional[MitigationOutcome] = None,
                              limit: int = 100) -> List[MitigationAttempt]:
        """Retrieve mitigation attempt history."""
        with self._get_db() as conn:
            query = "SELECT * FROM mitigation_attempts WHERE 1=1"
            params = []
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            if outcome:
                query += " AND outcome = ?"
                params.append(outcome.name)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor:
                attempt = MitigationAttempt(
                    attempt_id=row['attempt_id'],
                    timestamp=row['timestamp'],
                    initial_state=LifecycleState[row['initial_state']],
                    initial_metrics=self._decompress(row['initial_metrics']),
                    error_context=json.loads(row['error_context']),
                    strategy_name=row['strategy_name'],
                    interventions_executed=json.loads(row['interventions_executed']),
                    execution_time=row['execution_time'],
                    outcome=MitigationOutcome[row['outcome']],
                    final_state=LifecycleState[row['final_state']] if row['final_state'] else None,
                    final_metrics=self._decompress(row['final_metrics']),
                    side_effects=json.loads(row['side_effects'])
                )
                results.append(attempt)
            
            return results
    
    # Plugin Persistence
    
    def save_plugins(self, plugins: Dict[str, Any]):
        """Save dynamically loaded plugins."""
        plugins_path = self.config.storage_dir / self.config.plugins_file
        
        with self._lock:
            # Convert plugin objects to serializable format
            serializable_plugins = {}
            for name, plugin_info in plugins.items():
                serializable_plugins[name] = {
                    'type': plugin_info.get('type', 'unknown'),
                    'source_code': plugin_info.get('source_code', ''),
                    'config': plugin_info.get('config', {}),
                    'enabled': plugin_info.get('enabled', True),
                    'dependencies': plugin_info.get('dependencies', []),
                    'created_at': plugin_info.get('created_at', time.time()),
                    'updated_at': time.time()
                }
            
            with open(plugins_path, 'w') as f:
                json.dump(serializable_plugins, f, indent=2)
    
    def load_plugins(self) -> Dict[str, Any]:
        """Load saved plugins."""
        plugins_path = self.config.storage_dir / self.config.plugins_file
        
        if not plugins_path.exists():
            return {}
        
        with open(plugins_path, 'r') as f:
            return json.load(f)
    
    # Guardrail Persistence
    
    def save_guardrails(self, guardrails: Union[Dict[str, GuardrailPolicy], List[GuardrailPolicy]]):
        """Save guardrail policies."""
        guardrails_path = self.config.storage_dir / self.config.guardrails_file
        
        with self._lock:
            serializable_guardrails = []
            # Handle both dict and list inputs
            if isinstance(guardrails, dict):
                guardrails_list = list(guardrails.values())
            else:
                guardrails_list = guardrails
                
            for policy in guardrails_list:
                # Note: We can't serialize the condition function, so we store metadata about it
                serializable_guardrails.append({
                    'name': policy.name,
                    'description': policy.description,
                    'condition': '<function>',  # Can't serialize function
                    'intervention': policy.intervention,
                    'severity': policy.severity,
                    'enabled': policy.enabled,
                    'metadata': policy.metadata
                })
            
            with open(guardrails_path, 'w') as f:
                json.dump(serializable_guardrails, f, indent=2)
    
    def load_guardrails(self) -> List[GuardrailPolicy]:
        """Load saved guardrail policies."""
        guardrails_path = self.config.storage_dir / self.config.guardrails_file
        
        if not guardrails_path.exists():
            return []
        
        with open(guardrails_path, 'r') as f:
            data = json.load(f)
        
        guardrails = []
        for item in data:
            policy = GuardrailPolicy(
                name=item['name'],
                description=item['description'],
                condition=item['condition'],
                action=item['action'],
                severity=item['severity'],
                enabled=item['enabled'],
                metadata=item['metadata']
            )
            guardrails.append(policy)
        
        return guardrails
    
    # Strategy Persistence
    
    def save_strategies(self, strategies: Dict[str, Any]):
        """Save mitigation strategies with their metadata."""
        strategies_path = self.config.storage_dir / self.config.strategies_file
        
        with self._lock:
            serializable_strategies = {}
            for name, strategy_info in strategies.items():
                serializable_strategies[name] = {
                    'description': strategy_info.get('description', ''),
                    'source_code': strategy_info.get('source_code', ''),
                    'success_rate': strategy_info.get('success_rate', 0.0),
                    'avg_execution_time': strategy_info.get('avg_execution_time', 0.0),
                    'usage_count': strategy_info.get('usage_count', 0),
                    'last_used': strategy_info.get('last_used', None),
                    'compatible_states': strategy_info.get('compatible_states', []),
                    'required_resources': strategy_info.get('required_resources', []),
                    'metadata': strategy_info.get('metadata', {})
                }
            
            with open(strategies_path, 'w') as f:
                json.dump(serializable_strategies, f, indent=2)
    
    def load_strategies(self) -> Dict[str, Any]:
        """Load saved mitigation strategies."""
        strategies_path = self.config.storage_dir / self.config.strategies_file
        
        if not strategies_path.exists():
            return {}
        
        with open(strategies_path, 'r') as f:
            return json.load(f)
    
    # Happy Path Persistence
    
    def save_happy_path(self, path_id: str, name: str, checkpoints: List[StateCheckpoint],
                       metrics_profile: Dict[str, Any], guardrails: List[str]):
        """Save a happy path recording."""
        with self._get_db() as conn:
            # Serialize checkpoints
            checkpoint_data = []
            for cp in checkpoints:
                checkpoint_data.append({
                    'timestamp': cp.timestamp,
                    'state': cp.state.name,
                    'metrics': cp.metrics
                })
            
            duration = checkpoints[-1].timestamp - checkpoints[0].timestamp if checkpoints else 0
            
            conn.execute("""
                INSERT OR REPLACE INTO happy_paths 
                (path_id, name, description, start_state, end_state, checkpoints,
                 metrics_profile, guardrails, duration, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                path_id,
                name,
                f"Happy path from {checkpoints[0].state.name} to {checkpoints[-1].state.name}",
                checkpoints[0].state.name,
                checkpoints[-1].state.name,
                self._compress(checkpoint_data),
                self._compress(metrics_profile),
                json.dumps(guardrails),
                duration
            ))
            conn.commit()
    
    def get_happy_paths(self, start_state: Optional[LifecycleState] = None,
                       end_state: Optional[LifecycleState] = None) -> List[Dict[str, Any]]:
        """Retrieve happy path recordings."""
        with self._get_db() as conn:
            query = "SELECT * FROM happy_paths WHERE 1=1"
            params = []
            
            if start_state:
                query += " AND start_state = ?"
                params.append(start_state.name)
            
            if end_state:
                query += " AND end_state = ?"
                params.append(end_state.name)
            
            query += " ORDER BY success_count DESC, updated_at DESC"
            
            cursor = conn.execute(query, params)
            
            results = []
            for row in cursor:
                results.append({
                    'path_id': row['path_id'],
                    'name': row['name'],
                    'start_state': row['start_state'],
                    'end_state': row['end_state'],
                    'checkpoints': self._decompress(row['checkpoints']),
                    'metrics_profile': self._decompress(row['metrics_profile']),
                    'guardrails': json.loads(row['guardrails']),
                    'duration': row['duration'],
                    'success_count': row['success_count'],
                    'failure_count': row['failure_count']
                })
            
            return results
    
    def update_happy_path_stats(self, path_id: str, success: bool):
        """Update happy path success/failure statistics."""
        with self._get_db() as conn:
            if success:
                conn.execute("""
                    UPDATE happy_paths 
                    SET success_count = success_count + 1,
                        last_success = CURRENT_TIMESTAMP
                    WHERE path_id = ?
                """, (path_id,))
            else:
                conn.execute("""
                    UPDATE happy_paths 
                    SET failure_count = failure_count + 1
                    WHERE path_id = ?
                """, (path_id,))
            conn.commit()
    
    # Learning Data Persistence
    
    def save_learning_data(self, learning_type: str, state_context: Dict[str, Any],
                          action_taken: str, reward: float, metadata: Optional[Dict[str, Any]] = None):
        """Save learning/training data."""
        with self._lock:
            self._write_buffer['learning'].append({
                'timestamp': time.time(),
                'learning_type': learning_type,
                'state_context': self._compress(state_context),
                'action_taken': action_taken,
                'reward': reward,
                'metadata': self._compress(metadata or {})
            })
            
            if len(self._write_buffer['learning']) >= self.config.batch_size:
                self._flush_learning()
    
    # Model Persistence
    
    def save_model(self, model_name: str, model_data: Any, metadata: Optional[Dict[str, Any]] = None):
        """Save a trained model (neural network, decision tree, etc.)."""
        model_dir = self.config.storage_dir / self.config.models_dir
        model_path = model_dir / f"{model_name}.pkl"
        meta_path = model_dir / f"{model_name}_meta.json"
        
        with self._lock:
            # Save model binary
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            # Save metadata
            metadata = metadata or {}
            metadata['saved_at'] = time.time()
            metadata['model_name'] = model_name
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict[str, Any]]:
        """Load a saved model and its metadata."""
        model_dir = self.config.storage_dir / self.config.models_dir
        model_path = model_dir / f"{model_name}.pkl"
        meta_path = model_dir / f"{model_name}_meta.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found")
        
        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load metadata
        metadata = {}
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        return model_data, metadata
    
    # Batch flushing methods
    
    def _flush_metrics(self):
        """Flush metrics buffer to database."""
        if not self._write_buffer['metrics']:
            return
        
        with self._get_db() as conn:
            conn.executemany("""
                INSERT INTO metrics 
                (timestamp, metric_type, metric_name, value, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (m['timestamp'], m['metric_type'], m['metric_name'],
                 m['value'], m['tags'], m['metadata'])
                for m in self._write_buffer['metrics']
            ])
            conn.commit()
        
        self._write_buffer['metrics'].clear()
    
    def _flush_checkpoints(self):
        """Flush checkpoints buffer to database."""
        if not self._write_buffer['checkpoints']:
            return
        
        with self._get_db() as conn:
            conn.executemany("""
                INSERT INTO checkpoints 
                (timestamp, state, previous_state, transition_type, arguments,
                 environment, hardware_state, metrics, guardrail_violations, interventions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (c['timestamp'], c['state'], c['previous_state'], c['transition_type'],
                 c['arguments'], c['environment'], c['hardware_state'], c['metrics'],
                 c['guardrail_violations'], c['interventions'])
                for c in self._write_buffer['checkpoints']
            ])
            conn.commit()
        
        self._write_buffer['checkpoints'].clear()
    
    def _flush_predictions(self):
        """Flush predictions buffer to database."""
        if not self._write_buffer['predictions']:
            return
        
        with self._get_db() as conn:
            conn.executemany("""
                INSERT INTO predictions 
                (timestamp, checkpoint_id, failure_type, probability, confidence,
                 time_to_failure, contributing_factors, preventive_actions,
                 pattern_matches, actual_outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (p['timestamp'], p['checkpoint_id'], p['failure_type'],
                 p['probability'], p['confidence'], p['time_to_failure'],
                 p['contributing_factors'], p['preventive_actions'],
                 p['pattern_matches'], p['actual_outcome'])
                for p in self._write_buffer['predictions']
            ])
            conn.commit()
        
        self._write_buffer['predictions'].clear()
    
    def _flush_attempts(self):
        """Flush mitigation attempts buffer to database."""
        if not self._write_buffer['attempts']:
            return
        
        with self._get_db() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO mitigation_attempts 
                (attempt_id, timestamp, initial_state, initial_metrics, error_context,
                 strategy_name, interventions_executed, execution_time, outcome,
                 final_state, final_metrics, side_effects)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (a['attempt_id'], a['timestamp'], a['initial_state'], a['initial_metrics'],
                 a['error_context'], a['strategy_name'], a['interventions_executed'],
                 a['execution_time'], a['outcome'], a['final_state'],
                 a['final_metrics'], a['side_effects'])
                for a in self._write_buffer['attempts']
            ])
            conn.commit()
        
        self._write_buffer['attempts'].clear()
    
    def _flush_learning(self):
        """Flush learning data buffer to database."""
        if not self._write_buffer['learning']:
            return
        
        with self._get_db() as conn:
            conn.executemany("""
                INSERT INTO learning_data 
                (timestamp, learning_type, state_context, action_taken, reward, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, [
                (l['timestamp'], l['learning_type'], l['state_context'],
                 l['action_taken'], l['reward'], l['metadata'])
                for l in self._write_buffer['learning']
            ])
            conn.commit()
        
        self._write_buffer['learning'].clear()
    
    def flush_all(self):
        """Flush all pending writes."""
        with self._lock:
            self._flush_metrics()
            self._flush_checkpoints()
            self._flush_predictions()
            self._flush_attempts()
            self._flush_learning()
            self._last_flush = time.time()
    
    def _flush_worker(self):
        """Background worker to periodically flush buffers."""
        while True:
            time.sleep(self.config.flush_interval_seconds)
            
            if time.time() - self._last_flush >= self.config.flush_interval_seconds:
                self.flush_all()
    
    # Data retention and cleanup
    
    def cleanup_old_data(self):
        """Remove data older than retention policies."""
        with self._get_db() as conn:
            # Calculate cutoff timestamps
            now = time.time()
            metrics_cutoff = now - (self.config.metrics_retention_days * 86400)
            checkpoint_cutoff = now - (self.config.checkpoint_retention_days * 86400)
            prediction_cutoff = now - (self.config.prediction_retention_days * 86400)
            
            # Delete old metrics
            conn.execute("DELETE FROM metrics WHERE timestamp < ?", (metrics_cutoff,))
            
            # Delete old checkpoints
            conn.execute("DELETE FROM checkpoints WHERE timestamp < ?", (checkpoint_cutoff,))
            
            # Delete old predictions
            conn.execute("DELETE FROM predictions WHERE timestamp < ?", (prediction_cutoff,))
            
            # Vacuum to reclaim space
            conn.execute("VACUUM")
            conn.commit()
    
    # Export/Import functionality
    
    def export_data(self, output_dir: Path, 
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None):
        """Export all data to a directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export database
        db_path = self.config.storage_dir / self.config.db_file
        import shutil
        shutil.copy(db_path, output_dir / "monitor.db")
        
        # Export JSON files
        for filename in [self.config.plugins_file, self.config.guardrails_file,
                        self.config.strategies_file, self.config.happy_paths_file]:
            src = self.config.storage_dir / filename
            if src.exists():
                shutil.copy(src, output_dir / filename)
        
        # Export models
        models_src = self.config.storage_dir / self.config.models_dir
        if models_src.exists():
            shutil.copytree(models_src, output_dir / self.config.models_dir)
        
        # Create metadata file
        metadata = {
            'export_time': time.time(),
            'start_time': start_time,
            'end_time': end_time,
            'config': asdict(self.config)
        }
        
        with open(output_dir / "export_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def import_data(self, import_dir: Path, merge: bool = True):
        """Import data from an export directory."""
        if not import_dir.exists():
            raise FileNotFoundError(f"Import directory {import_dir} not found")
        
        # TODO: Implement import logic with merge capabilities
        pass
    
    # Statistics and analytics
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._get_db() as conn:
            stats = {}
            
            # Get table sizes
            tables = ['metrics', 'checkpoints', 'predictions', 
                     'mitigation_attempts', 'happy_paths', 'learning_data']
            
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                stats[f"{table}_count"] = cursor.fetchone()[0]
            
            # Get database size
            db_path = self.config.storage_dir / self.config.db_file
            stats['db_size_mb'] = db_path.stat().st_size / (1024 * 1024)
            
            # Get date ranges
            cursor = conn.execute("SELECT MIN(timestamp), MAX(timestamp) FROM metrics")
            row = cursor.fetchone()
            if row[0]:
                stats['oldest_metric'] = datetime.fromtimestamp(row[0])
                stats['newest_metric'] = datetime.fromtimestamp(row[1])
            
            return stats


class PersistentMonitor:
    """
    Wrapper that adds persistence to any monitor component.
    """
    
    def __init__(self, monitor, persistence_manager: PersistenceManager):
        self.monitor = monitor
        self.persistence = persistence_manager
        
        # Hook into monitor methods
        self._setup_hooks()
    
    def _setup_hooks(self):
        """Setup hooks to persist data automatically."""
        # Hook metric collection
        if hasattr(self.monitor, 'collect_state'):
            original_collect = self.monitor.collect_state
            
            def persistent_collect(state: Dict[str, Any]):
                result = original_collect(state)
                
                # Save metrics to persistence
                for key, value in state.items():
                    if isinstance(value, (int, float)):
                        self.persistence.save_metric(
                            metric_type='state',
                            metric_name=key,
                            value=value
                        )
                
                return result
            
            self.monitor.collect_state = persistent_collect
        
        # Hook checkpoint tracking
        if hasattr(self.monitor, 'track_lifecycle_state'):
            original_track = self.monitor.track_lifecycle_state
            
            def persistent_track(*args, **kwargs):
                checkpoint = original_track(*args, **kwargs)
                
                # Save checkpoint to persistence
                self.persistence.save_checkpoint(checkpoint)
                
                return checkpoint
            
            self.monitor.track_lifecycle_state = persistent_track


# Utility functions

def create_persistent_monitor(monitor, config: Optional[PersistenceConfig] = None):
    """Create a monitor with persistence enabled."""
    persistence = PersistenceManager(config)
    return PersistentMonitor(monitor, persistence)


def get_default_persistence_manager() -> PersistenceManager:
    """Get the default persistence manager instance."""
    return PersistenceManager()