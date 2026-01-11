# Streaming Module API Reference

## Overview

The Streaming module provides real-time data streaming capabilities for AM-QADF, enabling live monitoring and processing of manufacturing process data. It includes Kafka integration, incremental processing, buffer management, low-latency stream processing, and stream data storage.

## StreamingConfig

Configuration dataclass for streaming operations.

```python
from am_qadf.streaming import StreamingConfig

config = StreamingConfig(
    kafka_bootstrap_servers: List[str] = ['localhost:9092'],
    kafka_topic: str = 'am_qadf_monitoring',
    consumer_group_id: str = 'am_qadf_consumers',
    enable_auto_commit: bool = True,
    auto_commit_interval_ms: int = 5000,
    max_poll_records: int = 100,
    session_timeout_ms: int = 30000,
    buffer_size: int = 1000,
    processing_batch_size: int = 100,
    enable_redis_cache: bool = True,
    redis_host: str = 'localhost',
    redis_port: int = 6379,
    redis_db: int = 0,
    enable_mongodb_storage: bool = True,
    storage_batch_size: int = 1000
)
```

### Fields

- `kafka_bootstrap_servers` (List[str]): Kafka bootstrap servers (default: `['localhost:9092']`)
- `kafka_topic` (str): Default Kafka topic name (default: `'am_qadf_monitoring'`)
- `consumer_group_id` (str): Kafka consumer group ID (default: `'am_qadf_consumers'`)
- `enable_auto_commit` (bool): Enable automatic offset commits (default: `True`)
- `auto_commit_interval_ms` (int): Auto commit interval in milliseconds (default: `5000`)
- `max_poll_records` (int): Maximum records per poll (default: `100`)
- `session_timeout_ms` (int): Session timeout in milliseconds (default: `30000`)
- `buffer_size` (int): Buffer size for temporal windows (default: `1000`)
- `processing_batch_size` (int): Batch size for incremental processing (default: `100`)
- `enable_redis_cache` (bool): Enable Redis caching (default: `True`)
- `redis_host` (str): Redis host (default: `'localhost'`)
- `redis_port` (int): Redis port (default: `6379`)
- `redis_db` (int): Redis database number (default: `0`)
- `enable_mongodb_storage` (bool): Enable MongoDB storage (default: `True`)
- `storage_batch_size` (int): Batch size for MongoDB writes (default: `1000`)

---

## StreamingResult

Result dataclass for streaming data processing.

```python
from am_qadf.streaming import StreamingResult
from datetime import datetime

result = StreamingResult(
    timestamp: datetime,
    data_batch: np.ndarray,
    processed_count: int,
    processing_time_ms: float,
    voxel_updates: Optional[Dict[str, Any]] = None,
    quality_metrics: Optional[Dict[str, float]] = None,
    spc_results: Optional[Dict[str, Any]] = None,
    alerts_generated: List[str] = field(default_factory=list),
    metadata: Dict[str, Any] = field(default_factory=dict)
)
```

### Fields

- `timestamp` (datetime): Processing timestamp
- `data_batch` (np.ndarray): Processed data batch
- `processed_count` (int): Number of messages processed
- `processing_time_ms` (float): Processing time in milliseconds
- `voxel_updates` (Optional[Dict[str, Any]]): Voxel grid update results
- `quality_metrics` (Optional[Dict[str, float]]): Quality assessment metrics
- `spc_results` (Optional[Dict[str, Any]]): SPC analysis results
- `alerts_generated` (List[str]): List of generated alert IDs
- `metadata` (Dict[str, Any]): Additional metadata

---

## StreamingClient

Main streaming client interface for all streaming operations.

```python
from am_qadf.streaming import StreamingClient, StreamingConfig

client = StreamingClient(config: Optional[StreamingConfig] = None)
```

### Methods

#### `__init__(config: Optional[StreamingConfig] = None)`

Initialize the streaming client.

**Parameters**:
- `config` (Optional[StreamingConfig]): Streaming configuration. If None, uses default config.

---

#### `start_consumer(topics: List[str], callback: Callable) -> None`

Start Kafka consumer and process messages in background thread.

**Parameters**:
- `topics` (List[str]): List of Kafka topics to consume from
- `callback` (Callable): Callback function called with each message batch: `callback(messages: List[Dict]) -> None`

**Example**:
```python
def process_batch(messages):
    print(f"Received {len(messages)} messages")
    for msg in messages:
        print(f"Message: {msg}")

client.start_consumer(['am_qadf_monitoring'], process_batch)
```

---

#### `stop_consumer() -> None`

Stop Kafka consumer and background thread.

**Example**:
```python
client.stop_consumer()
```

---

#### `process_stream_batch(data_batch: List[Dict[str, Any]]) -> StreamingResult`

Process a batch of streaming data through registered processors.

**Parameters**:
- `data_batch` (List[Dict[str, Any]]): List of data point dictionaries

**Returns**: `StreamingResult` containing processing results and statistics.

**Example**:
```python
batch = [
    {'value': 1.0, 'timestamp': datetime.now()},
    {'value': 2.0, 'timestamp': datetime.now()},
    {'value': 3.0, 'timestamp': datetime.now()},
]
result = client.process_stream_batch(batch)
print(f"Processed {result.processed_count} messages in {result.processing_time_ms:.2f} ms")
```

---

#### `register_processor(name: str, processor: Callable) -> None`

Register a custom processor function.

**Parameters**:
- `name` (str): Processor name (identifier)
- `processor` (Callable): Processor function: `processor(data_batch: List[Dict]) -> Dict[str, Any]`

**Example**:
```python
def my_processor(data_batch):
    values = [item.get('value', 0.0) for item in data_batch]
    return {
        'processed': True,
        'count': len(values),
        'sum': sum(values),
    }

client.register_processor('my_processor', my_processor)
```

---

#### `get_stream_statistics() -> Dict[str, Any]`

Get streaming statistics.

**Returns**: Dictionary containing:
- `messages_processed` (int): Total messages processed
- `batches_processed` (int): Total batches processed
- `average_latency_ms` (float): Average processing latency
- `throughput_messages_per_sec` (float): Throughput in messages/second
- `errors` (int): Number of errors encountered

**Example**:
```python
stats = client.get_stream_statistics()
print(f"Throughput: {stats['throughput_messages_per_sec']:.2f} msg/s")
print(f"Average latency: {stats['average_latency_ms']:.2f} ms")
```

---

#### `reset_statistics() -> None`

Reset streaming statistics.

**Example**:
```python
client.reset_statistics()
```

---

## KafkaConsumer

Kafka consumer implementation with support for multiple backends.

```python
from am_qadf.streaming import KafkaConsumer

consumer = KafkaConsumer(
    bootstrap_servers: List[str],
    topic: str,
    group_id: str = 'am_qadf_consumers',
    auto_commit: bool = True
)
```

### Methods

#### `__init__(bootstrap_servers: List[str], topic: str, group_id: str = 'am_qadf_consumers', auto_commit: bool = True)`

Initialize Kafka consumer.

**Parameters**:
- `bootstrap_servers` (List[str]): Kafka bootstrap servers
- `topic` (str): Kafka topic name
- `group_id` (str): Consumer group ID (default: `'am_qadf_consumers'`)
- `auto_commit` (bool): Enable automatic offset commits (default: `True`)

---

#### `poll(timeout_ms: int = 1000) -> List[Dict[str, Any]]`

Poll for messages from Kafka.

**Parameters**:
- `timeout_ms` (int): Poll timeout in milliseconds (default: `1000`)

**Returns**: List of message dictionaries with keys: `topic`, `partition`, `offset`, `timestamp`, `key`, `value`

**Example**:
```python
messages = consumer.poll(timeout_ms=5000)
for msg in messages:
    print(f"Message: {msg['value']}")
```

---

#### `commit() -> None`

Manually commit offsets.

**Example**:
```python
consumer.commit()
```

---

#### `seek_to_beginning() -> None`

Seek to beginning of topic.

**Example**:
```python
consumer.seek_to_beginning()
```

---

#### `pause() -> None`

Pause message consumption.

**Example**:
```python
consumer.pause()
```

---

#### `resume() -> None`

Resume message consumption.

**Example**:
```python
consumer.resume()
```

---

#### `close() -> None`

Close consumer and release resources.

**Example**:
```python
consumer.close()
```

---

## KafkaProducer

Kafka producer for publishing messages.

```python
from am_qadf.streaming import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers: List[str],
    topic: str
)
```

### Methods

#### `__init__(bootstrap_servers: List[str], topic: str)`

Initialize Kafka producer.

**Parameters**:
- `bootstrap_servers` (List[str]): Kafka bootstrap servers
- `topic` (str): Kafka topic name

---

#### `produce(value: Dict[str, Any], key: Optional[str] = None) -> None`

Produce a message to Kafka topic.

**Parameters**:
- `value` (Dict[str, Any]): Message value (will be JSON serialized)
- `key` (Optional[str]): Optional message key

**Example**:
```python
producer.produce({'sensor_id': 'sensor1', 'value': 100.0}, key='sensor1')
```

---

#### `flush() -> None`

Flush pending messages to Kafka.

**Example**:
```python
producer.flush()
```

---

#### `close() -> None`

Close producer and release resources.

**Example**:
```python
producer.close()
```

---

## IncrementalProcessor

Processes streaming data incrementally to update voxel grids.

```python
from am_qadf.streaming import IncrementalProcessor
from am_qadf.voxelization.voxel_grid import VoxelGrid

processor = IncrementalProcessor(voxel_grid: VoxelGrid)
```

### Methods

#### `__init__(voxel_grid: VoxelGrid)`

Initialize incremental processor.

**Parameters**:
- `voxel_grid` (VoxelGrid): Voxel grid to update incrementally

---

#### `update_voxel_grid(new_data: np.ndarray, coordinates: np.ndarray) -> VoxelGrid`

Update voxel grid with new data points.

**Parameters**:
- `new_data` (np.ndarray): Array of signal values (shape: `[n_points]`)
- `coordinates` (np.ndarray): Array of coordinates (shape: `[n_points, 3]`)

**Returns**: Updated `VoxelGrid`

**Example**:
```python
import numpy as np

coordinates = np.array([
    [10.0, 20.0, 30.0],
    [11.0, 21.0, 31.0],
])
new_data = np.array([1000.0, 1050.0])

updated_grid = processor.update_voxel_grid(new_data, coordinates)
```

---

#### `get_updated_regions() -> List[Tuple[int, int, int]]`

Get list of updated voxel regions (for efficient visualization).

**Returns**: List of (i, j, k) voxel indices that were updated

**Example**:
```python
regions = processor.get_updated_regions()
print(f"Updated {len(regions)} voxels")
```

---

#### `reset_updated_regions() -> None`

Reset updated regions tracking.

**Example**:
```python
processor.reset_updated_regions()
```

---

#### `get_statistics() -> Dict[str, Any]`

Get processing statistics.

**Returns**: Dictionary with `total_points_processed`, `updated_regions_count`, etc.

**Example**:
```python
stats = processor.get_statistics()
print(f"Processed {stats['total_points_processed']} points")
```

---

## BufferManager

Manages temporal windows and buffers for streaming data.

```python
from am_qadf.streaming import BufferManager

buffer = BufferManager(
    window_size: int = 100,
    buffer_size: int = 1000
)
```

### Methods

#### `__init__(window_size: int = 100, buffer_size: int = 1000)`

Initialize buffer manager.

**Parameters**:
- `window_size` (int): Default sliding window size (default: `100`)
- `buffer_size` (int): Maximum buffer size (default: `1000`)

---

#### `add_data(data: np.ndarray, timestamp: datetime, metadata: Optional[Dict[str, Any]] = None) -> None`

Add data point to buffer.

**Parameters**:
- `data` (np.ndarray): Data point array
- `timestamp` (datetime): Timestamp for data point
- `metadata` (Optional[Dict[str, Any]]): Optional metadata dictionary

**Example**:
```python
from datetime import datetime
import numpy as np

buffer.add_data(np.array([100.0]), datetime.now(), metadata={'sensor_id': 'sensor1'})
```

---

#### `get_sliding_window(size: Optional[int] = None) -> Tuple[np.ndarray, List[datetime], List[Dict[str, Any]]]`

Get sliding window of recent data.

**Parameters**:
- `size` (Optional[int]): Window size. If None, uses default window_size.

**Returns**: Tuple of (data_array, timestamps, metadata_list)

**Example**:
```python
window, timestamps, metadata = buffer.get_sliding_window(50)
print(f"Window shape: {window.shape}")
print(f"Time range: {timestamps[0]} to {timestamps[-1]}")
```

---

#### `get_time_window(duration_seconds: float) -> Tuple[np.ndarray, List[datetime], List[Dict[str, Any]]]`

Get all data within time duration.

**Parameters**:
- `duration_seconds` (float): Time duration in seconds

**Returns**: Tuple of (data_array, timestamps, metadata_list)

**Example**:
```python
window, timestamps, metadata = buffer.get_time_window(60.0)  # Last 60 seconds
print(f"Found {len(window)} points in last minute")
```

---

#### `flush_buffer() -> Tuple[np.ndarray, List[datetime], List[Dict[str, Any]]]`

Flush entire buffer and return all data.

**Returns**: Tuple of (data_array, timestamps, metadata_list)

**Example**:
```python
all_data, timestamps, metadata = buffer.flush_buffer()
buffer.clear()  # Clear after flushing
```

---

#### `clear() -> None`

Clear buffer.

**Example**:
```python
buffer.clear()
```

---

#### `get_buffer_statistics() -> Dict[str, Any]`

Get buffer statistics.

**Returns**: Dictionary with `current_size`, `max_size`, `utilization_percent`, `oldest_timestamp`, `newest_timestamp`

**Example**:
```python
stats = buffer.get_buffer_statistics()
print(f"Buffer utilization: {stats['utilization_percent']:.1f}%")
```

---

## StreamProcessor

Low-latency stream processing pipeline.

```python
from am_qadf.streaming import StreamProcessor, StreamingConfig

processor = StreamProcessor(config: StreamingConfig)
```

### Methods

#### `__init__(config: StreamingConfig)`

Initialize stream processor.

**Parameters**:
- `config` (StreamingConfig): Streaming configuration

---

#### `create_processing_pipeline(stages: List[Callable]) -> Callable`

Create processing pipeline from stages.

**Parameters**:
- `stages` (List[Callable]): List of processing stage functions

**Returns**: Pipeline function that processes data through all stages

**Example**:
```python
def stage1(data):
    return [x * 2 for x in data]

def stage2(data):
    return [x + 10 for x in data]

pipeline = processor.create_processing_pipeline([stage1, stage2])
result = processor.process_with_pipeline([1, 2, 3], pipeline)
# Result: [12, 14, 16]  # [1*2+10, 2*2+10, 3*2+10]
```

---

#### `process_with_pipeline(data: List[Any], pipeline: Callable) -> Any`

Process data through pipeline.

**Parameters**:
- `data` (List[Any]): Input data
- `pipeline` (Callable): Pipeline function

**Returns**: Processed data (output of final stage)

**Example**:
```python
result = processor.process_with_pipeline([1, 2, 3], pipeline)
```

---

#### `add_quality_checkpoint(name: str, checkpoint: Callable) -> None`

Add quality checkpoint to pipeline.

**Parameters**:
- `name` (str): Checkpoint name
- `checkpoint` (Callable): Checkpoint function: `checkpoint(data) -> bool`

**Example**:
```python
processor.add_quality_checkpoint('non_empty', lambda x: len(x) > 0)
processor.add_quality_checkpoint('all_positive', lambda x: all(v > 0 for v in x))
```

---

#### `validate_all_checkpoints(data: List[Any]) -> Dict[str, bool]`

Validate data against all checkpoints.

**Parameters**:
- `data` (List[Any]): Data to validate

**Returns**: Dictionary mapping checkpoint names to validation results (True/False)

**Example**:
```python
results = processor.validate_all_checkpoints([1, 2, 3])
# Returns: {'non_empty': True, 'all_positive': True}
```

---

#### `enable_parallel_processing(num_workers: int = 4) -> None`

Enable parallel processing for multiple streams.

**Parameters**:
- `num_workers` (int): Number of worker threads (default: `4`)

**Example**:
```python
processor.enable_parallel_processing(num_workers=8)
```

---

#### `process_parallel(batches: List[List[Any]], pipeline: Callable) -> List[Any]`

Process multiple batches in parallel.

**Parameters**:
- `batches` (List[List[Any]]): List of data batches
- `pipeline` (Callable): Pipeline function

**Returns**: List of processed results (one per batch)

**Example**:
```python
batches = [[1, 2], [3, 4], [5, 6]]
results = processor.process_parallel(batches, pipeline)
```

---

#### `disable_parallel_processing() -> None`

Disable parallel processing.

**Example**:
```python
processor.disable_parallel_processing()
```

---

#### `get_statistics() -> Dict[str, Any]`

Get processing statistics.

**Returns**: Dictionary with latency, throughput, checkpoint results, etc.

**Example**:
```python
stats = processor.get_statistics()
print(f"Average latency: {stats['average_latency_ms']:.2f} ms")
```

---

## StreamStorage

Store stream data in Redis (caching) and MongoDB (persistence).

```python
from am_qadf.streaming import StreamStorage

storage = StreamStorage(
    redis_client: Optional[Any] = None,
    mongo_client: Optional[Any] = None,
    collection_name: str = 'stream_data'
)
```

### Methods

#### `__init__(redis_client: Optional[Any] = None, mongo_client: Optional[Any] = None, collection_name: str = 'stream_data')`

Initialize stream storage.

**Parameters**:
- `redis_client` (Optional[Any]): Optional Redis client
- `mongo_client` (Optional[Any]): Optional MongoDB client
- `collection_name` (str): MongoDB collection name (default: `'stream_data'`)

---

#### `cache_recent_data(key: str, data: Dict[str, Any], ttl_seconds: int = 3600) -> bool`

Cache recent data in Redis.

**Parameters**:
- `key` (str): Cache key
- `data` (Dict[str, Any]): Data to cache
- `ttl_seconds` (int): Time-to-live in seconds (default: `3600`)

**Returns**: `True` if successful, `False` otherwise

**Example**:
```python
storage.cache_recent_data('recent_batch', {'data': [1, 2, 3]}, ttl_seconds=1800)
```

---

#### `store_batch(batch_data: List[Dict[str, Any]]) -> int`

Store batch of data in MongoDB.

**Parameters**:
- `batch_data` (List[Dict[str, Any]]): List of data point dictionaries

**Returns**: Number of documents inserted

**Example**:
```python
batch = [
    {'value': 1.0, 'timestamp': datetime.now()},
    {'value': 2.0, 'timestamp': datetime.now()},
]
inserted = storage.store_batch(batch)
print(f"Inserted {inserted} documents")
```

---

#### `query_stream_history(start_time: datetime, end_time: datetime, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]`

Query stream history by time range.

**Parameters**:
- `start_time` (datetime): Start time
- `end_time` (datetime): End time
- `filters` (Optional[Dict[str, Any]]): Additional MongoDB query filters

**Returns**: List of matching documents

**Example**:
```python
from datetime import datetime, timedelta

start_time = datetime.now() - timedelta(hours=1)
end_time = datetime.now()
history = storage.query_stream_history(start_time, end_time)
print(f"Found {len(history)} records")
```

---

#### `create_time_series_index() -> None`

Create time-series index for efficient queries.

**Example**:
```python
storage.create_time_series_index()
```

---

#### `delete_old_data(before_time: datetime) -> int`

Delete old data before specified time.

**Parameters**:
- `before_time` (datetime): Delete data before this time

**Returns**: Number of documents deleted

**Example**:
```python
cutoff_time = datetime.now() - timedelta(days=30)
deleted = storage.delete_old_data(cutoff_time)
print(f"Deleted {deleted} old records")
```

---

## Related

- [Streaming Module](../05-modules/streaming.md) - Module documentation
- [Monitoring API](monitoring-api.md) - Monitoring integration
- [SPC API](spc-api.md) - SPC integration

---

**Parent**: [API Reference](README.md)
