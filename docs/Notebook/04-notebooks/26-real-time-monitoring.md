# Notebook 26: Real-time Process Monitoring and Control

**File**: `26_Real_Time_Monitoring.ipynb`  
**Category**: Advanced Topics / Real-time Monitoring  
**Duration**: 90-120 minutes

## Purpose

This notebook teaches you how to implement real-time process monitoring and control for additive manufacturing processes. You'll learn to set up streaming data processing, configure live quality dashboards, manage alerts and notifications, integrate SPC for real-time monitoring, and monitor system health using a unified interactive interface with real-time progress tracking and detailed logging.

## Learning Objectives

By the end of this notebook, you will:

- ✅ Set up Kafka-based streaming data processing
- ✅ Configure real-time data streaming with incremental processing
- ✅ Create live quality dashboards with WebSocket updates
- ✅ Configure alert thresholds and notification channels (Email, SMS, Dashboard)
- ✅ Integrate SPC for real-time process monitoring
- ✅ Monitor system and process health
- ✅ Execute complete end-to-end real-time monitoring workflows
- ✅ Monitor streaming and monitoring progress with real-time status and logs

## Topics Covered

### Real-time Monitoring Setup

- **Kafka Configuration**: Set up Kafka bootstrap servers, topics, and consumer groups
- **Streaming Configuration**: Configure buffer size, batch size, and processing options
- **Monitoring Configuration**: Set up alert cooldown, health check intervals, and notification channels
- **SPC Configuration**: Configure SPC control limits and subgroup sizes
- **Dashboard Configuration**: Set up WebSocket ports and update intervals

### Stream Processing

- **Data Stream Consumption**: Consume data from Kafka topics
- **Incremental Processing**: Process streaming data incrementally to update voxel grids
- **Buffer Management**: Manage temporal windows and buffers for streaming data
- **Low-Latency Processing**: Process streams with quality checkpoints
- **Batch Processing**: Process data in batches for efficiency
- **Statistics Tracking**: Monitor throughput, latency, and processing metrics

### Live Quality Dashboards

- **WebSocket Setup**: Configure WebSocket server for live updates
- **Dashboard Creation**: Create real-time quality dashboards
- **Metric Visualization**: Visualize real-time metrics (temperature, power, velocity)
- **Dashboard Updates**: Update dashboards in real-time with streaming data
- **Live Updates**: Enable/disable live updates with configurable intervals

### Alert Configuration

- **Threshold Setup**: Configure absolute, relative, rate-of-change, and SPC-based thresholds
- **Alert Rule Configuration**: Set up alert rules for different metric types
- **Notification Channel Setup**: Configure Email, SMS, and Dashboard notification channels
- **Alert Cooldown**: Prevent alert spam with cooldown periods
- **Alert Testing**: Test alert generation and notification delivery
- **Alert Acknowledgment**: Acknowledge and manage alerts

### Real-time SPC Integration

- **Streaming SPC Monitoring**: Integrate SPC with streaming data processing
- **Control Chart Updates**: Update control charts in real-time with streaming data
- **Rule Violation Detection**: Detect control rule violations in real-time
- **Baseline Management**: Establish and update SPC baselines adaptively
- **Real-time Alerts from SPC**: Generate alerts for out-of-control conditions

### Health Monitoring

- **System Health Checks**: Monitor CPU, memory, disk, and network usage
- **Process Health Monitoring**: Monitor custom process health metrics
- **Health Score Calculation**: Calculate health scores from metrics
- **Health History**: Track health status over time
- **Component Registration**: Register custom components for health monitoring
- **Health Alert Generation**: Generate alerts for degraded/unhealthy components

### Complete Real-time Workflow

- **End-to-End Streaming**: Complete streaming workflow from data source to output
- **Component Integration**: Integrate all components (streaming, monitoring, SPC, health)
- **Performance Monitoring**: Monitor performance metrics (throughput, latency, errors)
- **Troubleshooting**: Debug and troubleshoot real-time monitoring issues

## Interactive Widgets

### Top Panel

- **Operation Type**: Radio buttons
  - Real-time Setup
  - Stream Processing
  - Live Dashboard
  - Alert Configuration
  - Real-time SPC
  - Health Monitoring
  - Complete Workflow
- **Data Source**: Radio buttons (Demo Data / MongoDB / Kafka)
- **Execute Operation**: Button to execute selected operation
- **Stop Operation**: Button to stop current operation
- **Export Results**: Button to export results

### Left Panel (Configuration Accordion)

#### Kafka Configuration
- **Bootstrap Servers**: Text input (default: `localhost:9092`)
- **Topic**: Text input (default: `am_qadf_monitoring`)
- **Consumer Group**: Text input (default: `am_qadf_consumers`)

#### Streaming Configuration
- **Buffer Size**: IntSlider (100-10000, default: 1000)
- **Batch Size**: IntSlider (10-1000, default: 100)
- **Enable Incremental Processing**: Checkbox (default: True)
- **Enable Buffer Management**: Checkbox (default: True)

#### Dashboard Configuration
- **WebSocket Port**: IntText (default: 8765)
- **Update Interval (s)**: FloatSlider (0.1-10.0, default: 1.0)
- **Enable Live Updates**: Checkbox (default: True)

#### Alert Configuration
- **Cooldown (s)**: FloatSlider (0-3600, default: 300.0)
- **Enable Email Notifications**: Checkbox (default: False)
- **Enable SMS Notifications**: Checkbox (default: False)
- **Enable Dashboard Alerts**: Checkbox (default: True)

#### Threshold Configuration
- **Metric Name**: Text input (default: `temperature`)
- **Threshold Type**: Dropdown (absolute, relative, rate_of_change, spc_limit)
- **Lower Threshold**: FloatText (default: 800.0)
- **Upper Threshold**: FloatText (default: 1200.0)

#### SPC Configuration
- **Control Limit (σ)**: FloatSlider (1.0-5.0, default: 3.0)
- **Subgroup Size**: IntSlider (2-20, default: 5)
- **Enable SPC Monitoring**: Checkbox (default: True)

#### Health Configuration
- **Check Interval (s)**: FloatSlider (1.0-600.0, default: 60.0)
- **Enable System Health**: Checkbox (default: True)
- **Enable Process Health**: Checkbox (default: True)

### Center Panel

- **Main Output**: Output widget for plots, logs, dashboards (height: 600px)
  - Displays operation results
  - Shows visualizations (if applicable)
  - Displays execution logs
  - Shows configuration summaries

### Right Panel

- **Status Display**: Current operation status (height: 150px)
  - Operation type
  - Execution status
  - Elapsed time
- **Metrics Display**: Current metric values (height: 150px)
  - Active metrics
  - Current values
  - Last update time
- **Active Alerts**: Alert display (height: 150px)
  - Alert list
  - Severity indicators
  - Acknowledgment status
- **Health Status**: Health monitoring display (height: 150px)
  - Component health status
  - Health scores
  - Identified issues

### Bottom Panel

- **Progress Bar**: Progress indicator (0-100%)
- **Status Text**: Overall status message
- **Logs Output**: Detailed execution logs (height: 200px)
  - Timestamped logs with emoji indicators:
    - ✅ Success messages
    - ⚠️ Warning messages
    - ❌ Error messages (with full tracebacks)
  - Operation progress tracking
  - Error handling and diagnostics

## Key Features

### Real-Time Progress Tracking

- **Progress Bars**: Visual progress indicators (0-100%)
- **Status Updates**: Real-time status updates with elapsed time
- **Time Tracking**: Automatic tracking of execution time for all operations
- **Batch Progress**: Track progress within batch processing operations

### Detailed Logging

- **Timestamped Logs**: All operations logged with timestamps
- **Log Levels**: Info, success, warning, and error messages
- **Error Tracebacks**: Full error tracebacks in logs for debugging
- **Operation Logging**: Log all streaming, monitoring, and SPC operations

### Comprehensive Operations

- **Setup Operations**: Configure all real-time monitoring components
- **Streaming Operations**: Process streaming data with incremental updates
- **Dashboard Operations**: Set up and manage live dashboards
- **Alert Operations**: Configure and manage alerts
- **SPC Operations**: Integrate SPC for real-time monitoring
- **Health Operations**: Monitor system and process health
- **Complete Workflow**: Execute end-to-end real-time monitoring workflow

## Usage Examples

### Real-time Setup

```python
# Select "Real-time Setup" in Operation Type
# Configure settings in left panel accordion:
#   - Kafka: Set bootstrap servers, topic, consumer group
#   - Streaming: Set buffer size (1000), batch size (100)
#   - Dashboard: Set WebSocket port (8765), update interval (1.0s)
#   - Alerts: Set cooldown (300s), enable dashboard alerts
#   - SPC: Set sigma (3.0), subgroup size (5)
#   - Health: Set check interval (60s), enable system/process health
# Click "Execute Operation"

# Results displayed:
# - Configuration summary
# - All components initialized
# - Status: Ready
```

### Stream Processing

```python
# Select "Stream Processing" in Operation Type
# Select "Demo Data" in Data Source
# Configure streaming settings:
#   - Buffer Size: 1000
#   - Batch Size: 100
#   - Enable Incremental Processing: True
#   - Enable Buffer Management: True
# Click "Execute Operation"

# Results displayed:
# - Stream processing statistics
#   - Messages processed
#   - Batches processed
#   - Average latency (ms)
#   - Throughput (messages/second)
# - Progress tracked in logs
```

### Live Dashboard Setup

```python
# Select "Live Dashboard" in Operation Type
# Configure dashboard settings:
#   - WebSocket Port: 8765
#   - Update Interval: 1.0s
#   - Enable Live Updates: True
# Click "Execute Operation"

# Results displayed:
# - Dashboard configuration
# - WebSocket server URL
# - Update interval confirmation
# - Live updates enabled status
```

### Alert Configuration

```python
# Select "Alert Configuration" in Operation Type
# Configure threshold settings:
#   - Metric Name: temperature
#   - Threshold Type: absolute
#   - Lower Threshold: 800.0
#   - Upper Threshold: 1200.0
# Configure notification channels:
#   - Enable Dashboard Alerts: True
#   - Enable Email: False (optional)
#   - Enable SMS: False (optional)
# Set cooldown: 300s
# Click "Execute Operation"

# Results displayed:
# - Alert configuration summary
# - Metric registered
# - Notification channels configured
# - Cooldown period set
```

### Real-time SPC Integration

```python
# Select "Real-time SPC" in Operation Type
# Select "Demo Data" in Data Source
# Configure SPC settings:
#   - Control Limit (σ): 3.0
#   - Subgroup Size: 5
#   - Enable SPC Monitoring: True
# Click "Execute Operation"

# Results displayed:
# - SPC baseline established
# - Control chart generated
#   - Center line
#   - UCL (Upper Control Limit)
#   - LCL (Lower Control Limit)
#   - Out-of-control points count
# - Real-time SPC monitoring active
```

### Health Monitoring

```python
# Select "Health Monitoring" in Operation Type
# Configure health settings:
#   - Check Interval: 60s
#   - Enable System Health: True
#   - Enable Process Health: True
# Click "Execute Operation"

# Results displayed:
# - Health monitoring started
# - Health status for all components:
#   - Component name
#   - Status (healthy/degraded/unhealthy/critical)
#   - Health score (0.0-1.0)
#   - Identified issues (if any)
# - Health monitoring stopped
```

### Complete Workflow

```python
# Select "Complete Workflow" in Operation Type
# Select "Demo Data" in Data Source
# Configure all settings as needed
# Click "Execute Operation"

# Results displayed:
# - All operations executed in sequence:
#   1. Setup completed
#   2. Stream processing completed
#   3. Alert configuration completed
#   4. SPC integration completed (if enabled)
#   5. Health monitoring completed (if enabled)
# - Complete workflow summary
# - Performance metrics
# - All components integrated
```

## Best Practices

1. **Start with Setup**: Always configure all components using "Real-time Setup" before running operations
2. **Configure Thresholds Carefully**: Set appropriate thresholds based on process specifications and historical data
3. **Use Alert Cooldowns**: Set reasonable cooldown periods to prevent alert spam
4. **Monitor Performance**: Regularly check throughput and latency metrics
5. **Health Monitoring**: Enable health monitoring for critical components
6. **SPC Integration**: Establish baselines before enabling real-time SPC monitoring
7. **Dashboard Updates**: Set appropriate update intervals based on data rate and UI responsiveness needs
8. **Buffer Sizing**: Adjust buffer size based on data rate and processing time
9. **Batch Sizing**: Optimize batch size for throughput vs latency trade-off
10. **Error Handling**: Monitor logs regularly for warnings and errors
11. **Complete Workflow**: Use "Complete Workflow" for end-to-end testing and validation
12. **Stop Operations**: Use "Stop Operation" button to gracefully halt long-running operations

## Common Use Cases

### Real-Time Process Monitoring

Monitor manufacturing processes in real-time as they occur:
- Consume streaming data from Kafka topics
- Process data incrementally to update voxel grids
- Monitor quality metrics (temperature, power, velocity)
- Generate alerts for threshold violations
- Update live dashboards with real-time metrics

### Live Quality Dashboards

Create real-time quality dashboards for process monitoring:
- Set up WebSocket server for live updates
- Configure update intervals for responsive UI
- Display real-time quality metrics
- Visualize streaming data trends
- Enable/disable live updates as needed

### Alert Management

Configure and manage alerts for process monitoring:
- Set up thresholds for different metrics
- Configure notification channels (Email, SMS, Dashboard)
- Test alert generation and delivery
- Manage alert cooldowns to prevent spam
- Acknowledge and track alerts

### SPC Real-Time Integration

Integrate SPC for real-time process control:
- Establish SPC baselines from historical data
- Process streaming data with SPC control charts
- Detect out-of-control conditions in real-time
- Generate alerts for SPC violations
- Update baselines adaptively as process changes

### Health Monitoring

Monitor system and process health:
- Monitor system resources (CPU, memory, disk, network)
- Register custom components for health monitoring
- Calculate health scores for all components
- Generate alerts for degraded/unhealthy components
- Track health history over time

### End-to-End Real-Time Workflow

Execute complete real-time monitoring workflows:
- Set up all components (streaming, monitoring, SPC, health)
- Process streaming data end-to-end
- Integrate all components together
- Monitor performance metrics
- Troubleshoot and debug issues

## Troubleshooting

### Streaming Not Working

- **Check Kafka Connection**: Verify Kafka bootstrap servers and topic exist
- **Check Buffer Size**: Ensure buffer size is sufficient for data rate
- **Check Batch Size**: Verify batch size is appropriate for processing time
- **Check Logs**: Review logs for connection errors or processing errors

### Alerts Not Generated

- **Check Thresholds**: Verify thresholds are set correctly
- **Check Metric Registration**: Ensure metric is registered with threshold
- **Check Cooldown**: Verify cooldown period has elapsed
- **Check Metric Values**: Ensure metric values are actually violating thresholds

### Dashboard Not Updating

- **Check WebSocket Port**: Verify port is available and not in use
- **Check Update Interval**: Verify update interval is reasonable (> 0.1s)
- **Check Live Updates**: Ensure "Enable Live Updates" is checked
- **Check Connection**: Verify WebSocket connection is established

### SPC Integration Issues

- **Check Baseline**: Ensure baseline is established before monitoring
- **Check Data Format**: Verify streaming data format matches SPC requirements
- **Check Sample Size**: Ensure sufficient samples for SPC analysis
- **Check Control Limits**: Verify control limits are appropriate for process

### Health Monitoring Issues

- **Check Health Components**: Verify components are registered correctly
- **Check Health Checkers**: Ensure health checker functions return correct format
- **Check Check Interval**: Verify check interval is appropriate
- **Check System Resources**: Verify system resources are available for monitoring

### Performance Issues

- **Check Throughput**: Monitor throughput and adjust batch size if needed
- **Check Latency**: Monitor latency and optimize processing if needed
- **Check Buffer Utilization**: Monitor buffer utilization and adjust size if needed
- **Check Resource Usage**: Monitor CPU/memory usage and optimize if needed

## Related Notebooks

- **[07: Quality Assessment](07-quality.md)** - Quality assessment fundamentals
- **[08: Quality Dashboard](08-quality-dashboard.md)** - Quality monitoring dashboard
- **[25: Statistical Process Control](25-statistical-process-control.md)** - SPC fundamentals and workflows
- **[14: Anomaly Detection Workflow](14-anomaly-workflow.md)** - Anomaly detection using SPC

## Related Documentation

- **[Streaming Module](../../AM_QADF/05-modules/streaming.md)** - Complete streaming module documentation
- **[Monitoring Module](../../AM_QADF/05-modules/monitoring.md)** - Complete monitoring module documentation
- **[Streaming API Reference](../../AM_QADF/06-api-reference/streaming-api.md)** - Streaming API documentation
- **[Monitoring API Reference](../../AM_QADF/06-api-reference/monitoring-api.md)** - Monitoring API documentation
- **[SPC Module Documentation](../../AM_QADF/05-modules/spc.md)** - SPC module documentation
- **[SPC API Reference](../../AM_QADF/06-api-reference/spc-api.md)** - SPC API documentation
- **[Implementation Plan](../../../implementation_plans/REALTIME_MONITORING_IMPLEMENTATION.md)** - Implementation details

---

**Next**: Explore the streaming and monitoring module APIs for advanced use cases, integrate real-time monitoring into your quality assessment workflows, and customize alert thresholds and notification channels for your specific manufacturing processes.

**Previous**: [25: Statistical Process Control](25-statistical-process-control.md)
