# This is a reference Python file for notebook 25: Statistical Process Control
# The code here will be converted into notebook cells

# Cell 3: Create Interactive SPC Interface

# Global state
spc_results = {}
control_chart_results = {}
capability_results = {}
multivariate_results = {}
rule_violation_results = {}
baseline_results = {}
current_spc_type = None

# ============================================
# Helper Functions for Demo Data Generation
# ============================================

def generate_demo_process_data(n_samples=100, mean=10.0, std=1.0, subgroup_size=5, include_outliers=False):
    """Generate demo process data for control charts."""
    np.random.seed(42)
    if subgroup_size > 1:
        # Generate subgrouped data for X-bar/R charts
        n_subgroups = n_samples // subgroup_size
        data = []
        for i in range(n_subgroups):
            subgroup = np.random.normal(mean, std, subgroup_size)
            if include_outliers and i % 10 == 0:
                subgroup[0] += 4 * std  # Add outlier
            data.append(subgroup)
        return np.array(data)
    else:
        # Individual measurements
        data = np.random.normal(mean, std, n_samples)
        if include_outliers:
            data[::10] += 4 * std  # Add outliers
        return data

def generate_demo_multivariate_data(n_samples=100, n_variables=3, correlation=0.7):
    """Generate demo multivariate process data."""
    np.random.seed(42)
    # Create correlated multivariate data
    cov_matrix = np.eye(n_variables) * 0.5
    for i in range(n_variables):
        for j in range(i+1, n_variables):
            cov_matrix[i, j] = correlation
            cov_matrix[j, i] = correlation
    
    mean = np.zeros(n_variables)
    data = np.random.multivariate_normal(mean, cov_matrix, n_samples)
    return data

def generate_demo_capability_data(n_samples=200, mean=10.0, std=1.0, usl=14.0, lsl=6.0):
    """Generate demo data for capability analysis."""
    np.random.seed(42)
    data = np.random.normal(mean, std, n_samples)
    return data, (lsl, usl)

# ============================================
# Top Panel: SPC Type Selection and Actions
# ============================================

spc_type_label = WidgetHTML("<b>SPC Analysis Type:</b>")
spc_type = RadioButtons(
    options=[
        ('Control Charts', 'control_charts'),
        ('Process Capability', 'capability'),
        ('Multivariate SPC', 'multivariate'),
        ('Control Rules', 'rules'),
        ('Baseline Calculation', 'baseline'),
        ('Comprehensive Analysis', 'comprehensive')
    ],
    value='control_charts',
    description='Type:',
    style={'description_width': 'initial'}
)

data_source_label = WidgetHTML("<b>Data Source:</b>")
data_source_mode = RadioButtons(
    options=[('Demo Data', 'demo'), ('MongoDB', 'mongodb')],
    value='demo',
    description='Source:',
    style={'description_width': 'initial'}
)

execute_button = Button(
    description='Execute SPC Analysis',
    button_style='success',
    icon='check',
    layout=Layout(width='200px')
)

export_button = Button(
    description='Export Report',
    button_style='',
    icon='download',
    layout=Layout(width='150px')
)

top_panel = VBox([
    HBox([spc_type_label, spc_type]),
    HBox([data_source_label, data_source_mode, execute_button, export_button])
], layout=Layout(padding='10px', border='1px solid #ccc'))

# ============================================
# Left Panel: Configuration Accordion
# ============================================

# 1. Control Charts Configuration
control_charts_label = WidgetHTML("<b>Control Chart Configuration:</b>")
chart_type = Dropdown(
    options=[
        ('X-bar', 'xbar'),
        ('R Chart', 'r'),
        ('S Chart', 's'),
        ('Individual', 'individual'),
        ('Moving Range', 'moving_range'),
        ('X-bar & R', 'xbar_r'),
        ('X-bar & S', 'xbar_s')
    ],
    value='xbar',
    description='Chart Type:',
    style={'description_width': 'initial'}
)

subgroup_size = IntSlider(
    value=5,
    min=2,
    max=20,
    step=1,
    description='Subgroup Size:',
    style={'description_width': 'initial'}
)

control_limit_sigma = FloatSlider(
    value=3.0,
    min=2.0,
    max=4.0,
    step=0.1,
    description='Control Limit (σ):',
    style={'description_width': 'initial'}
)

enable_warnings = Checkbox(
    value=True,
    description='Enable Warning Limits (2σ)'
)

n_samples = IntSlider(
    value=100,
    min=20,
    max=500,
    step=10,
    description='Sample Size:',
    style={'description_width': 'initial'}
)

control_charts_config = VBox([
    control_charts_label,
    chart_type,
    subgroup_size,
    control_limit_sigma,
    enable_warnings,
    n_samples
], layout=Layout(padding='5px', border='1px solid #ddd'))

# 2. Process Capability Configuration
capability_label = WidgetHTML("<b>Process Capability Configuration:</b>")
spec_usl = FloatText(
    value=14.0,
    description='USL:',
    style={'description_width': 'initial'}
)

spec_lsl = FloatText(
    value=6.0,
    description='LSL:',
    style={'description_width': 'initial'}
)

target_value = FloatText(
    value=10.0,
    description='Target:',
    style={'description_width': 'initial'}
)

capability_sample_size = IntSlider(
    value=200,
    min=30,
    max=1000,
    step=10,
    description='Sample Size:',
    style={'description_width': 'initial'}
)

capability_config = VBox([
    capability_label,
    spec_usl,
    spec_lsl,
    target_value,
    capability_sample_size
], layout=Layout(padding='5px', border='1px solid #ddd'))

# 3. Multivariate SPC Configuration
multivariate_label = WidgetHTML("<b>Multivariate SPC Configuration:</b>")
multivariate_method = RadioButtons(
    options=[('Hotelling T²', 'hotelling_t2'), ('PCA-based', 'pca')],
    value='hotelling_t2',
    description='Method:',
    style={'description_width': 'initial'}
)

n_variables = IntSlider(
    value=3,
    min=2,
    max=10,
    step=1,
    description='# Variables:',
    style={'description_width': 'initial'}
)

multivariate_sample_size = IntSlider(
    value=100,
    min=30,
    max=500,
    step=10,
    description='Sample Size:',
    style={'description_width': 'initial'}
)

n_components = IntSlider(
    value=2,
    min=1,
    max=5,
    step=1,
    description='PCA Components:',
    style={'description_width': 'initial'}
)

multivariate_config = VBox([
    multivariate_label,
    multivariate_method,
    n_variables,
    multivariate_sample_size,
    n_components
], layout=Layout(padding='5px', border='1px solid #ddd'))

# 4. Control Rules Configuration
rules_label = WidgetHTML("<b>Control Rules Configuration:</b>")
rule_set = RadioButtons(
    options=[('Western Electric', 'western_electric'), ('Nelson', 'nelson'), ('Both', 'both')],
    value='both',
    description='Rule Set:',
    style={'description_width': 'initial'}
)

enable_all_rules = Checkbox(
    value=True,
    description='Enable All Rules'
)

rules_config = VBox([
    rules_label,
    rule_set,
    enable_all_rules
], layout=Layout(padding='5px', border='1px solid #ddd'))

# 5. Baseline Configuration
baseline_label = WidgetHTML("<b>Baseline Configuration:</b>")
baseline_sample_size = IntSlider(
    value=100,
    min=30,
    max=500,
    step=10,
    description='Baseline Samples:',
    style={'description_width': 'initial'}
)

adaptive_limits = Checkbox(
    value=False,
    description='Enable Adaptive Limits'
)

update_frequency = IntSlider(
    value=50,
    min=10,
    max=200,
    step=10,
    description='Update Frequency:',
    style={'description_width': 'initial'}
)

baseline_method = RadioButtons(
    options=[('Exponential Smoothing', 'exponential_smoothing'), ('Cumulative', 'cumulative')],
    value='exponential_smoothing',
    description='Update Method:',
    style={'description_width': 'initial'}
)

baseline_config = VBox([
    baseline_label,
    baseline_sample_size,
    adaptive_limits,
    update_frequency,
    baseline_method
], layout=Layout(padding='5px', border='1px solid #ddd'))

# Combine into Accordion
config_accordion = Accordion(children=[
    control_charts_config,
    capability_config,
    multivariate_config,
    rules_config,
    baseline_config
])

config_accordion.set_title(0, '📊 Control Charts')
config_accordion.set_title(1, '📈 Process Capability')
config_accordion.set_title(2, '🔬 Multivariate SPC')
config_accordion.set_title(3, '🎯 Control Rules')
config_accordion.set_title(4, '📐 Baseline')

left_panel = VBox([
    WidgetHTML("<h3>SPC Configuration</h3>"),
    config_accordion
], layout=Layout(width='300px', padding='10px', border='1px solid #ccc'))

# ============================================
# Center Panel: Visualization and Results
# ============================================

viz_mode = RadioButtons(
    options=[
        ('Control Charts', 'charts'),
        ('Process Capability', 'capability'),
        ('Multivariate SPC', 'multivariate'),
        ('Rule Violations', 'rules'),
        ('Baseline Statistics', 'baseline'),
        ('Comprehensive Report', 'report')
    ],
    value='charts',
    description='View:',
    style={'description_width': 'initial'}
)

main_output = Output(layout=Layout(height='600px', overflow='auto'))

center_panel = VBox([
    WidgetHTML("<h3>SPC Results</h3>"),
    viz_mode,
    main_output
], layout=Layout(flex='1 1 auto', padding='10px', border='1px solid #ccc'))

# ============================================
# Right Panel: Status and Summary
# ============================================

status_label = WidgetHTML("<b>Status:</b>")
status_display = WidgetHTML("Ready for SPC analysis")

progress_bar = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    description='Progress:',
    bar_style='info',
    layout=Layout(width='100%')
)

status_section = VBox([
    status_label,
    status_display,
    progress_bar
], layout=Layout(padding='5px', border='2px solid #4CAF50'))

results_summary_label = WidgetHTML("<b>Results Summary:</b>")
results_summary_display = WidgetHTML("No SPC analysis executed yet")
results_summary_section = VBox([
    results_summary_label,
    results_summary_display
], layout=Layout(padding='5px'))

metrics_display_label = WidgetHTML("<b>Key Metrics:</b>")
metrics_display = WidgetHTML("No metrics available")
metrics_section = VBox([
    metrics_display_label,
    metrics_display
], layout=Layout(padding='5px'))

spc_status_label = WidgetHTML("<b>SPC Status:</b>")
spc_status_display = WidgetHTML("Not analyzed")
spc_status_section = VBox([
    spc_status_label,
    spc_status_display
], layout=Layout(padding='5px'))

# SPC logs output
spc_logs = Output(layout=Layout(height='200px', border='1px solid #ccc', overflow_y='auto'))

# Initialize logs
with spc_logs:
    display(HTML("<p><i>SPC logs will appear here...</i></p>"))

# Bottom status bar
bottom_status = WidgetHTML(value='<b>Status:</b> Ready | <b>Progress:</b> 0% | <b>Time:</b> 0:00')
bottom_progress = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    description='Overall:',
    bar_style='info',
    layout=Layout(width='100%')
)

right_panel = VBox([
    status_section,
    results_summary_section,
    metrics_section,
    spc_status_section
], layout=Layout(width='250px', padding='10px', border='1px solid #ccc'))

# Enhanced bottom panel
bottom_panel = VBox([
    WidgetHTML("<b>SPC Analysis Logs:</b>"),
    spc_logs,
    WidgetHTML("<hr>"),
    bottom_status,
    bottom_progress
], layout=Layout(padding='10px', border='1px solid #ccc'))

# Global time tracking
operation_start_time = None

# ============================================
# Logging Functions
# ============================================

def log_message(message: str, level: str = 'info'):
    """Log a message to the SPC logs with timestamp and emoji."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    icons = {'info': 'ℹ️', 'success': '✅', 'warning': '⚠️', 'error': '❌'}
    icon = icons.get(level, 'ℹ️')
    with spc_logs:
        print(f"[{timestamp}] {icon} {message}")

def update_status(operation: str, progress: int = None):
    """Update the status display and progress."""
    status_display.value = f"<b>{operation}</b>"
    if progress is not None:
        progress_bar.value = progress
        bottom_progress.value = progress
        elapsed = time.time() - operation_start_time if operation_start_time else 0
        mins, secs = divmod(int(elapsed), 60)
        bottom_status.value = f'<b>Status:</b> {operation} | <b>Progress:</b> {progress}% | <b>Time:</b> {mins}:{secs:02d}'

# ... (Execution functions will be added in next cells)