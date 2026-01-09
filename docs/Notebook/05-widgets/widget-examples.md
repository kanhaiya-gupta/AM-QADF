# Widget Examples

**Version**: 1.0  
**Last Updated**: 2024

## Overview

This document provides code examples for common widget patterns used in AM-QADF Interactive Notebooks.

## Basic Examples

### Example 1: Simple Dropdown

```python
from ipywidgets import Dropdown

mode = Dropdown(
    options=[('Option 1', 'opt1'), ('Option 2', 'opt2')],
    value='opt1',
    description='Mode:'
)
display(mode)
```

### Example 2: Parameter Slider

```python
from ipywidgets import FloatSlider

parameter = FloatSlider(
    value=1.0,
    min=0.0,
    max=10.0,
    step=0.1,
    description='Parameter:'
)
display(parameter)
```

### Example 3: Action Button

```python
from ipywidgets import Button

def on_button_click(button):
    print("Button clicked!")

button = Button(description='Click Me')
button.on_click(on_button_click)
display(button)
```

## Layout Examples

### Example 4: Standard Panel Layout

```python
from ipywidgets import VBox, HBox, HTML, Output, Layout

# Top Panel
top_panel = HBox([
    HTML("<h3>Top Panel</h3>")
], layout=Layout(padding='10px', border='1px solid #ccc'))

# Left Panel
left_panel = VBox([
    HTML("<h3>Left Panel</h3>")
], layout=Layout(width='300px', padding='10px', border='1px solid #ccc'))

# Center Panel
center_panel = VBox([
    HTML("<h3>Center Panel</h3>")
], layout=Layout(flex='1 1 auto', padding='10px', border='1px solid #ccc'))

# Right Panel
right_panel = VBox([
    HTML("<h3>Right Panel</h3>")
], layout=Layout(width='250px', padding='10px', border='1px solid #ccc'))

# Bottom Panel
bottom_panel = VBox([
    HTML("<h3>Bottom Panel</h3>")
], layout=Layout(padding='10px', border='1px solid #ccc'))

# Main Layout
main_layout = VBox([
    top_panel,
    HBox([left_panel, center_panel, right_panel]),
    bottom_panel
])

display(main_layout)
```

### Example 5: Configuration Accordion

```python
from ipywidgets import Accordion, VBox, FloatSlider

config1 = VBox([
    FloatSlider(value=1.0, min=0.0, max=10.0, description='Param 1:')
])

config2 = VBox([
    FloatSlider(value=2.0, min=0.0, max=10.0, description='Param 2:')
])

accordion = Accordion(children=[config1, config2])
accordion.set_title(0, 'Configuration 1')
accordion.set_title(1, 'Configuration 2')

display(accordion)
```

## Interactive Examples

### Example 6: Real-Time Updates

```python
from ipywidgets import FloatSlider, Output
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

slider = FloatSlider(value=1.0, min=0.0, max=10.0, description='Value:')
output = Output()

def update_plot(change):
    with output:
        clear_output(wait=True)
        value = change['new']
        plt.figure(figsize=(8, 6))
        plt.plot([0, value], [0, value])
        plt.title(f'Value: {value}')
        plt.show()

slider.observe(update_plot, names='value')

display(slider, output)
```

### Example 7: Button with Progress

```python
from ipywidgets import Button, IntProgress, HTML
import time

button = Button(description='Process')
progress = IntProgress(value=0, min=0, max=100)
status = HTML("Ready")

def process(button):
    status.value = "Processing..."
    for i in range(100):
        time.sleep(0.01)
        progress.value = i + 1
    status.value = "✅ Complete"

button.on_click(process)
display(button, progress, status)
```

## Advanced Examples

### Example 8: Mode-Dependent Configuration

```python
from ipywidgets import Dropdown, Accordion, VBox, FloatSlider

mode = Dropdown(
    options=[('Mode 1', 'mode1'), ('Mode 2', 'mode2')],
    value='mode1'
)

config1 = VBox([FloatSlider(description='Param 1:')])
config2 = VBox([FloatSlider(description='Param 2:')])

accordion = Accordion(children=[config1, config2])
accordion.set_title(0, 'Mode 1 Config')
accordion.set_title(1, 'Mode 2 Config')

def update_config(change):
    if change['new'] == 'mode1':
        accordion.selected_index = 0
    else:
        accordion.selected_index = 1

mode.observe(update_config, names='value')
display(mode, accordion)
```

### Example 9: Results Display

```python
from ipywidgets import Button, Output, HTML
from IPython.display import display, clear_output

button = Button(description='Generate Results')
results_output = Output()
summary = HTML("No results yet")

def generate_results(button):
    with results_output:
        clear_output(wait=True)
        # Generate and display results
        display(HTML("<h4>Results:</h4>"))
        display(HTML("<p>Result 1: Value 1</p>"))
        display(HTML("<p>Result 2: Value 2</p>"))
    summary.value = "✅ Results generated"

button.on_click(generate_results)
display(button, results_output, summary)
```

## Related Documentation

- **[Widget Patterns](widget-patterns.md)** - Common patterns
- **[Widget Specifications](widget-specifications.md)** - Detailed specs
- **[Widget System](README.md)** - System overview

---

**Last Updated**: 2024

