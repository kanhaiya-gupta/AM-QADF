# Widget System Documentation

**Version**: 1.0  
**Last Updated**: 2024

## Overview

The AM-QADF Interactive Notebooks use a comprehensive widget system based on `ipywidgets` to provide interactive, no-code interfaces for exploring framework capabilities.

## Widget System Architecture

### Core Principle

**All notebooks use interactive widget-based interfaces.** This is mandatory for all notebooks.

### Widget Library

- **ipywidgets**: Primary widget library
- **matplotlib**: Visualization widgets
- **numpy/pandas**: Data widgets
- **Custom widgets**: Framework-specific widgets

## Standard Widget Layout

All notebooks follow a consistent 5-panel layout:

```
┌─────────────────────────────────────────────────────────┐
│  Top Panel: Mode Selectors, Action Buttons             │
├──────────┬──────────────────────────────┬──────────────┤
│          │                              │              │
│  Left    │                              │  Right       │
│  Panel:  │    Center Panel:             │  Panel:      │
│  Config  │    Visualizations            │  Results     │
│  Accordion│    (Multiple Views)         │  Display     │
│          │                              │              │
├──────────┴──────────────────────────────┴──────────────┤
│  Bottom Panel: Status, Progress, Logs                   │
└─────────────────────────────────────────────────────────┘
```

## Widget Types

### Basic Widgets

- **Dropdown**: Mode/method selection
- **RadioButtons**: Single choice selection
- **Checkbox**: Boolean toggles
- **Button**: Action triggers
- **Slider**: Parameter adjustment (IntSlider, FloatSlider)
- **Text**: Text input
- **Textarea**: Multi-line text input

### Container Widgets

- **VBox**: Vertical layout
- **HBox**: Horizontal layout
- **Accordion**: Collapsible sections
- **Tab**: Multiple views
- **Box**: Flexible layout

### Display Widgets

- **Output**: Display area
- **HTML**: HTML content
- **Markdown**: Markdown content
- **Label**: Text labels

### Advanced Widgets

- **SelectMultiple**: Multi-select dropdown
- **IntText/FloatText**: Numeric input
- **Progress**: Progress bars
- **Output**: Interactive output areas

## Widget Patterns

### Pattern 1: Mode Selection

```python
mode = Dropdown(
    options=[('Mode 1', 'mode1'), ('Mode 2', 'mode2')],
    value='mode1',
    description='Mode:'
)
```

### Pattern 2: Parameter Control

```python
parameter = FloatSlider(
    value=1.0,
    min=0.0,
    max=10.0,
    step=0.1,
    description='Parameter:'
)
```

### Pattern 3: Action Button

```python
execute_button = Button(
    description='Execute',
    button_style='success',
    icon='play'
)
```

### Pattern 4: Configuration Accordion

```python
config_accordion = Accordion(children=[
    config_section1,
    config_section2
])
config_accordion.set_title(0, 'Section 1')
config_accordion.set_title(1, 'Section 2')
```

## Widget Specifications

See [Widget Specifications](widget-specifications.md) for detailed specifications for each notebook.

## Widget Best Practices

### Layout

- Use consistent panel structure
- Organize widgets logically
- Group related widgets
- Use accordions for complex configurations

### Interaction

- Provide immediate feedback
- Update visualizations in real-time
- Show progress for long operations
- Display clear error messages

### Design

- Use descriptive labels
- Provide tooltips where helpful
- Use appropriate widget types
- Maintain consistent styling

## Widget Examples

See [Widget Examples](widget-examples.md) for code examples and patterns.

## Related Documentation

- **[Widget Specifications](widget-specifications.md)** - Detailed specs
- **[Widget Patterns](widget-patterns.md)** - Common patterns
- **[Widget Examples](widget-examples.md)** - Code examples
- **[Notebook Structure](../02-structure.md)** - Notebook organization

---

**Last Updated**: 2024

