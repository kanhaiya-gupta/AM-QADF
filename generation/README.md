# Data Generation Module

This directory contains all data generation logic for the PBF-LB/M process chain. **Data generation is kept external to the core framework** to maintain clear separation between framework code and data generation logic.

## 🎯 Purpose

In **production**, data comes from real sensors and experiments. In **demo/testing**, this module generates realistic synthetic data that mimics real sensor outputs and process data.

## 📁 Directory Structure

```
data_generation/
├── sensors/              # Sensor data generators
│   ├── ispm_thermal_generator.py   # ISPM_Thermal (thermal monitoring) data
│   ├── ispm_optical_generator.py   # ISPM_Optical (photodiodes, cameras) data
│   ├── ispm_acoustic_generator.py # ISPM_Acoustic (acoustic emissions) data
│   ├── ispm_strain_generator.py    # ISPM_Strain (strain gauges) data
│   ├── ispm_plume_generator.py     # ISPM_Plume (vapor plume) data
│   ├── ct_scan_generator.py        # Computed Tomography scan data
│   └── laser_parameter_generator.py # Laser monitoring data (LBD - Laser Beam Diagnostics)
│
├── process/             # Process data generators
│   ├── stl_processor.py            # STL file processing
│   ├── hatching_generator.py       # Hatching path generation (pyslm)
│   └── build_simulator.py          # Build process simulation
│
└── scripts/             # Orchestration and utility scripts
    ├── generate_all_data.py        # Generate all data types
    ├── generate_for_demo.py        # Demo-specific generation
    ├── check_mongodb.py            # Check MongoDB connection and database status
    ├── start_mongodb.py            # Start MongoDB container
    ├── stop_mongodb.py             # Stop MongoDB container
    └── mongodb_status.py            # Check MongoDB container status
```

python generation/scripts/populate_mongodb.py --stl-files frameGuide.stl --collections hatching_layers stl_models --delete-existing

## 🔄 Data Flow

### Production Flow:
```
Real Sensors/Experiments → Phase 12 (NoSQL) → Framework
```

### Demo/Testing Flow:
```
data_generation/ → Phase 12 (NoSQL) → Framework
```

## 📊 Data Types Generated

### Sensor Data (`sensors/`)
- **ISPM (In-Situ Process Monitoring) Data**: Multiple ISPM sensor types for comprehensive process monitoring
  - **ISPM_Thermal**: Melt pool temperature, size, thermal gradients, cooling rates, process events
  - **ISPM_Optical**: Photodiode signals, melt pool imaging, spatter detection, keyhole detection
  - **ISPM_Acoustic**: Acoustic emission signals, frequency spectra, event detection (spatter, defects, anomalies)
  - **ISPM_Strain**: Strain measurements, deformation/displacement, residual stress, warping detection
  - **ISPM_Plume**: Vapor plume characteristics, geometry, composition, dynamics, contamination detection
- **CT Scan Data**: Voxel grids, density maps, porosity maps, defect locations
- **Laser Monitoring Data (LBD)**: Laser Beam Diagnostics - Process parameters (setpoints/commanded values) and temporal sensor measurements:
  - **Process Parameters**: Commanded power, scan speed, hatch spacing, energy density, exposure time, region type
  - **Temporal Power Sensors**: Actual power, power error, power stability, power fluctuations
  - **Beam Temporal Characteristics**: Pulse frequency, pulse duration, pulse energy, duty cycle, beam modulation
  - **Laser System Health**: Laser temperature, cooling system, power supply, diode metrics, operating hours, pulse count

### Process Data (`process/`)
- **STL Models**: Processed STL files with metadata (bounding box, volume, complexity)
- **Hatching Paths**: Layer-by-layer scan paths generated using pyslm
- **Build Simulation**: Simulated build process with layer progression

**Note**: Data fusion (combining laser parameters, CT scans, and ISPM data) is handled by the **framework** (Phase 5: Data Fusion), not by data generation. Data generation only produces the raw data sources.

## 📁 STL Models Directory

STL files for hatching generation should be placed in `data_generation/models/`.

The `STLProcessor` automatically looks for STL files in this directory:

```python
from data_generation.process.stl_processor import STLProcessor

processor = STLProcessor()
stl_files = processor.find_stl_files()  # Find all STL files
stl_file = processor.get_stl_file("frameGuide.stl")  # Get specific file
```

## 🚀 Usage

### Generate All Data
```python
from data_generation.scripts.generate_all_data import generate_all_data

# Generate data for 10 STL files (from models directory)
generate_all_data(n_models=10, output_dir="generated_data/")

# Generate data for specific STL files
generate_all_data(stl_files=["frameGuide.stl", "cube.stl"], output_dir="generated_data/")

# Generate data for all available STL files
generate_all_data(output_dir="generated_data/")
```

### Generate Specific Data Types
```python
from data_generation.sensors.ispm_generator import ISPMGenerator
from data_generation.sensors.ct_scan_generator import CTScanGenerator

# Generate ISPM data
ispm_gen = ISPMGenerator()
ispm_data = ispm_gen.generate_for_build(build_id="build_001", n_layers=100)

# Generate CT scan data
ct_gen = CTScanGenerator()
ct_data = ct_gen.generate_for_build(build_id="build_001")
```

### For Demo Notebooks
```python
from data_generation.scripts.generate_for_demo import generate_demo_data

# Generate demo data
demo_data = generate_demo_data(n_samples=1000)
```

### MongoDB Management

Manage the MongoDB container for Phase 12 (NoSQL Data Warehouse):

```bash
# Start MongoDB container
python data_generation/scripts/start_mongodb.py

# Check MongoDB container status
python data_generation/scripts/mongodb_status.py

# Check MongoDB connection and database
python data_generation/scripts/check_mongodb.py

# Stop MongoDB container
python data_generation/scripts/stop_mongodb.py
```

### Direct MongoDB Population

**For demo/testing**: Populate MongoDB directly from data generation (bypasses ingestion layer):

```bash
# Populate all collections for all STL files
python generation/scripts/populate_mongodb.py

# Populate specific number of models
python generation/scripts/populate_mongodb.py --n-models 5

# Populate specific STL files
python generation/scripts/populate_mongodb.py --stl-files frameGuide.stl cube.stl

# Populate only specific collections
python generation/scripts/populate_mongodb.py --collections stl_models laser_monitoring_data ispm_thermal_monitoring_data ispm_optical_monitoring_data ispm_acoustic_monitoring_data ispm_strain_monitoring_data ispm_plume_monitoring_data
```

**Note**: This direct population is for **demo/testing**. In production, data should go through the Phase 12 ingestion layer for validation and transformation.

## 🔗 Integration with Framework

### Phase 12: Data Warehouse
Generated data is ingested by Phase 12's data warehouse:

```python
from src.data_pipeline.data_warehouse.ingestion import ingest_generated_data

# Ingest generated data into NoSQL
ingest_generated_data(generated_data_dir="generated_data/")
```

### Phase 5: Data Fusion
The framework's data fusion module (Phase 5) combines the generated data sources:

```python
from src.data_pipeline.data_warehouse_clients.synchronization.data_fusion import DataFusion

# Framework handles fusion - not data generation's responsibility
fusion = DataFusion()
fused_data = fusion.fuse_signals(...)  # Combines laser, CT, ISPM data
```

## 📝 Notes

- **Separation of Concerns**: Framework code (`src/`) never generates data, only consumes it
- **Production Ready**: Framework works with real data without modification
- **Testable**: Easy to swap real data for generated data in tests
- **Maintainable**: Data generation logic isolated and versioned separately

## 🎯 Key Principles

1. **Framework Independence**: Core framework doesn't depend on data generation
2. **Realistic Data**: Generated data mimics real sensor outputs and process characteristics
3. **Configurable**: All generators accept configuration for different scenarios
4. **Reproducible**: Uses random seeds for reproducible data generation
5. **Extensible**: Easy to add new data generators or modify existing ones

