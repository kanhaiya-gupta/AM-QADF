"""
Core Domain Entities

Domain entities for the AM-QADF framework.
These are the fundamental building blocks of the domain model.
"""

import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class VoxelData:
    """
    Data stored in a single voxel.

    This is a core domain entity representing the data contained within
    a single voxel in the 3D voxel grid.
    """

    signals: Dict[str, Union[float, List[float]]] = field(default_factory=dict)
    count: int = 0  # Number of data points contributing to this voxel

    def add_signal(self, signal_name: str, value: float, aggregation: str = "mean"):
        """
        Add a signal value to this voxel.

        Args:
            signal_name: Name of the signal (e.g., 'power', 'velocity', 'energy')
            value: Signal value to add
            aggregation: How to aggregate multiple values ('mean', 'max', 'min', 'sum')
        """
        if signal_name not in self.signals:
            self.signals[signal_name] = []

        current_value = self.signals[signal_name]
        if isinstance(current_value, list):
            current_value.append(value)
        else:
            # Convert existing value to list
            self.signals[signal_name] = [float(current_value), value]

        self.count += 1

    def finalize(self, aggregation: str = "mean"):
        """
        Finalize voxel data by aggregating multiple values.

        Args:
            aggregation: Aggregation method ('mean', 'max', 'min', 'sum')
        """
        for signal_name in list(self.signals.keys()):
            values = self.signals[signal_name]
            if isinstance(values, list):
                if aggregation == "mean":
                    self.signals[signal_name] = float(np.mean(values))
                elif aggregation == "max":
                    self.signals[signal_name] = float(np.max(values))
                elif aggregation == "min":
                    self.signals[signal_name] = float(np.min(values))
                elif aggregation == "sum":
                    self.signals[signal_name] = float(np.sum(values))
                else:
                    self.signals[signal_name] = float(np.mean(values))  # Default to mean


# Signal type definitions (if needed as entities)
# Currently signals are represented as strings, but we can add a Signal class if needed
