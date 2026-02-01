"""
ParaView Launcher - Launch ParaView with .vdb files

Utilities to launch ParaView programmatically and create Jupyter notebook widgets
for interactive ParaView launching.
"""

import subprocess
import platform
import logging
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def find_paraview_executable() -> Optional[str]:
    """
    Find ParaView executable on the system.
    
    Returns:
        Path to ParaView executable, or None if not found
    
    Checks common installation paths:
    - Windows: Program Files, AppData
    - Linux: /usr/bin, /usr/local/bin, ~/.local/bin
    - macOS: /Applications
    """
    system = platform.system()
    
    if system == "Windows":
        # Common Windows paths
        possible_paths = [
            r"C:\Program Files\ParaView 5.11.2\bin\paraview.exe",
            r"C:\Program Files\ParaView 5.10.1\bin\paraview.exe",
            r"C:\Program Files\ParaView 5.9.1\bin\paraview.exe",
            r"C:\Program Files (x86)\ParaView 5.11.2\bin\paraview.exe",
            r"C:\Program Files (x86)\ParaView 5.10.1\bin\paraview.exe",
            r"%LOCALAPPDATA%\Programs\ParaView\bin\paraview.exe",
            r"%APPDATA%\ParaView\bin\paraview.exe",
        ]
        # Also check PATH
        possible_paths.append("paraview.exe")
        
    elif system == "Linux":
        # Common Linux paths
        possible_paths = [
            "/usr/bin/paraview",
            "/usr/local/bin/paraview",
            "~/.local/bin/paraview",
            "/opt/paraview/bin/paraview",
        ]
        # Also check PATH
        possible_paths.append("paraview")
        
    elif system == "Darwin":  # macOS
        # Common macOS paths
        possible_paths = [
            "/Applications/ParaView-5.11.2.app/Contents/MacOS/paraview",
            "/Applications/ParaView-5.10.1.app/Contents/MacOS/paraview",
            "/Applications/ParaView-5.9.1.app/Contents/MacOS/paraview",
            "/Applications/ParaView.app/Contents/MacOS/paraview",
            "/usr/local/bin/paraview",
            "~/.local/bin/paraview",
        ]
        # Also check PATH
        possible_paths.append("paraview")
    else:
        logger.warning(f"Unknown system: {system}, trying generic 'paraview'")
        possible_paths = ["paraview"]
    
    # Expand user and environment variables
    import os
    for path in possible_paths:
        expanded = os.path.expanduser(os.path.expandvars(path))
        if Path(expanded).exists():
            return expanded
        
        # Try to find in PATH
        if path == "paraview" or path == "paraview.exe":
            try:
                result = subprocess.run(
                    ["which", path] if system != "Windows" else ["where", path],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    found_path = result.stdout.strip().split('\n')[0]
                    if Path(found_path).exists():
                        return found_path
            except Exception:
                pass
    
    return None


def launch_paraview(vdb_file: str, paraview_path: Optional[str] = None) -> bool:
    """
    Launch ParaView with a .vdb file.
    
    Args:
        vdb_file: Path to .vdb file to open
        paraview_path: Optional path to ParaView executable (auto-detected if None)
    
    Returns:
        True if ParaView was launched successfully, False otherwise
    
    Example:
        >>> from am_qadf.visualization import launch_paraview
        >>> launch_paraview("output.vdb")
        True
    """
    vdb_path = Path(vdb_file)
    if not vdb_path.exists():
        logger.error(f"VDB file not found: {vdb_file}")
        return False
    
    # Find ParaView executable
    if paraview_path is None:
        paraview_path = find_paraview_executable()
    
    if paraview_path is None:
        logger.error(
            "ParaView executable not found. "
            "Please install ParaView or specify the path manually."
        )
        return False
    
    paraview_exe = Path(paraview_path)
    if not paraview_exe.exists():
        logger.error(f"ParaView executable not found: {paraview_path}")
        return False
    
    # Launch ParaView with the .vdb file
    try:
        # Use absolute paths
        vdb_abs = str(vdb_path.resolve())
        exe_abs = str(paraview_exe.resolve())
        
        logger.info(f"Launching ParaView: {exe_abs} {vdb_abs}")
        
        # Launch in background (non-blocking)
        if platform.system() == "Windows":
            # Windows: use CREATE_NEW_PROCESS_GROUP to detach
            subprocess.Popen(
                [exe_abs, vdb_abs],
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            # Linux/macOS: use setsid to detach
            subprocess.Popen(
                [exe_abs, vdb_abs],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        
        logger.info(f"ParaView launched successfully with {vdb_abs}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to launch ParaView: {e}", exc_info=True)
        return False


def create_paraview_button(vdb_file: str, button_text: str = "Open in ParaView") -> Optional[Any]:
    """
    Create a Jupyter notebook button to launch ParaView.
    
    Args:
        vdb_file: Path to .vdb file to open
        button_text: Text to display on button
    
    Returns:
        ipywidgets.Button if ipywidgets is available, None otherwise
    
    Example:
        >>> from am_qadf.visualization import create_paraview_button
        >>> button = create_paraview_button("output.vdb")
        >>> display(button)  # In Jupyter notebook
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        
        button = widgets.Button(
            description=button_text,
            button_style='info',
            tooltip=f'Launch ParaView with {vdb_file}',
            icon='external-link',
        )
        
        def on_button_clicked(b):
            success = launch_paraview(vdb_file)
            if success:
                print(f"✅ ParaView launched with {vdb_file}")
            else:
                print(f"❌ Failed to launch ParaView. Please check if ParaView is installed.")
        
        button.on_click(on_button_clicked)
        return button
        
    except ImportError:
        logger.warning(
            "ipywidgets not available. Install with: pip install ipywidgets"
        )
        return None


def export_and_launch_paraview(
    voxel_grid: Any,
    output_path: str,
    signal_names: Optional[List[str]] = None,
    auto_launch: bool = True,
) -> str:
    """
    Export voxel grid to ParaView and optionally launch ParaView.
    
    Convenience function that combines export and launch.
    
    Args:
        voxel_grid: VoxelGrid or AdaptiveResolutionGrid instance
        output_path: Path to output .vdb file
        signal_names: Optional list of signal names to export
        auto_launch: Whether to automatically launch ParaView after export
    
    Returns:
        Path to created .vdb file
    
    Example:
        >>> from am_qadf.visualization import export_and_launch_paraview
        >>> export_and_launch_paraview(voxel_grid, "output.vdb")
        'output.vdb'
    """
    from .paraview_exporter import export_voxel_grid_to_paraview
    
    # Export to .vdb
    vdb_path = export_voxel_grid_to_paraview(
        voxel_grid,
        output_path,
        signal_names=signal_names,
    )
    
    # Launch ParaView if requested
    if auto_launch:
        launch_paraview(vdb_path)
    
    return vdb_path
