"""
PyVista HTML Exporter

Reusable HTML export utilities for PyVista visualizations.
Exports PyVista Plotter objects to interactive HTML or wraps images in HTML.

Can be used by:
- Web clients (FastAPI, Flask, etc.)
- Jupyter notebooks
- CLI tools
- Other modules
"""
from __future__ import annotations

import logging
import json
import base64
import io
from typing import Dict, Any, Optional
from pathlib import Path
import tempfile

try:
    import pyvista as pv
    import numpy as np
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None
    np = None

logger = logging.getLogger(__name__)


class PyVistaHTMLExporter:
    """
    Export PyVista plotters to interactive HTML.
    
    Provides fallback mechanisms:
    1. Try PyVista's native export_html()
    2. Fallback to Three.js-based HTML from mesh data
    3. Fallback to static image wrapper
    """
    
    def export_plotter_to_html(
        self,
        plotter: pv.Plotter,
        grid_data: Dict[str, Any],
        fallback_to_threejs: bool = True,
        voxelized_mesh: Optional[Any] = None,
        stl_client: Optional[Any] = None,
        prefer_threejs_for_surface: bool = False,
    ) -> str:
        """
        Export PyVista plotter to interactive HTML.
        
        Args:
            plotter: PyVista Plotter object
            grid_data: Grid data dictionary for metadata
            fallback_to_threejs: Whether to fallback to Three.js HTML if PyVista export fails
            voxelized_mesh: Optional mesh (surface or voxelized) for Three.js viewer
            stl_client: Optional STL client for mesh recreation
            prefer_threejs_for_surface: If True, skip PyVista export_html and use Three.js
                immediately so the result is always interactive (orbit/pan/zoom).
                Use this for STL surface view on servers where PyVista export_html
                may fail or produce static content.
            
        Returns:
            HTML content as string
        """
        # Prefer Three.js path for surface (e.g. STL): always interactive, no trame required
        if prefer_threejs_for_surface and voxelized_mesh is not None:
            logger.info("Using Three.js interactive viewer for surface (prefer_threejs_for_surface=True)")
            return self.create_threejs_viewer_from_mesh(
                mesh=voxelized_mesh,
                grid_data=grid_data,
                stl_client=stl_client,
            )
        try:
            # PyVista's export_html requires trame; may fail or be static on headless servers
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                temp_path = f.name
            
            try:
                try:
                    plotter.export_html(
                        temp_path,
                        title=f"Voxel Grid Visualization - {grid_data.get('grid_id', 'Unknown')}"
                    )
                except TypeError:
                    plotter.export_html(temp_path)
                
                with open(temp_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                Path(temp_path).unlink()
                
                if html_content and len(html_content.strip()) > 100:
                    logger.info("Successfully exported PyVista interactive HTML")
                    return html_content
                else:
                    raise ValueError("export_html returned empty or invalid HTML")
                
            except Exception as export_error:
                logger.warning(f"PyVista export_html failed: {export_error}, using mesh-based interactive viewer")
                
                if fallback_to_threejs:
                    return self.create_threejs_viewer_from_mesh(
                        mesh=voxelized_mesh,
                        grid_data=grid_data,
                        stl_client=stl_client
                    )
                else:
                    raise export_error
            
        except Exception as e:
            logger.error(f"HTML export failed: {e}", exc_info=True)
            if fallback_to_threejs:
                try:
                    return self.create_threejs_viewer_from_mesh(
                        mesh=voxelized_mesh,
                        grid_data=grid_data,
                        stl_client=stl_client
                    )
                except Exception as mesh_error:
                    logger.warning(f"Mesh-based viewer failed: {mesh_error}, falling back to static image")
                    # Final fallback: static image
                    try:
                        # Export to image first
                        image_data = self._export_plotter_to_base64(plotter)
                        return self.wrap_image_in_html(image_data, grid_data)
                    except Exception as img_error:
                        logger.error(f"Image export also failed: {img_error}")
                        return f"<div style='padding: 20px; color: red;'>Error generating visualization: {str(e)}</div>"
            else:
                raise
    
    def create_threejs_viewer_from_mesh(
        self,
        mesh: Optional[Any],
        grid_data: Dict[str, Any],
        stl_client: Optional[Any] = None
    ) -> str:
        """
        Create Three.js HTML from PyVista mesh.
        
        Args:
            mesh: PyVista PolyData mesh (if None, will try to recreate from STL)
            grid_data: Grid data dictionary for metadata
            stl_client: Optional STL client for loading STL file if mesh is None
            
        Returns:
            HTML content as string
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for mesh-based HTML generation")
        
        # Try to get mesh from parameter first
        if mesh is None:
            # Try to recreate it from STL if we have the client
            if stl_client:
                try:
                    metadata = grid_data.get("metadata", {})
                    model_id = grid_data.get("model_id") or metadata.get("model_id")
                    voxel_size = metadata.get("voxel_size")
                    
                    if model_id and voxel_size:
                        stl_path = stl_client.load_stl_file(model_id)
                        if stl_path and stl_path.exists():
                            logger.info(f"Recreating voxelized mesh from STL: {stl_path}")
                            stl_mesh = pv.read(str(stl_path))
                            if voxel_size > 0:
                                # Check for non-uniform spacing
                                spacing_x = metadata.get("voxel_size_x", voxel_size)
                                spacing_y = metadata.get("voxel_size_y", voxel_size)
                                spacing_z = metadata.get("voxel_size_z", voxel_size)
                                
                                if spacing_x != spacing_y or spacing_y != spacing_z:
                                    spacing_tuple = (spacing_x, spacing_y, spacing_z)
                                    logger.info(f"Using non-uniform spacing: {spacing_tuple}")
                                    mesh = stl_mesh.voxelize(spacing=spacing_tuple)
                                else:
                                    mesh = stl_mesh.voxelize(spacing=voxel_size)
                            else:
                                mesh = stl_mesh.voxelize()
                            logger.info(f"Recreated voxelized mesh: {mesh.n_cells} cells")
                except Exception as e:
                    logger.warning(f"Failed to recreate mesh: {e}")
        
        if mesh is None:
            # Final fallback: return error message
            return "<div style='padding: 20px; color: red;'>Error: No mesh data available for visualization. Please ensure the grid was created with STL-based voxelization.</div>"
        
        # Extract mesh data
        vertices = mesh.points
        # Handle different face formats (PyVista can have different face array formats)
        try:
            if len(mesh.faces) > 0 and mesh.faces[0] == 3:
                # Standard format: [3, v1, v2, v3, 3, v4, v5, v6, ...]
                faces = mesh.faces.reshape(-1, 4)[:, 1:4]
            else:
                # Alternative format or already flattened
                faces = mesh.faces.reshape(-1, 3)
        except Exception:
            # Fallback: try to get faces directly
            try:
                faces = mesh.faces.reshape(-1, 3)
            except Exception:
                # If all else fails, create faces from cells
                faces = []
                for i in range(mesh.n_cells):
                    cell = mesh.get_cell(i)
                    if cell.n_points >= 3:
                        faces.append([cell.point_ids[0], cell.point_ids[1], cell.point_ids[2]])
                faces = np.array(faces)
        
        # Convert to JavaScript arrays
        vertices_js = json.dumps(vertices.tolist())
        faces_js = json.dumps(faces.tolist())
        
        grid_id = grid_data.get('grid_id', 'Unknown')
        metadata = grid_data.get('metadata', {})
        bbox_min = metadata.get('bbox_min')
        bbox_max = metadata.get('bbox_max')
        if bbox_min is None or bbox_max is None:
            # Derive from mesh bounds (e.g. for STL surface)
            b = getattr(mesh, 'bounds', None)
            if b is not None and len(b) >= 6:
                bbox_min = [float(b[0]), float(b[2]), float(b[4])]
                bbox_max = [float(b[1]), float(b[3]), float(b[5])]
            else:
                bbox_min = bbox_min or [0, 0, 0]
                bbox_max = bbox_max or [100, 100, 100]
        bbox_min = list(bbox_min)
        bbox_max = list(bbox_max)
        
        # Calculate center and size
        center = [
            (bbox_min[0] + bbox_max[0]) / 2,
            (bbox_min[1] + bbox_max[1]) / 2,
            (bbox_min[2] + bbox_max[2]) / 2
        ]
        size = [
            bbox_max[0] - bbox_min[0],
            bbox_max[1] - bbox_min[1],
            bbox_max[2] - bbox_min[2]
        ]
        max_dim = max(size)
        page_title = "STL Surface" if (isinstance(grid_id, str) and grid_id.startswith("stl_")) else "3D View"
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{page_title}</title>
            <style>
                body {{ margin: 0; padding: 0; background: #f5f5f5; font-family: Arial, sans-serif; overflow: hidden; }}
                #container {{ width: 100%; height: 100vh; }}
                #info {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); color: #333; padding: 10px; border-radius: 5px; z-index: 100; font-size: 12px; }}
                #info h3 {{ margin: 0 0 5px 0; color: #4A90E2; }}
            </style>
        </head>
        <body>
            <div id="container"></div>
            <div id="info">
                <h3>{page_title}</h3>
                <p>Cells: {mesh.n_cells:,}</p>
                <p><small>Drag to rotate, scroll to zoom, right-drag to pan</small></p>
            </div>
            <script type="importmap">
            {{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
            </script>
            <script type="module">
                import * as THREE from 'three';
                import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf5f5f5);
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.getElementById('container').appendChild(renderer.domElement);
                const center = {json.dumps(center)};
                const maxDim = {max_dim};
                const distance = maxDim * 2.5;
                camera.position.set(center[0] + distance, center[1] + distance, center[2] + distance);
                camera.lookAt(center[0], center[1], center[2]);
                scene.add(new THREE.AmbientLight(0xffffff, 0.65));
                const d1 = new THREE.DirectionalLight(0xffffff, 0.9);
                d1.position.set(1.2, 1.5, 1);
                scene.add(d1);
                const d2 = new THREE.DirectionalLight(0xffffff, 0.4);
                d2.position.set(-1, 0.5, -0.5);
                scene.add(d2);
                const d3 = new THREE.DirectionalLight(0xe8f4fc, 0.25);
                d3.position.set(0, -1, 0.5);
                scene.add(d3);
                const vertices = {vertices_js};
                const faces = {faces_js};
                const geometry = new THREE.BufferGeometry();
                const positions = [];
                const indices = [];
                for (let i = 0; i < vertices.length; i++) positions.push(vertices[i][0], vertices[i][1], vertices[i][2]);
                for (let i = 0; i < faces.length; i++) indices.push(faces[i][0], faces[i][1], faces[i][2]);
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setIndex(indices);
                geometry.computeVertexNormals();
                const material = new THREE.MeshStandardMaterial({{ color: 0xADD8E6, side: THREE.FrontSide, metalness: 0.12, roughness: 0.6, flatShading: true }});
                const meshObj = new THREE.Mesh(geometry, material);
                scene.add(meshObj);
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.target.set(center[0], center[1], center[2]);
                controls.update();
                function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); }}
                animate();
                window.addEventListener('resize', () => {{ camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix(); renderer.setSize(window.innerWidth, window.innerHeight); }});
            </script>
        </body>
        </html>
        """
    
    def create_threejs_viewer_from_hatching_arrays(
        self,
        positions: Any,
        vertex_colors_rgb: Any,
        scalar_bar_min: Optional[float] = None,
        scalar_bar_max: Optional[float] = None,
        scalar_name: Optional[str] = None,
        grid_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create Three.js HTML from C++ hatching arrays (no Python color loop).
        C++ provides positions (6*n) and per-vertex RGB (6*n). Only JSON-serialize and embed.
        """
        grid_data = grid_data or {}
        n6 = len(positions) if hasattr(positions, "__len__") else 0
        if n6 == 0 or n6 % 6 != 0:
            return "<div style='padding: 20px; color: #666;'>No paths to display.</div>"
        n = n6 // 6
        rgb = vertex_colors_rgb
        if rgb is None or (hasattr(rgb, "__len__") and len(rgb) < n6):
            return "<div style='padding: 20px; color: #666;'>Missing vertex colors from C++.</div>"
        pos_list = positions.tolist() if hasattr(positions, "tolist") else list(positions)
        rgb_list = rgb.tolist() if hasattr(rgb, "tolist") else list(rgb)
        x_min = y_min = z_min = float("inf")
        x_max = y_max = z_max = float("-inf")
        for i in range(0, n6, 3):
            x, y, z = float(pos_list[i]), float(pos_list[i + 1]), float(pos_list[i + 2])
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
            if z < z_min:
                z_min = z
            if z > z_max:
                z_max = z
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        max_dim = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
        grid_id = grid_data.get("grid_id", "hatching")
        title = "Hatching paths" if "hatch" in str(grid_id).lower() else "Line paths"
        show_bar = (
            scalar_name
            and scalar_bar_min is not None
            and scalar_bar_max is not None
            and scalar_bar_min == scalar_bar_min  # not NaN
            and scalar_bar_max == scalar_bar_max
        )
        if show_bar:
            v_min_fmt = f"{scalar_bar_min:.4g}" if abs(scalar_bar_min) < 1e4 and abs(scalar_bar_min) >= 1e-3 or scalar_bar_min == 0 else f"{scalar_bar_min:.2e}"
            v_max_fmt = f"{scalar_bar_max:.4g}" if abs(scalar_bar_max) < 1e4 and abs(scalar_bar_max) >= 1e-3 or scalar_bar_max == 0 else f"{scalar_bar_max:.2e}"
            scalar_label_esc = (scalar_name or "value").replace("<", "&lt;").replace(">", "&gt;")
            scalar_bar_html = f"""
            <div id="scalar-bar">
                <div class="scalar-bar-title">{scalar_label_esc}</div>
                <div class="scalar-bar-strip"></div>
                <div class="scalar-bar-labels">
                    <span class="scalar-bar-max">{v_max_fmt}</span>
                    <span class="scalar-bar-min">{v_min_fmt}</span>
                </div>
            </div>"""
        else:
            scalar_bar_html = ""
        positions_js = json.dumps(pos_list)
        colors_js = json.dumps(rgb_list)
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script type="importmap">
            {{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
            </script>
            <style>
                body {{ margin: 0; padding: 0; background: #eeeeee; overflow: hidden; }}
                #container {{ width: 100%; height: 100vh; }}
                #info {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; z-index: 100; font-size: 12px; }}
                #scalar-bar {{ position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.9); padding: 8px 10px; border-radius: 5px; z-index: 100; font-size: 11px; display: flex; flex-direction: column; align-items: center; }}
                #scalar-bar .scalar-bar-title {{ font-weight: 600; margin-bottom: 4px; color: #333; }}
                #scalar-bar .scalar-bar-strip {{ width: 14px; height: 120px; border-radius: 3px; background: linear-gradient(to top, #1f5994 0%, #7ac4ff 100%); }}
                #scalar-bar .scalar-bar-labels {{ display: flex; flex-direction: column; align-items: flex-end; margin-top: 4px; color: #555; }}
                #scalar-bar .scalar-bar-max {{ margin-bottom: 2px; }}
                #scalar-bar .scalar-bar-min {{ }}
            </style>
        </head>
        <body>
            <div id="container"></div>
            <div id="info">
                <h3>{title}</h3>
                <p>Paths: {n:,}</p>
                <p><small>Drag to rotate, scroll to zoom, right-drag to pan</small></p>
            </div>
            {scalar_bar_html}
            <script type="module">
                import * as THREE from 'three';
                import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xeeeeee);
                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1e7);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.getElementById('container').appendChild(renderer.domElement);
                const center = {json.dumps(center)};
                const maxDim = Math.max({max_dim}, 1e-6);
                const distance = Math.max(maxDim * 2.2, maxDim + 10);
                camera.position.set(center[0] + distance, center[1] + distance, center[2] + distance);
                camera.lookAt(center[0], center[1], center[2]);
                scene.add(new THREE.AmbientLight(0xffffff, 0.9));
                const d1 = new THREE.DirectionalLight(0xffffff, 1.0);
                d1.position.set(1, 1, 1);
                scene.add(d1);
                const positions = {positions_js};
                const vertexColorsRgb = {colors_js};
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(vertexColorsRgb, 3));
                const material = new THREE.LineBasicMaterial({{ vertexColors: true }});
                const lines = new THREE.LineSegments(geometry, material);
                scene.add(lines);
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.target.set(center[0], center[1], center[2]);
                controls.update();
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            </script>
        </body>
        </html>
        """

    def create_threejs_viewer_from_point_cloud(
        self,
        positions: Any,
        vertex_colors_rgb: Any,
        scalar_bar_min: Optional[float] = None,
        scalar_bar_max: Optional[float] = None,
        scalar_name: Optional[str] = None,
        grid_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create Three.js HTML from C++ point cloud arrays (laser_monitoring, ISPM).
        C++ provides positions (3*n) and per-point RGB (3*n). Only JSON-serialize and embed.
        """
        grid_data = grid_data or {}
        n3 = len(positions) if hasattr(positions, "__len__") else 0
        if n3 == 0 or n3 % 3 != 0:
            return "<div style='padding: 20px; color: #666;'>No points to display.</div>"
        n = n3 // 3
        rgb = vertex_colors_rgb
        if rgb is None or (hasattr(rgb, "__len__") and len(rgb) < n3):
            return "<div style='padding: 20px; color: #666;'>Missing vertex colors from C++.</div>"
        pos_list = positions.tolist() if hasattr(positions, "tolist") else list(positions)
        rgb_list = rgb.tolist() if hasattr(rgb, "tolist") else list(rgb)
        x_min = y_min = z_min = float("inf")
        x_max = y_max = z_max = float("-inf")
        for i in range(0, n3, 3):
            x, y, z = float(pos_list[i]), float(pos_list[i + 1]), float(pos_list[i + 2])
            if x < x_min: x_min = x
            if x > x_max: x_max = x
            if y < y_min: y_min = y
            if y > y_max: y_max = y
            if z < z_min: z_min = z
            if z > z_max: z_max = z
        center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
        max_dim = max(x_max - x_min, y_max - y_min, z_max - z_min, 1.0)
        grid_id = grid_data.get("grid_id", "point_cloud")
        title = grid_data.get("title", "Point cloud")
        show_bar = (
            scalar_name
            and scalar_bar_min is not None
            and scalar_bar_max is not None
            and scalar_bar_min == scalar_bar_min
            and scalar_bar_max == scalar_bar_max
        )
        if show_bar:
            v_min_fmt = f"{scalar_bar_min:.4g}" if abs(scalar_bar_min) < 1e4 and abs(scalar_bar_min) >= 1e-3 or scalar_bar_min == 0 else f"{scalar_bar_min:.2e}"
            v_max_fmt = f"{scalar_bar_max:.4g}" if abs(scalar_bar_max) < 1e4 and abs(scalar_bar_max) >= 1e-3 or scalar_bar_max == 0 else f"{scalar_bar_max:.2e}"
            scalar_label_esc = (scalar_name or "value").replace("<", "&lt;").replace(">", "&gt;")
            scalar_bar_html = f"""
            <div id="scalar-bar">
                <div class="scalar-bar-title">{scalar_label_esc}</div>
                <div class="scalar-bar-strip"></div>
                <div class="scalar-bar-labels">
                    <span class="scalar-bar-max">{v_max_fmt}</span>
                    <span class="scalar-bar-min">{v_min_fmt}</span>
                </div>
            </div>"""
        else:
            scalar_bar_html = ""
        positions_js = json.dumps(pos_list)
        colors_js = json.dumps(rgb_list)
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script type="importmap">
            {{"imports": {{"three": "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js", "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
            </script>
            <style>
                body {{ margin: 0; padding: 0; background: #eeeeee; overflow: hidden; }}
                #container {{ width: 100%; height: 100vh; }}
                #info {{ position: absolute; top: 10px; left: 10px; background: rgba(255,255,255,0.9); padding: 10px; border-radius: 5px; z-index: 100; font-size: 12px; }}
                #scalar-bar {{ position: absolute; top: 10px; right: 10px; background: rgba(255,255,255,0.9); padding: 8px 10px; border-radius: 5px; z-index: 100; font-size: 11px; display: flex; flex-direction: column; align-items: center; }}
                #scalar-bar .scalar-bar-title {{ font-weight: 600; margin-bottom: 4px; color: #333; }}
                #scalar-bar .scalar-bar-strip {{ width: 14px; height: 120px; border-radius: 3px; background: linear-gradient(to top, #1f5994 0%, #7ac4ff 100%); }}
                #scalar-bar .scalar-bar-labels {{ display: flex; flex-direction: column; align-items: flex-end; margin-top: 4px; color: #555; }}
            </style>
        </head>
        <body>
            <div id="container"></div>
            <div id="info">
                <h3>{title}</h3>
                <p>Points: {n:,}</p>
                <p><small>Drag to rotate, scroll to zoom, right-drag to pan</small></p>
            </div>
            {scalar_bar_html}
            <script type="module">
                import * as THREE from 'three';
                import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xeeeeee);
                const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 1e7);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                document.getElementById('container').appendChild(renderer.domElement);
                const center = {json.dumps(center)};
                const maxDim = Math.max({max_dim}, 1e-6);
                const distance = Math.max(maxDim * 2.2, maxDim + 10);
                camera.position.set(center[0] + distance, center[1] + distance, center[2] + distance);
                camera.lookAt(center[0], center[1], center[2]);
                scene.add(new THREE.AmbientLight(0xffffff, 0.9));
                const d1 = new THREE.DirectionalLight(0xffffff, 1.0);
                d1.position.set(1, 1, 1);
                scene.add(d1);
                const positions = {positions_js};
                const vertexColorsRgb = {colors_js};
                const geometry = new THREE.BufferGeometry();
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.Float32BufferAttribute(vertexColorsRgb, 3));
                const material = new THREE.PointsMaterial({{ size: 1.5, vertexColors: true, sizeAttenuation: true }});
                const points = new THREE.Points(geometry, material);
                scene.add(points);
                const controls = new OrbitControls(camera, renderer.domElement);
                controls.target.set(center[0], center[1], center[2]);
                controls.update();
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
                window.addEventListener('resize', () => {{
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);
                }});
            </script>
        </body>
        </html>
        """

    def wrap_image_in_html(
        self,
        image_data: str,
        grid_data: Dict[str, Any]
    ) -> str:
        """
        Wrap base64-encoded image in HTML.
        
        Args:
            image_data: Base64-encoded image data
            grid_data: Grid data dictionary for metadata
            
        Returns:
            HTML content as string
        """
        grid_id = grid_data.get('grid_id', 'Unknown')
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Voxel Grid Visualization - {grid_id}</title>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    background: #1a1a2e;
                    color: #fff;
                    font-family: Arial, sans-serif;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    text-align: center;
                }}
                h1 {{
                    color: #8b5cf6;
                    margin-bottom: 20px;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 2px solid #8b5cf6;
                    border-radius: 8px;
                    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.3);
                }}
                .info {{
                    margin-top: 20px;
                    padding: 15px;
                    background: rgba(139, 92, 246, 0.1);
                    border-radius: 8px;
                    text-align: left;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Voxel Grid Visualization</h1>
                <p>Grid ID: {grid_id}</p>
                <img src="data:image/png;base64,{image_data}" alt="Voxel Grid Visualization">
                <div class="info">
                    <p><strong>Note:</strong> This is a static image. For interactive visualization, 
                    ensure PyVista's interactive HTML export is available.</p>
                </div>
            </div>
        </body>
        </html>
        """
    
    def _export_plotter_to_base64(
        self,
        plotter: pv.Plotter,
        width: int = 1200,
        height: int = 800
    ) -> str:
        """
        Export plotter to base64-encoded PNG.
        
        Args:
            plotter: PyVista Plotter object
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Base64-encoded image data as string
        """
        plotter.window_size = (width, height)
        screenshot = plotter.screenshot()
        
        # Convert to base64
        buffer = io.BytesIO()
        try:
            from PIL import Image
            img = Image.fromarray(screenshot)
            img.save(buffer, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        except ImportError:
            # Fallback: use imageio if PIL not available
            import imageio
            buffer = io.BytesIO()
            imageio.imwrite(buffer, screenshot, format='PNG')
            image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        plotter.close()
        return image_data
