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
        stl_client: Optional[Any] = None
    ) -> str:
        """
        Export PyVista plotter to interactive HTML.
        
        Args:
            plotter: PyVista Plotter object
            grid_data: Grid data dictionary for metadata
            fallback_to_threejs: Whether to fallback to Three.js HTML if PyVista export fails
            voxelized_mesh: Optional voxelized mesh for Three.js fallback
            stl_client: Optional STL client for mesh recreation
            
        Returns:
            HTML content as string
        """
        try:
            # PyVista's export_html requires pythreejs or panel
            # Let's try it first, but if it fails, use a simpler approach
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                temp_path = f.name
            
            try:
                # Try PyVista's export_html (requires pythreejs or panel)
                # Note: Some versions don't support 'title' parameter
                try:
                    plotter.export_html(
                        temp_path,
                        title=f"Voxel Grid Visualization - {grid_data.get('grid_id', 'Unknown')}"
                    )
                except TypeError:
                    # Fallback for versions that don't support title parameter
                    plotter.export_html(temp_path)
                
                # Read and verify HTML content
                with open(temp_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Clean up temp file
                Path(temp_path).unlink()
                
                # Verify HTML is not empty
                if html_content and len(html_content.strip()) > 100:
                    logger.info("Successfully exported PyVista interactive HTML")
                    return html_content
                else:
                    raise ValueError("export_html returned empty or invalid HTML")
                
            except Exception as export_error:
                logger.warning(f"PyVista export_html failed: {export_error}, using mesh-based interactive viewer")
                
                # Fallback: Create interactive viewer from mesh data
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
            # Try mesh-based interactive viewer first
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
        bbox_min = metadata.get('bbox_min', [0, 0, 0])
        bbox_max = metadata.get('bbox_max', [100, 100, 100])
        
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
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Voxel Grid Visualization - {grid_id}</title>
            <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: #f5f5f5;
                    font-family: Arial, sans-serif;
                    overflow: hidden;
                }}
                #container {{
                    width: 100%;
                    height: 100vh;
                }}
                #info {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(255, 255, 255, 0.9);
                    color: #333;
                    padding: 10px;
                    border-radius: 5px;
                    z-index: 100;
                    font-size: 12px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                #info h3 {{
                    margin: 0 0 5px 0;
                    color: #4A90E2;
                }}
            </style>
        </head>
        <body>
            <div id="container"></div>
            <div id="info">
                <h3>Voxel Grid: {grid_id}</h3>
                <p>Cells: {mesh.n_cells:,}</p>
                <p><small>Use mouse to rotate, scroll to zoom, right-click to pan</small></p>
            </div>
            <script>
                // Scene setup
                const scene = new THREE.Scene();
                scene.background = new THREE.Color(0xf5f5f5);
                
                const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 10000);
                const renderer = new THREE.WebGLRenderer({{ antialias: true }});
                renderer.setSize(window.innerWidth, window.innerHeight);
                renderer.setClearColor(0xf5f5f5, 1);
                document.getElementById('container').appendChild(renderer.domElement);
                
                // Camera position
                const center = {json.dumps(center)};
                const maxDim = {max_dim};
                const distance = maxDim * 2.5;
                camera.position.set(center[0] + distance, center[1] + distance, center[2] + distance);
                camera.lookAt(center[0], center[1], center[2]);
                
                // Lights
                const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
                scene.add(ambientLight);
                const directionalLight1 = new THREE.DirectionalLight(0xffffff, 1.0);
                directionalLight1.position.set(1, 1, 1);
                scene.add(directionalLight1);
                const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.6);
                directionalLight2.position.set(-1, -1, -1);
                scene.add(directionalLight2);
                
                // Load mesh data
                const vertices = {vertices_js};
                const faces = {faces_js};
                
                // Create geometry
                const geometry = new THREE.BufferGeometry();
                const positions = [];
                const indices = [];
                
                // Add vertices
                for (let i = 0; i < vertices.length; i++) {{
                    positions.push(vertices[i][0], vertices[i][1], vertices[i][2]);
                }}
                
                // Add faces
                for (let i = 0; i < faces.length; i++) {{
                    indices.push(faces[i][0], faces[i][1], faces[i][2]);
                }}
                
                geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
                geometry.setIndex(indices);
                geometry.computeVertexNormals();
                
                // Create material
                const material = new THREE.MeshStandardMaterial({{
                    color: 0x4A90E2,
                    transparent: true,
                    opacity: 0.8,
                    side: THREE.DoubleSide,
                    metalness: 0.3,
                    roughness: 0.4
                }});
                
                // Create mesh
                const mesh = new THREE.Mesh(geometry, material);
                scene.add(mesh);
                
                // Add wireframe
                const wireframe = new THREE.WireframeGeometry(geometry);
                const wireframeLine = new THREE.LineSegments(wireframe, new THREE.LineBasicMaterial({{ color: 0x000000, opacity: 0.2, transparent: true }}));
                scene.add(wireframeLine);
                
                // Controls
                const controls = new THREE.OrbitControls(camera, renderer.domElement);
                controls.target.set(center[0], center[1], center[2]);
                controls.update();
                
                // Animation loop
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                animate();
                
                // Handle resize
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
