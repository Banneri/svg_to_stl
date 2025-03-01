import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import xml.etree.ElementTree as ET
from svg.path import parse_path, Move, Line, CubicBezier, QuadraticBezier, Arc, Close
import pyclipper
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely.affinity import affine_transform
import trimesh
import numpy as np
import os
import re
import time

# Patch trimesh for NumPy 1.23.3 compatibility
import trimesh.creation
original_triangulate = trimesh.creation.triangulate_polygon
def patched_triangulate_polygon(polygon, **kwargs):
    if 'dtype' in kwargs:
        del kwargs['dtype']
    vertices, faces = original_triangulate(polygon, **kwargs)
    vertices = np.array(vertices, dtype=np.float64)
    return vertices, faces
trimesh.creation.triangulate_polygon = patched_triangulate_polygon

SCALE = 10000  # High quality for precise output

class StampApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SVG to 3D Stamp Converter (High Quality)")
        self.svg_file = None
        self.polygons = []

        # GUI Elements
        self.import_button = tk.Button(root, text="Import SVG/XML", command=self.import_file)
        self.import_button.pack(pady=5)

        self.height_label = tk.Label(root, text="Extrusion Height (mm):")
        self.height_label.pack()
        self.height_entry = tk.Entry(root)
        self.height_entry.insert(0, "5.0")
        self.height_entry.pack()

        self.base_label = tk.Label(root, text="Base Thickness (mm):")
        self.base_label.pack()
        self.base_entry = tk.Entry(root)
        self.base_entry.insert(0, "2.0")
        self.base_entry.pack()

        self.export_button = tk.Button(root, text="Export STL", command=self.export_stl)
        self.export_button.pack(pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(root, length=300, mode='determinate')
        self.progress.pack(pady=5)
        self.progress_label = tk.Label(root, text="Progress: 0%")
        self.progress_label.pack(pady=5)

        self.status_label = tk.Label(root, text="")
        self.status_label.pack(pady=5)

    def update_progress(self, value, max_value, label_text, force_update=False):
        """Update the progress bar and label with finer granularity."""
        if max_value > 0:
            progress = (value / max_value) * 100
            self.progress['value'] = progress
            self.progress_label.config(text=f"Progress: {progress:.1f}% - {label_text}")
            if force_update or value % 5 == 0:  # Update every 5% or force update for key steps
                self.root.update_idletasks()
            time.sleep(0.01)  # Small delay to ensure UI updates are visible

    def parse_transform(self, transform_str):
        """Parse SVG transform attribute into a 6-element affine transform matrix."""
        if not transform_str:
            return [1, 0, 0, 1, 0, 0]  # Identity matrix
        matrix = [1, 0, 0, 1, 0, 0]  # [a, b, c, d, e, f] for affine_transform
        transforms = re.findall(r'(translate|scale|rotate|matrix)\(([^)]+)\)', transform_str)
        for transform_type, args in transforms:
            args = [float(x) for x in re.split(r'[,\s]+', args.strip())]
            if transform_type == 'translate':
                tx, ty = args if len(args) == 2 else (args[0], 0)
                matrix[4] += tx
                matrix[5] += ty
            elif transform_type == 'scale':
                sx, sy = args if len(args) == 2 else (args[0], args[0])
                matrix[0] *= sx
                matrix[3] *= sy
            elif transform_type == 'rotate':
                angle = np.radians(args[0])
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                if len(args) == 3:  # Rotate around point (cx, cy)
                    cx, cy = args[1], args[2]
                    matrix[4] += cx - cx * cos_a + cy * sin_a
                    matrix[5] += cy - cx * sin_a - cy * cos_a
                new_matrix = [cos_a, sin_a, -sin_a, cos_a, 0, 0]
                matrix = [matrix[0] * new_matrix[0] + matrix[2] * new_matrix[1],
                          matrix[1] * new_matrix[0] + matrix[3] * new_matrix[1],
                          matrix[0] * new_matrix[2] + matrix[2] * new_matrix[3],
                          matrix[1] * new_matrix[2] + matrix[3] * new_matrix[3],
                          matrix[4], matrix[5]]
            elif transform_type == 'matrix':
                matrix = args  # Direct matrix: [a, b, c, d, e, f]
        return matrix

    def sample_path_segment(self, segment, num_points=50):
        """Sample points along a path segment for higher resolution."""
        if isinstance(segment, (CubicBezier, QuadraticBezier, Arc)):
            points = [segment.point(t / num_points) for t in range(num_points + 1)]
            return [(p.real, p.imag) for p in points]
        return [(segment.end.real, segment.end.imag)]

    def element_to_polygon(self, elem, transform_matrix):
        """Convert an SVG element to a shapely Polygon with transformations, higher quality."""
        polygons = []
        
        # Check if filled
        fill = elem.get('fill', 'black')  # Default to black if unspecified
        style = elem.get('style', '')
        if style:
            style_dict = dict(item.split(':') for item in style.split(';') if ':' in item)
            fill = style_dict.get('fill', fill).strip()
        if fill in ['none', 'transparent', '']:
            return polygons  # Skip unfilled shapes

        # Apply transformation
        def apply_transform(points):
            return [affine_transform(Point(x, y), transform_matrix) for x, y in points]

        # Handle different SVG elements
        tag = elem.tag.split('}')[-1]  # Remove namespace
        if tag == 'path':
            d = elem.get('d')
            if not d:
                return polygons
            svg_path = parse_path(d)

            subpaths = []
            current_subpath = []
            for segment in svg_path:
                if isinstance(segment, Move):
                    if current_subpath:
                        subpaths.append(current_subpath)
                    current_subpath = [segment.end]
                elif isinstance(segment, (Line, CubicBezier, QuadraticBezier, Arc)):
                    points = self.sample_path_segment(segment, num_points=50)
                    current_subpath.extend(points)
                elif isinstance(segment, Close):
                    if current_subpath and len(current_subpath) > 1:
                        current_subpath.append(current_subpath[0])
                else:
                    print(f"Skipping unrecognized segment: {type(segment).__name__}")
            if current_subpath:
                subpaths.append(current_subpath)

            if subpaths:
                # Collect all points for pyclipper, ensuring closed paths
                pc = pyclipper.Pyclipper()
                for subpath in subpaths:
                    points = [(p[0], p[1]) for p in subpath if isinstance(p, tuple) and len(p) == 2]
                    if len(points) >= 3:
                        # Ensure the path is closed
                        if points[0] != points[-1]:
                            points.append(points[0])
                        scaled_poly = [(int(p[0] * SCALE), int(p[1] * SCALE)) for p in points]
                        pc.AddPath(scaled_poly, pyclipper.PT_SUBJECT, True)
                solution = pc.Execute2(pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD)

                def traverse_polytree(node):
                    if not node.IsHole:
                        exterior = [(p[0] / SCALE, p[1] / SCALE) for p in node.Contour]
                        holes = []
                        for child in node.Childs:
                            if child.IsHole:
                                hole = [(p[0] / SCALE, p[1] / SCALE) for p in child.Contour]
                                if len(hole) >= 3:
                                    holes.append(hole)
                            traverse_polytree(child)
                        exterior = apply_transform(exterior)
                        holes = [apply_transform(h) for h in holes]
                        exterior_coords = [(p.x, -p.y) for p in exterior]  # Flip y-axis
                        hole_coords = [[(p.x, -p.y) for p in h] for h in holes]
                        if len(exterior_coords) >= 3:
                            # Ensure the exterior is closed and valid
                            if exterior_coords[0] != exterior_coords[-1]:
                                exterior_coords.append(exterior_coords[0])
                            poly = Polygon(exterior_coords, hole_coords)
                            if poly.is_valid:
                                polygons.append(poly)
                                print(f"Added polygon with exterior: {exterior_coords[:5]}... and {len(holes)} holes")

                for node in solution.Childs:
                    traverse_polytree(node)

        elif tag == 'rect':
            x, y = float(elem.get('x', 0)), float(elem.get('y', 0))
            w, h = float(elem.get('width', 0)), float(elem.get('height', 0))
            if w > 0 and h > 0:
                points = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
                exterior = apply_transform(points)
                exterior_coords = [(p.x, -p.y) for p in exterior]
                poly = Polygon(exterior_coords)
                if poly.is_valid:
                    polygons.append(poly)

        elif tag == 'circle':
            cx, cy = float(elem.get('cx', 0)), float(elem.get('cy', 0))
            r = float(elem.get('r', 0))
            if r > 0:
                circle = Point(cx, cy).buffer(r, resolution=32)  # High quality
                circle = affine_transform(circle, transform_matrix)
                exterior_coords = [(x, -y) for x, y in circle.exterior.coords[:-1]]  # Flip y-axis
                poly = Polygon(exterior_coords)
                if poly.is_valid:
                    polygons.append(poly)

        elif tag == 'ellipse':
            cx, cy = float(elem.get('cx', 0)), float(elem.get('cy', 0))
            rx, ry = float(elem.get('rx', 0)), float(elem.get('ry', 0))
            if rx > 0 and ry > 0:
                t = np.linspace(0, 2 * np.pi, 64)  # High quality
                points = [(cx + rx * np.cos(ti), cy + ry * np.sin(ti)) for ti in t]
                exterior = apply_transform(points)
                exterior_coords = [(p.x, -p.y) for p in exterior]
                poly = Polygon(exterior_coords)
                if poly.is_valid:
                    polygons.append(poly)

        elif tag == 'polygon':
            points_str = elem.get('points', '')
            points = [float(x) for x in re.split(r'[,\s]+', points_str.strip()) if x]
            if len(points) >= 6:  # At least 3 points (x, y pairs)
                point_pairs = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
                exterior = apply_transform(point_pairs)
                exterior_coords = [(p.x, -p.y) for p in exterior]
                poly = Polygon(exterior_coords)
                if poly.is_valid:
                    polygons.append(poly)

        return polygons

    def import_file(self):
        """Import an SVG/XML file and parse it."""
        self.svg_file = filedialog.askopenfilename(filetypes=[("SVG/XML files", "*.svg *.xml")])
        if self.svg_file:
            self.parse_file()
        else:
            messagebox.showerror("Error", "No file selected")

    def parse_file(self):
        self.polygons = []
        try:
            tree = ET.parse(self.svg_file)
            root = tree.getroot()
            ns = {'svg': 'http://www.w3.org/2000/svg'}

            def process_element(elem, parent_transform=[1, 0, 0, 1, 0, 0]):
                transform = elem.get('transform', '')
                current_transform = self.parse_transform(transform)
                combined_transform = [
                    parent_transform[0] * current_transform[0] + parent_transform[2] * current_transform[1],
                    parent_transform[1] * current_transform[0] + parent_transform[3] * current_transform[1],
                    parent_transform[0] * current_transform[2] + parent_transform[2] * current_transform[3],
                    parent_transform[1] * current_transform[2] + parent_transform[3] * current_transform[3],
                    parent_transform[4] + current_transform[4],
                    parent_transform[5] + current_transform[5]
                ]

                if elem.tag.endswith('path') or elem.tag.endswith('rect') or \
                   elem.tag.endswith('circle') or elem.tag.endswith('ellipse') or \
                   elem.tag.endswith('polygon'):
                    polys = self.element_to_polygon(elem, combined_transform)
                    self.polygons.extend(polys)
                    print(f"Added {len(polys)} polygons, total: {len(self.polygons)}")
                    for i, poly in enumerate(polys):
                        print(f"Polygon {i}: Exterior={poly.exterior.coords[:5]}..., Holes={len(poly.interiors)}")

                for child in elem:
                    process_element(child, combined_transform)

            process_element(root)

            if not self.polygons:
                self.status_label.config(text="No filled shapes found")
            else:
                self.status_label.config(text=f"File imported: {len(self.polygons)} shapes found")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse file: {str(e)}")

    def export_stl(self):
        if not self.polygons:
            messagebox.showerror("Error", "No shapes to export")
            return

        try:
            height = float(self.height_entry.get())
            base_thickness = float(self.base_entry.get())
            if height <= 0 or base_thickness <= 0:
                raise ValueError("Height and base thickness must be positive")
        except ValueError:
            messagebox.showerror("Error", "Invalid input: Enter positive numbers")
            return

        self.status_label.config(text="Processing...")
        self.progress['value'] = 0
        self.progress_label.config(text="Progress: 0%")

        try:
            available_engines = trimesh.boolean._engines.keys()
            print(f"Available engines: {list(available_engines)}")
            engine = 'manifold'

            # Calculate total steps for finer granularity
            total_polygons = len(self.polygons)
            total_steps = 3 + total_polygons + (total_polygons * 2)  # Base (2 steps), extrude (1 per poly), union (2 per poly for batching)
            current_step = 0

            # Step 1: Create base polygon
            print("Creating base polygon...")
            self.update_progress(current_step, total_steps, "Creating base polygon", force_update=True)
            all_poly = unary_union(self.polygons)
            if all_poly.is_empty or not all_poly.is_valid:
                print("Warning: Union resulted in empty or invalid geometry, using individual valid polygons")
                valid_polygons = [p for p in self.polygons if p.is_valid]
                if valid_polygons:
                    all_poly = unary_union(valid_polygons)
                else:
                    raise ValueError("No valid polygons to create base from")
            minx, miny, maxx, maxy = all_poly.bounds
            padding = 10
            base_poly = Polygon([
                (minx - padding, miny - padding),
                (maxx + padding, miny - padding),
                (maxx + padding, maxy + padding),
                (minx - padding, maxy + padding)
            ])
            print(f"Base polygon bounds: min({minx}, {miny}), max({maxx}, {maxy})")
            current_step += 1

            # Step 2: Extrude base mesh
            print("Extruding base mesh...")
            self.update_progress(current_step, total_steps, "Extruding base mesh", force_update=True)
            if not base_poly.is_valid:
                raise ValueError("Base polygon is invalid")
            base_mesh = trimesh.creation.extrude_polygon(base_poly, height=base_thickness)
            print("Base mesh created:", base_mesh)
            current_step += 1

            # Step 3: Extrude individual polygons
            print("Extruding individual polygons...")
            extruded_meshes = []
            for i, poly in enumerate(self.polygons):
                print(f"Extruding polygon {i+1}/{len(self.polygons)}...")
                self.update_progress(current_step, total_steps, f"Extruding polygon {i+1}/{len(self.polygons)}", force_update=True)
                if poly.is_valid:
                    mesh = trimesh.creation.extrude_polygon(poly, height=height)
                    mesh.apply_translation([0, 0, base_thickness])
                    extruded_meshes.append(mesh)
                else:
                    print(f"Invalid polygon skipped at index {i}")
                current_step += 1

            # Step 4: Combine meshes (batch union with finer progress)
            print("Combining meshes...")
            combined = base_mesh
            batch_size = 50
            total_batches = (len(extruded_meshes) + batch_size - 1) // batch_size  # Ceiling division
            batch_step_increment = 2  # Two steps per batch (start and end of union)

            for batch_idx in range(0, len(extruded_meshes), batch_size):
                batch = extruded_meshes[batch_idx:batch_idx + batch_size]
                batch_size_actual = len(batch)
                print(f"Processing batch {batch_idx//batch_size + 1}/{total_batches} ({batch_size_actual} meshes)...")
                
                # Start of batch
                self.update_progress(current_step, total_steps, f"Starting batch {batch_idx//batch_size + 1}/{total_batches}", force_update=True)
                
                if batch:
                    try:
                        # Update progress for each mesh in the batch
                        for mesh_idx, mesh in enumerate(batch):
                            self.update_progress(current_step + (mesh_idx / batch_size_actual) * batch_step_increment,
                                               total_steps,
                                               f"Unioning mesh {mesh_idx + 1}/{batch_size_actual} in batch {batch_idx//batch_size + 1}",
                                               force_update=True)
                            combined = trimesh.boolean.union([combined, mesh], engine=engine)
                    except Exception as e:
                        print(f"Union failed for batch {batch_idx//batch_size + 1}: {e}")
                        combined = trimesh.util.concatenate([combined] + batch)  # Fallback
                current_step += batch_step_increment

            # Step 5: Export STL
            print("Exporting STL...")
            self.update_progress(current_step, total_steps, "Exporting STL", force_update=True)
            stl_file = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
            if stl_file:
                combined.export(stl_file)
                self.status_label.config(text="STL exported successfully")
            else:
                self.status_label.config(text="Export cancelled")
            self.progress['value'] = 100
            self.progress_label.config(text="Progress: 100% - Complete")

        except Exception as e:
            self.progress['value'] = 0
            self.progress_label.config(text="Progress: 0% - Error")
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    root = tk.Tk()
    app = StampApp(root)
    root.mainloop()