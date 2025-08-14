import math
from django.core.management.base import BaseCommand
from django.contrib.gis.geos import Point
from DataExtractor.models import GISFeature

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# Try to import matplotlib for graph visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyBboxPatch
    import numpy as np # Added for graph visualization
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Graph visualization will be disabled.")

# Configuration constants
MIN_FPOI_POINTS_PER_RING = 7  # Minimum number of FPOI points required to form a ring
MAX_FPOI_POINTS_PER_RING = 10  # Maximum number of FPOI points allowed per ring


def get_angle_deg(center, point):
    dx = point.x - center.x
    dy = point.y - center.y
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360


def points_in_sector(center, points, radius, sector_start, sector_angle):
    sector_end = (sector_start + sector_angle) % 360

    def in_sector(point):
        angle = get_angle_deg(center, point.geometry)
        if sector_start < sector_end:
            in_ang = sector_start <= angle <= sector_end
        else:
            in_ang = angle >= sector_start or angle <= sector_end
        dist = center.distance(point.geometry)
        return in_ang and dist <= radius

    return [pt for pt in points if in_sector(pt)]


def create_ring_graphs(center, rings, max_radius, all_points, block_name):
    """Create matplotlib graphs showing the potential rings and their coverage."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping graph visualization.")
        return
    
    print(f"Creating graph visualization...")
    print(f"Center: {center.x:.2f}, {center.y:.2f}")
    print(f"Max radius: {max_radius}")
    print(f"Total FPOI points: {len(all_points)}")
    print(f"Number of rings: {len(rings)}")
    
    # Debug: Check if we have valid coordinates
    if len(all_points) > 0:
        first_point = all_points[0]
        print(f"First FPOI point: {first_point.name} at ({first_point.geometry.x:.2f}, {first_point.geometry.y:.2f})")
    
    # Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'OLT-FPOI Ring Analysis: {block_name}', fontsize=16)
    
    # Plot 1: Ring Coverage Map
    ax1.set_title('Ring Coverage Map')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    
    # Plot the OLT center
    ax1.plot(center.x, center.y, 'ro', markersize=10, label='OLT Center')
    
    # Plot all FPOI points with better visibility
    if len(all_points) > 0:
        fpoi_x = [pt.geometry.x for pt in all_points]
        fpoi_y = [pt.geometry.y for pt in all_points]
        
        # Calculate plot bounds
        min_x, max_x = min(fpoi_x), max(fpoi_x)
        min_y, max_y = min(fpoi_y), max(fpoi_y)
        
        # Add some padding
        x_padding = (max_x - min_x) * 0.1 if max_x > min_x else 100
        y_padding = (max_y - min_y) * 0.1 if max_y > min_y else 100
        
        ax1.set_xlim(min_x - x_padding, max_x + x_padding)
        ax1.set_ylim(min_y - y_padding, max_y + y_padding)
        
        # Plot FPOI points with larger markers and better colors
        ax1.scatter(fpoi_x, fpoi_y, c='blue', s=100, alpha=0.8, label=f'FPOI Points ({len(all_points)})')
        
        # Add labels for some FPOI points (first 5)
        for i, pt in enumerate(all_points[:5]):
            ax1.annotate(pt.name, (pt.geometry.x, pt.geometry.y), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot each ring with different colors
    if len(rings) > 0:
        colors = plt.cm.Set3(np.linspace(0, 1, len(rings)))
        for i, ring in enumerate(rings):
            color = colors[i]
            
            # Get ring points
            ring_points = ring['points']
            if len(ring_points) > 0:
                ring_x = [pt.geometry.x for pt in ring_points]
                ring_y = [pt.geometry.y for pt in ring_points]
                
                # Add OLT center to start and end of ring
                ring_x = [center.x] + ring_x + [center.x]
                ring_y = [center.y] + ring_y + [center.y]
                
                # Plot ring path
                ax1.plot(ring_x, ring_y, 'o-', color=color, linewidth=3, markersize=8, 
                        label=f'Ring {i+1} ({len(ring_points)} FPOIs)')
                
                # Highlight ring points with larger markers
                ax1.scatter([pt.geometry.x for pt in ring_points], 
                           [pt.geometry.y for pt in ring_points], 
                           c=[color], s=150, alpha=0.9, edgecolors='black', linewidth=1)
                
                # Add ring labels
                for j, pt in enumerate(ring_points):
                    ax1.annotate(f'R{i+1}-{j+1}', (pt.geometry.x, pt.geometry.y), 
                                xytext=(3, 3), textcoords='offset points', fontsize=7, 
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
    
    # Add max radius circle
    circle = patches.Circle((center.x, center.y), max_radius, fill=False, 
                           linestyle='--', color='gray', alpha=0.7, linewidth=2, 
                           label=f'Max Radius ({max_radius:.1f})')
    ax1.add_patch(circle)
    
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Sector Coverage
    ax2.set_title('Sector Coverage Analysis')
    ax2.set_xlabel('Angle (degrees)')
    ax2.set_ylabel('Number of FPOI Points')
    
    # Create angle bins for analysis
    angle_bins = np.arange(0, 361, 30)
    fpoi_counts = []
    
    for i in range(len(angle_bins) - 1):
        start_angle = angle_bins[i]
        end_angle = angle_bins[i + 1]
        count = len(points_in_sector(center, all_points, max_radius, start_angle, end_angle - start_angle))
        fpoi_counts.append(count)
    
    # Plot sector bars
    bars = ax2.bar(angle_bins[:-1], fpoi_counts, width=30, alpha=0.7, 
                   color='skyblue', edgecolor='navy')
    
    # Highlight sectors that formed rings
    for ring in rings:
        start_angle = ring['start_angle']
        sector_angle = ring['sector_angle']
        
        # Find which bins this ring covers
        for i, bin_start in enumerate(angle_bins[:-1]):
            bin_end = bin_start + 30
            if (start_angle < bin_end and start_angle + sector_angle > bin_start):
                bars[i].set_color('lightgreen')
                bars[i].set_alpha(0.8)
    
    ax2.set_xticks(angle_bins)
    ax2.set_xticklabels([f'{int(angle)}°' for angle in angle_bins])
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, max(fpoi_counts) * 1.1 if fpoi_counts else 10)
    
    # Add text annotations
    for i, (angle, count) in enumerate(zip(angle_bins[:-1], fpoi_counts)):
        ax2.text(angle + 15, count + 0.1, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save the plot
    filename = f'ring_analysis_{block_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Graph visualization saved as: {filename}")
    
    # Show the plot
    plt.show()


def validate_data_availability():
    """Validate that OLT and FPOI data exists in the database."""
    olt_count = GISFeature.objects.filter(source_file='olt').count()
    fpoi_count = GISFeature.objects.filter(source_file='fpoi').count()
    
    return olt_count, fpoi_count


def visualize_sectors(center, rings, max_radius, all_points):
    """Create a visual representation of the sectors and rings."""
    print("\n" + "="*60)
    print("OLT-FPOI RING VISUALIZATION")
    print("="*60)
    
    # Create a simple ASCII visualization
    print(f"OLT Center: {center.x:.2f}, {center.y:.2f}")
    print(f"Max Radius: {max_radius:.2f}")
    print(f"Total FPOI Points Available: {len(all_points)}")
    print()
    
    # Show each ring with its sector information
    for i, ring in enumerate(rings, 1):
        start_angle = ring['start_angle']
        sector_angle = ring['sector_angle']
        end_angle = (start_angle + sector_angle) % 360
        points_count = len(ring['points'])
        path_length = ring['path_length']
        
        print(f"Ring {i}:")
        print(f"  Sector: {start_angle:>6.1f}° to {end_angle:>6.1f}° (width: {sector_angle:>5.1f}°)")
        merged_info = f" (merged {ring.get('merged_sectors', 1)} sectors)" if ring.get('merged_sectors', 1) > 1 else ""
        print(f"  FPOI Points: {points_count:>3}{merged_info} | Path Length: {path_length:>8.2f}")
        print(f"  Route: OLT → {', '.join([pt.name for pt in ring['points'][:3]])}{'...' if len(ring['points']) > 3 else ''} → OLT")
        
        # Show points in this sector
        sector_points = ring['points']
        if sector_points:
            print("  FPOI Points in sector:")
            for j, point in enumerate(sector_points[:5]):  # Show first 5 points
                angle = get_angle_deg(center, point.geometry)
                distance = center.distance(point.geometry)
                print(f"    {j+1:2d}. {point.name:<20} | Angle: {angle:>6.1f}° | Dist: {distance:>8.2f}")
            if len(sector_points) > 5:
                print(f"    ... and {len(sector_points) - 5} more FPOI points")
        print()
    
    # Create a circular sector diagram
    print("Circular Sector Diagram:")
    print("(N = North, E = East, S = South, W = West)")
    print("OLT at center, FPOI points in sectors")
    print()
    
    # Simple ASCII circle representation
    print("        N (0°)")
    print("           |")
    print("    NW     |     NE")
    print("           |")
    print("W (270°) --+-- E (90°)")
    print("           |")
    print("    SW     |     SE")
    print("           |")
    print("        S (180°)")
    print()
    
    # Show sector coverage
    print("Sector Coverage:")
    covered_angles = []
    for ring in rings:
        start = ring['start_angle']
        end = (start + ring['sector_angle']) % 360
        covered_angles.append((start, end))
    
    # Sort and merge overlapping sectors
    covered_angles.sort(key=lambda x: x[0])
    merged = []
    for start, end in covered_angles:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    
    # Display coverage
    for start, end in merged:
        if start < end:
            print(f"  {start:>6.1f}° to {end:>6.1f}° (covered by FPOI rings)")
        else:
            print(f"  {start:>6.1f}° to 360.0° + 0.0° to {end:>6.1f}° (covered by FPOI rings)")
    
    # Find uncovered areas
    uncovered = []
    current_angle = 0
    for start, end in merged:
        if start > current_angle:
            uncovered.append((current_angle, start))
        current_angle = max(current_angle, end)
    
    if current_angle < 360:
        uncovered.append((current_angle, 360))
    
    for start, end in uncovered:
        print(f"  {start:>6.1f}° to {end:>6.1f}° (no FPOI coverage)")
    
    print("="*60)


def create_distance_matrix(points):
    """Create a distance matrix for OR-Tools TSP solver."""
    size = len(points)
    dist_matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            if i == j:
                row.append(0)
            else:
                d = points[i].geometry.distance(points[j].geometry)
                row.append(d)
        dist_matrix.append(row)
    return dist_matrix


def solve_tsp(points, center_point=None):
    """Solve TSP using OR-Tools and return ordered points and path length."""
    if len(points) <= 2:
        return points, 0

    dist_matrix = create_distance_matrix(points)
    size = len(dist_matrix)

    # Create the routing index manager.
    manager = pywrapcp.RoutingIndexManager(size, 1, 0)

    # Create Routing Model.
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * 1000)  # scaled to int for OR-Tools

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    # Solve the problem.
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        path_length = 0
        prev_index = None
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route.append(points[node_index])
            if prev_index is not None:
                path_length += dist_matrix[manager.IndexToNode(prev_index)][node_index]
            prev_index = index
            index = solution.Value(routing.NextVar(index))
        # Add distance back to start point to close the loop
        path_length += dist_matrix[manager.IndexToNode(prev_index)][manager.IndexToNode(routing.Start(0))]
        return route, path_length
    else:
        return points, float('inf')


class Command(BaseCommand):
    help = "Generate rings around an OLT (Optical Line Terminal) using FPOI (Fiber Point of Interest) points with angular sweep and TSP optimization"

    def add_arguments(self, parser):
        parser.add_argument(
            '--block_name', type=str, required=True, help="Name of the OLT block center (must be of type 'olt')")
        parser.add_argument(
            '--max_radius', type=float, required=True, help="Max radius for ring in projection units")
        parser.add_argument(
            '--base_sector_angle', type=float, default=30, help="Starting sector angle in degrees")
        parser.add_argument(
            '--sector_increment', type=float, default=15, help="Sector angle increment when expanding")
        parser.add_argument(
            '--min_fpoi_points', type=int, default=MIN_FPOI_POINTS_PER_RING, help=f"Minimum FPOI points required per ring (default: {MIN_FPOI_POINTS_PER_RING})")
        parser.add_argument(
            '--max_fpoi_points', type=int, default=MAX_FPOI_POINTS_PER_RING, help=f"Maximum FPOI points allowed per ring (default: {MAX_FPOI_POINTS_PER_RING})")
        parser.add_argument(
            '--visualize', action='store_true', help="Enable visualization of sectors and rings")
        parser.add_argument(
            '--graphs', action='store_true', help="Generate matplotlib graphs of ring analysis (requires matplotlib)")

    def handle(self, *args, **options):
        block_name = options['block_name']
        max_radius = options['max_radius']
        base_sector_angle = options['base_sector_angle']
        sector_increment = options['sector_increment']
        min_fpoi_points = options['min_fpoi_points']
        max_fpoi_points = options['max_fpoi_points']
        visualize = options['visualize']
        generate_graphs = options['graphs']

        # Validate data availability first
        olt_count, fpoi_count = validate_data_availability()
        
        if olt_count == 0:
            self.stderr.write("No OLT data found in the database. Please import OLT data first using the import_kmz command.")
            return
            
        if fpoi_count == 0:
            self.stderr.write("No FPOI data found in the database. Please import FPOI data first using the import_kmz command.")
            return

        self.stdout.write(f"Data validation: Found {olt_count} OLT points and {fpoi_count} FPOI points in database.")
        self.stdout.write(f"Ring formation requires {min_fpoi_points}-{max_fpoi_points} FPOI points per sector.")
        self.stdout.write(f"Sectors will be merged if insufficient points are found.")

        # Find the OLT block center
        block = GISFeature.objects.filter(name__iexact=block_name, source_file='olt').first()
        if not block:
            self.stderr.write(f"OLT block '{block_name}' not found. Make sure the block is of type 'olt'.")
            self.stdout.write(f"Available OLT names: {list(GISFeature.objects.filter(source_file='olt').values_list('name', flat=True))}")
            return

        center = block.geometry.centroid
        
        # Only consider FPOI points for the rings
        all_points = GISFeature.objects.filter(source_file='fpoi').exclude(id=block.id)
        
        if not all_points.exists():
            self.stderr.write("No FPOI points found in the database. Make sure you have imported FPOI data.")
            return

        self.stdout.write(f"Found {all_points.count()} FPOI points to consider for ring formation.")
        self.stdout.write(f"Using OLT '{block_name}' as ring center at coordinates: {center.x:.2f}, {center.y:.2f}")

        # Debug: Show FPOI point distribution by angle
        fpoi_angles = []
        for point in all_points:
            angle = get_angle_deg(center, point.geometry)
            fpoi_angles.append((angle, point.name))
        
        fpoi_angles.sort(key=lambda x: x[0])
        self.stdout.write(f"\nFPOI Point Distribution by Angle:")
        for angle, name in fpoi_angles:
            self.stdout.write(f"  {angle:>6.1f}°: {name}")
        
        # Debug: Check coordinate ranges
        if len(all_points) > 0:
            x_coords = [pt.geometry.x for pt in all_points]
            y_coords = [pt.geometry.y for pt in all_points]
            self.stdout.write(f"\nCoordinate Ranges:")
            self.stdout.write(f"  X: {min(x_coords):.6f} to {max(x_coords):.6f}")
            self.stdout.write(f"  Y: {min(y_coords):.6f} to {max(y_coords):.6f}")
            self.stdout.write(f"  Center: ({center.x:.6f}, {center.y:.6f})")
            
            # Check if coordinates are reasonable (not all the same)
            if len(set(x_coords)) == 1 and len(set(y_coords)) == 1:
                self.stderr.write("WARNING: All FPOI points have identical coordinates!")
            elif max(x_coords) - min(x_coords) < 0.001 or max(y_coords) - min(y_coords) < 0.001:
                self.stderr.write("WARNING: FPOI points are very close together - coordinate system issue?")
        
        self.stdout.write(f"\nStarting ring generation...")
        self.stdout.write(f"Available FPOI points: {len(all_points)}")
        self.stdout.write(f"Used FPOI points so far: 0")

        rings = []
        baseline_angle = 0
        skipped_sectors = 0
        used_fpoi_points = set()  # Track which FPOI points have already been used

        while baseline_angle < 360:
            sector_angle = base_sector_angle
            ring_formed = False
            merged_sectors = 1  # Track how many sectors we've merged
            best_sector_angle = None
            best_sector_points = None

            # Try to form a ring by expanding sectors until we have enough points or hit max limit
            while sector_angle <= 120:  # Allow up to 120° sectors (4 base sectors of 30°)
                # Get all points in sector, excluding already used ones
                all_sector_points = points_in_sector(center, all_points, max_radius, baseline_angle, sector_angle)
                sector_points = [pt for pt in all_sector_points if pt.id not in used_fpoi_points]
                
                # Debug info for first few attempts
                if merged_sectors <= 3:  # Only show debug for first few sector expansions
                    self.stdout.write(f"  Sector {baseline_angle:>6.1f}° to {(baseline_angle + sector_angle):>6.1f}°: {len(all_sector_points)} total FPOIs, {len(sector_points)} available FPOIs")
                
                if len(sector_points) >= min_fpoi_points:
                    # We have enough points, now check if we're within max limit
                    if len(sector_points) <= max_fpoi_points:
                        # Perfect! We have a good number of points
                        best_path, path_length = solve_tsp(sector_points, center)
                        
                        # Simple acceptance condition: path length < max_radius * 4 (tune this)
                        if path_length < max_radius * 4:
                            ring_formed = True
                            rings.append({
                                'start_angle': baseline_angle,
                                'sector_angle': sector_angle,
                                'points': best_path,
                                'path_length': path_length,
                                'merged_sectors': merged_sectors,
                            })
                            # Mark these FPOI points as used
                            for point in best_path:
                                used_fpoi_points.add(point.id)
                            break
                    else:
                        # Too many points, but we might be able to use this sector with fewer points
                        # Store this as a potential candidate
                        if best_sector_points is None or len(sector_points) < len(best_sector_points):
                            best_sector_angle = sector_angle
                            best_sector_points = sector_points
                
                # If we haven't formed a ring yet, expand the sector
                sector_angle += sector_increment
                merged_sectors += 1
                
                # Safety check to prevent infinite loops
                if sector_angle > 120:
                    break

            # If we didn't form a ring but found a sector with too many points, try to optimize it
            if not ring_formed and best_sector_points is not None:
                # Try to find the optimal sector size that gives us close to max_ont_points
                optimal_sector_angle = best_sector_angle
                while optimal_sector_angle > base_sector_angle:
                    optimal_sector_angle -= sector_increment
                    optimal_all_points = points_in_sector(center, all_points, max_radius, baseline_angle, optimal_sector_angle)
                    optimal_points = [pt for pt in optimal_all_points if pt.id not in used_fpoi_points]
                    if min_fpoi_points <= len(optimal_points) <= max_fpoi_points:
                        best_path, path_length = solve_tsp(optimal_points, center)
                        if path_length < max_radius * 4:
                            ring_formed = True
                            rings.append({
                                'start_angle': baseline_angle,
                                'sector_angle': optimal_sector_angle,
                                'points': best_path,
                                'path_length': path_length,
                                'merged_sectors': merged_sectors - 1,  # Adjust for the reduction
                            })
                            # Mark these FPOI points as used
                            for point in best_path:
                                used_fpoi_points.add(point.id)
                            break

            if ring_formed:
                # Advance by the sector angle we used
                baseline_angle += rings[-1]['sector_angle']
                self.stdout.write(f"Formed ring at {baseline_angle - rings[-1]['sector_angle']:>6.1f}° with {len(rings[-1]['points'])} FPOI points (merged {rings[-1]['merged_sectors']} sectors)")
                self.stdout.write(f"  Progress: {len(used_fpoi_points)}/{all_points.count()} FPOIs used so far")
            else:
                # No ring could be formed, advance by base_sector_angle to avoid overlapping with previous attempts
                baseline_angle += base_sector_angle
                skipped_sectors += 1
                if baseline_angle % 30 == 0:  # Log every 30 degrees to avoid spam
                    self.stdout.write(f"No viable ring found around {baseline_angle}° - insufficient FPOI points or path too long")
                    self.stdout.write(f"  Progress: {len(used_fpoi_points)}/{all_points.count()} FPOIs used so far")

        # Output summary
        total_fpois_used = len(used_fpoi_points)
        total_fpois_available = all_points.count()
        
        self.stdout.write(f"\nRing Generation Summary:")
        self.stdout.write(f"Generated {len(rings)} rings around OLT '{block_name}'")
        self.stdout.write(f"Skipped {skipped_sectors} sectors due to insufficient FPOI points or path constraints")
        self.stdout.write(f"Total FPOI points used: {total_fpois_used} out of {total_fpois_available} available")
        
        for i, ring in enumerate(rings, 1):
            merged_info = f" (merged {ring['merged_sectors']} sectors)" if ring.get('merged_sectors', 1) > 1 else ""
            self.stdout.write(
                f"Ring {i}: Start Angle {ring['start_angle']}°, Sector {ring['sector_angle']}°, "
                f"FPOI Points {len(ring['points'])}{merged_info}, Path Length {ring['path_length']:.2f}"
            )
        
        # Verify no duplicate FPOI points
        all_ring_points = []
        for ring in rings:
            all_ring_points.extend([pt.id for pt in ring['points']])
        
        unique_points = set(all_ring_points)
        if len(all_ring_points) != len(unique_points):
            self.stderr.write(f"WARNING: Found {len(all_ring_points) - len(unique_points)} duplicate FPOI points across rings!")
        else:
            self.stdout.write(f"✓ Verified: No duplicate FPOI points across rings")
        
        # Show FPOI point breakdown
        self.stdout.write(f"\nFPOI Point Breakdown:")
        for i, ring in enumerate(rings, 1):
            point_names = [pt.name for pt in ring['points']]
            self.stdout.write(f"Ring {i}: {', '.join(point_names[:5])}{'...' if len(point_names) > 5 else ''}")
        
        # Analyze unused FPOI points
        unused_fpois = [pt for pt in all_points if pt.id not in used_fpoi_points]
        if unused_fpois:
            self.stdout.write(f"\nUnused FPOI Points ({len(unused_fpois)}):")
            for fpoi in unused_fpois:
                angle = get_angle_deg(center, fpoi.geometry)
                distance = center.distance(fpoi.geometry)
                self.stdout.write(f"  {fpoi.name}: Angle {angle:>6.1f}°, Distance {distance:>8.2f}")
            
            # Group unused FPOIs by angle ranges to see if there are patterns
            angle_ranges = {}
            for fpoi in unused_fpois:
                angle = get_angle_deg(center, fpoi.geometry)
                range_key = f"{(angle // 30) * 30:>3.0f}°-{((angle // 30) * 30 + 30):>3.0f}°"
                if range_key not in angle_ranges:
                    angle_ranges[range_key] = []
                angle_ranges[range_key].append(fpoi.name)
            
            self.stdout.write(f"\nUnused FPOI Points by Angle Ranges:")
            for range_key, names in sorted(angle_ranges.items()):
                self.stdout.write(f"  {range_key}: {', '.join(names)}")
        else:
            self.stdout.write(f"\n✓ All FPOI points were successfully used in rings!")

        if visualize:
            visualize_sectors(center, rings, max_radius, all_points)
            if generate_graphs:
                # Debug: Print more information about the data
                print(f"\nDebug Information for Graph Generation:")
                print(f"Center coordinates: ({center.x:.6f}, {center.y:.6f})")
                print(f"Max radius: {max_radius}")
                print(f"Total FPOI points: {len(all_points)}")
                print(f"Number of rings: {len(rings)}")
                
                if len(all_points) > 0:
                    print(f"FPOI point coordinates:")
                    for i, pt in enumerate(all_points[:5]):  # Show first 5
                        print(f"  {pt.name}: ({pt.geometry.x:.6f}, {pt.geometry.y:.6f})")
                    if len(all_points) > 5:
                        print(f"  ... and {len(all_points) - 5} more")
                
                if len(rings) > 0:
                    print(f"Ring information:")
                    for i, ring in enumerate(rings):
                        print(f"  Ring {i+1}: {len(ring['points'])} FPOI points")
                        if len(ring['points']) > 0:
                            first_pt = ring['points'][0]
                            print(f"    First point: {first_pt.name} at ({first_pt.geometry.x:.6f}, {first_pt.geometry.y:.6f})")
                
                create_ring_graphs(center, rings, max_radius, all_points, block_name)
