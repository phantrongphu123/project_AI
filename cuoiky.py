# -*- coding: utf-8 -*-
# !!!!! IMPORTANT !!!!!
# 1. Ensure you have Pygame installed: pip install pygame
# 2. Ensure you have OR-Tools installed (optional, for CP): pip install ortools
# 3. Save this entire code block as a single Python file (e.g., flow_solver_pygame.py)
# 4. Run from your terminal: python flow_solver_pygame.py
# ---------------------

import pygame
import sys
import copy
import time
import threading
import traceback
import heapq
from collections import deque
import itertools
import queue # To communicate with solver threads
import math # For layout calculations

# --- Check and Import OR-Tools ---
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
    print("OR-Tools imported successfully.")
except ImportError:
    print("Warning: OR-Tools library (google-ortools) not found.")
    print("Run: pip install ortools")
    print("Constraint Programming (CP) solver will be disabled.")
    cp_model = None
    ORTOOLS_AVAILABLE = False

# ============================================================
# UTILITY AND PARSING FUNCTIONS (From Original)
# ============================================================
def get_neighbors(r, c, size):
    """Gets valid neighboring cell coordinates."""
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return neighbors

def parse_puzzle_extended(puzzle_str):
    """Parses the puzzle string into grid, color data, and size."""
    lines = [line.strip() for line in puzzle_str.strip().split('\n') if line.strip()]
    if not lines: return None, None, 0, 0
    try:
        size = len(lines[0])
        if size == 0: return None, None, 0, 0
    except IndexError: return None, None, 0, 0

    grid = [[0] * size for _ in range(size)]
    points = {}
    # Mapping for colors '1'-'9' and 'A'-'Z'
    char_to_int = {str(i): i for i in range(1, 10)}
    for i, char_code in enumerate(range(ord('A'), ord('Z') + 1)):
        char_to_int[chr(char_code)] = 10 + i

    for r, row_str in enumerate(lines):
        if len(row_str) != size: return None, None, 0, 0 # Invalid row length
        for c, char in enumerate(row_str):
            if char == '.':
                grid[r][c] = 0
            elif char in char_to_int:
                color = char_to_int[char]
                if color <= 0: grid[r][c] = 0; continue # Ignore invalid color codes
                if color not in points: points[color] = []
                points[color].append((r, c))
                grid[r][c] = -color # Mark endpoints with negative values
            else:
                grid[r][c] = 0 # Treat unknown characters as empty

    colors_data = {} # Stores {'start': (r,c), 'end': (r,c)} for valid colors
    valid_colors_found = False
    invalid_colors = []
    for color, coords in points.items():
        if len(coords) == 2:
            if coords[0] == coords[1]: # Endpoint on the same cell
                invalid_colors.append(color)
            else:
                # Store start and end points
                colors_data[color] = {'start': coords[0], 'end': coords[1]}
                valid_colors_found = True
        else: # Not exactly two points for this color
            invalid_colors.append(color)

    # Clean up invalid colors from the grid
    for color in invalid_colors:
        if color in points:
             for r_err, c_err in points[color]:
                 # Check bounds before accessing grid
                 if 0 <= r_err < size and 0 <= c_err < size and grid[r_err][c_err] == -color:
                     grid[r_err][c_err] = 0 # Reset cell to empty

    if not valid_colors_found and points:
        print("Warning: No valid color pairs (exactly 2 distinct points) found.")
    # Calculate max color ID used (not strictly necessary for Pygame version)
    final_max_color_id = max(colors_data.keys()) if colors_data else 0
    return grid, colors_data, size, final_max_color_id


def reconstruct_paths(solution_grid, colors_data, size):
    """Reconstructs paths from a solved grid (typically from CP)."""
    if not solution_grid or not colors_data or size <= 0: return {}
    reconstructed = {}
    grid_copy = [row[:] for row in solution_grid]

    # Determine a marker base value safely
    all_abs_values = [abs(cell) for row in solution_grid for cell in row if cell != 0]
    marker_base = (max(all_abs_values) + 100) if all_abs_values else 100

    for color, data in colors_data.items():
        if 'start' not in data or 'end' not in data:
             print(f"Reconstruct Error: Missing start/end for color {color}")
             continue
        start_node, end_node = data['start'], data['end']

        # Validate endpoints are within bounds
        if not (0 <= start_node[0] < size and 0 <= start_node[1] < size and
                0 <= end_node[0] < size and 0 <= end_node[1] < size):
            print(f"Reconstruct Warning: Endpoint for color {color} out of bounds.")
            continue

        # --- BFS based path reconstruction ---
        queue = deque([(start_node, [start_node])]) # Store (current_pos, path_list)
        visited_bfs = {start_node} # Keep track of visited cells for this color's BFS
        found_path = False
        final_path_list = None

        while queue:
            (curr_r, curr_c), current_path_list = queue.popleft()

            if (curr_r, curr_c) == end_node:
                final_path_list = current_path_list
                found_path = True
                break # Found the target

            # Explore neighbors
            for nr, nc in get_neighbors(curr_r, curr_c, size):
                neighbor_pos = (nr, nc)
                if neighbor_pos in visited_bfs:
                    continue # Already visited in this BFS traversal

                # Check bounds just in case
                if not (0 <= nr < size and 0 <= nc < size): continue

                # Check the value in the *original* solution grid
                cell_val = solution_grid[nr][nc]

                # Check if the neighbor cell belongs to this color's path or is the endpoint
                is_correct_color_path = (cell_val == color)
                is_correct_color_end = (cell_val == -color and neighbor_pos == end_node)

                if is_correct_color_path or is_correct_color_end:
                    visited_bfs.add(neighbor_pos)
                    new_path = current_path_list + [neighbor_pos]
                    queue.append((neighbor_pos, new_path))

        # --- Store result ---
        if found_path and final_path_list:
            reconstructed[color] = final_path_list
        else:
            print(f"Reconstruct Warning: Could not complete path for color {color} from {start_node} to {end_node}.")
            # Optionally store partial path if needed: reconstructed[color] = current_path_list (from last element in queue?)

    # Final validation check
    if len(reconstructed) != len(colors_data):
        print(f"Reconstruct Error: Reconstructed {len(reconstructed)}/{len(colors_data)} paths.")
        missing = set(colors_data.keys()) - set(reconstructed.keys())
        print(f"Missing paths for colors: {missing}")
        # Depending on strictness, you might return {} here if incomplete

    return reconstructed


def is_grid_full(grid, size):
    """Checks if the grid has any empty (0) cells."""
    if not grid or size <= 0: return False
    for r in range(size):
        for c in range(size):
            # Negative values (endpoints) are considered filled
            if grid[r][c] == 0:
                return False
    return True

def get_path_data_copy(paths_dict):
    """Creates a deep copy of the paths dictionary."""
    return copy.deepcopy(paths_dict)

def manhattan_distance(p1, p2):
    """Calculates the Manhattan distance between two points."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# ============================================================
# SOLVING ALGORITHMS (Implementations from Original Tkinter Code)
# ============================================================

# --- Backtracking ---
def backtrack_solver(grid, current_paths, size, color_order):
    """Recursive backtracking helper function."""
    active_color = -1
    all_complete = True
    # Find the next incomplete color based on the predefined order
    for color in color_order:
        if color in current_paths and not current_paths[color]['complete']:
            active_color = color
            all_complete = False
            break

    # Base case: All colors have completed their paths
    if all_complete:
        # Important: The grid passed here might contain positive values for paths.
        # We need a final grid with negative endpoints for the is_grid_full check.
        final_check_grid = [row[:] for row in grid]
        for c, data in current_paths.items():
            if data.get('complete'): # Check if complete exists and is True
                # Check if coords exist and have at least start/end
                if 'coords' in data and len(data['coords']) >= 1:
                     sr, sc = data['coords'][0]
                     # Mark start point negative
                     if 0 <= sr < size and 0 <= sc < size:
                         final_check_grid[sr][sc] = -c
                     # Mark end point negative (if path has more than one node)
                     if len(data['coords']) > 1:
                          er, ec = data['coords'][-1]
                          if 0 <= er < size and 0 <= ec < size:
                              final_check_grid[er][ec] = -c
        # Return the grid with negative endpoints only if it's full
        return (final_check_grid, copy.deepcopy(current_paths)) if is_grid_full(final_check_grid, size) else (None, None)

    # If no active color found, but not all are complete (error state?)
    if active_color == -1:
        return None, None

    path_data = current_paths[active_color]
    # Ensure path coordinates exist and are not empty
    if not path_data.get('coords'): return None, None
    current_head = path_data['coords'][-1]
    target = path_data['target'] # Assumes 'target' is always present

    # Optimization: Sort neighbors by distance to target
    neighbors_sorted = sorted(get_neighbors(current_head[0], current_head[1], size),
                            key=lambda p: manhattan_distance(p, target))

    for nr, nc in neighbors_sorted:
        # Check bounds just in case
        if not (0 <= nr < size and 0 <= nc < size): continue

        cell_value = grid[nr][nc]
        is_target = (nr, nc) == target

        # Move conditions:
        # 1. Empty cell AND not already in the current path (avoids immediate U-turns)
        is_valid_empty_cell = (cell_value == 0 and (nr, nc) not in path_data['coords'])
        # 2. Target cell AND its value in the grid matches the negative endpoint value
        is_valid_target_cell = (is_target and cell_value == -active_color)

        if is_valid_empty_cell:
            # Try moving to the empty cell
            grid[nr][nc] = active_color  # Mark path with positive color value
            path_data['coords'].append((nr, nc))

            # Recurse
            result_grid, result_paths = backtrack_solver(grid, current_paths, size, color_order)
            if result_grid:
                return result_grid, result_paths # Solution found

            # Backtrack
            path_data['coords'].pop()
            grid[nr][nc] = 0 # Reset cell to empty

        elif is_valid_target_cell:
            # Try moving to the target cell
            # Avoid completing an already completed path (safety check)
            if path_data['complete']: continue

            path_data['coords'].append((nr, nc))
            path_data['complete'] = True # Mark as complete

            # Recurse
            result_grid, result_paths = backtrack_solver(grid, current_paths, size, color_order)
            if result_grid:
                return result_grid, result_paths # Solution found

            # Backtrack
            path_data['complete'] = False
            path_data['coords'].pop()
            # No need to reset grid[nr][nc] as it's a fixed negative endpoint

    # If no move leads to a solution from this state
    return None, None

def solve_backtracking(puzzle_str, time_limit=60.0):
    """Wrapper for Backtracking solver with time limit."""
    start_time_total = time.time()
    solution_grid, solution_paths_dict = None, None
    status_message = "Timeout" # Default status

    try:
        initial_grid, initial_colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid is None or not initial_colors_data or size <= 0:
            raise ValueError("Invalid puzzle data for Backtracking.")
    except ValueError as e:
        print(f"Backtracking Parse Error: {e}")
        return None, None, time.time() - start_time_total, f"Parse Error: {e}"

    # Create initial grid with only negative endpoints
    clean_grid = [[0] * size for _ in range(size)]
    current_paths_internal = {}
    valid_setup = True
    for color, data in initial_colors_data.items():
        sr, sc = data['start']
        er, ec = data['end']
        # Bounds check before setting endpoints
        if not (0 <= sr < size and 0 <= sc < size and 0 <= er < size and 0 <= ec < size):
            print(f"Backtracking Setup Error: Endpoint for color {color} out of bounds.")
            valid_setup = False
            break
        clean_grid[sr][sc] = -color
        clean_grid[er][ec] = -color
        # Initialize path data for the recursive solver
        current_paths_internal[color] = {'coords': [data['start']], 'target': data['end'], 'complete': False}

    if not valid_setup:
        return None, None, time.time() - start_time_total, "Setup Error: Invalid Endpoints"

    # Sort colors (e.g., by Manhattan distance) - Optimization
    color_order = sorted(initial_colors_data.keys(),
                         key=lambda c: manhattan_distance(initial_colors_data[c]['start'], initial_colors_data[c]['end']))

    print("Solving with Backtracking (Optimized)...")
    # Create copies to pass to the thread/recursive function
    grid_copy = [row[:] for row in clean_grid]
    paths_copy = copy.deepcopy(current_paths_internal)

    # --- Threading for Time Limit ---
    solver_thread_result = {}
    timed_out = False

    def solver_task():
        try:
            # Call the recursive helper function
            grid_res, paths_res = backtrack_solver(grid_copy, paths_copy, size, color_order)
            solver_thread_result['grid'] = grid_res
            solver_thread_result['paths'] = paths_res # This paths dict has 'coords', 'target', 'complete'
            solver_thread_result['error'] = None
        except Exception as e:
            solver_thread_result['error'] = e
            solver_thread_result['traceback'] = traceback.format_exc()
            print(f"Error in Backtracking thread: {e}")
            traceback.print_exc()

    solver_thread = threading.Thread(target=solver_task, daemon=True)
    solver_thread.start()
    solver_thread.join(timeout=time_limit)

    if solver_thread.is_alive():
        print(f"Backtracking timed out after {time_limit}s.")
        timed_out = True
        status_message = "Timeout"
    elif solver_thread_result.get('error'):
        status_message = f"Runtime Error" # Keep status brief
        print(f"Backtracking thread error: {solver_thread_result['error']}")
        # print(solver_thread_result.get('traceback', ''))
    else:
        # Thread finished within time limit
        solution_grid = solver_thread_result.get('grid')
        solved_paths_internal = solver_thread_result.get('paths') # Dict with 'coords', 'target', etc.

        if solution_grid and solved_paths_internal:
            # Grid should already have negative endpoints from backtrack_solver if successful
            if is_grid_full(solution_grid, size):
                 status_message = "Success"
                 # Extract just the coordinate lists for the final result
                 solution_paths_dict = {color: data['coords'] for color, data in solved_paths_internal.items()}
                 # Final check: ensure the number of paths matches the number of colors
                 if len(solution_paths_dict) != len(initial_colors_data):
                      print("Backtracking Error: Path count mismatch after solve.")
                      status_message = "Path Count Mismatch"
                      solution_grid = None
                      solution_paths_dict = None
            else:
                 print("Backtracking Warning: Solver returned a grid, but it's not full.")
                 status_message = "Incomplete Grid"
                 solution_grid = None # Invalidate if not full
                 solution_paths_dict = None
        else:
            # Solver finished but found no solution
            status_message = "No Solution Found"

    solve_time = time.time() - start_time_total
    print(f"Backtracking Time: {solve_time:.4f}s, Status: {status_message}")

    # Ensure consistency: if grid is None, paths should be None
    if solution_grid is None:
        solution_paths_dict = None

    return solution_grid, solution_paths_dict, solve_time, status_message


# --- Constraint Programming ---
def solve_cp(puzzle_str, time_limit=60.0):
    """Wrapper for Constraint Programming (OR-Tools)."""
    if not ORTOOLS_AVAILABLE:
        print("CP Error: OR-Tools not available.")
        return None, None, 0.0, "OR-Tools N/A"

    start_time_total = time.time()
    solution_grid, solution_paths_dict = None, None
    status_message = "Unknown Error" # Default status

    try:
        initial_grid_cp, colors_data_cp, size_cp, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid_cp is None or not colors_data_cp or size_cp <= 0:
            raise ValueError("Invalid puzzle data for CP.")
    except ValueError as e:
        print(f"CP Parse Error: {e}")
        return None, None, time.time() - start_time_total, f"Parse Error: {e}"

    colors = list(colors_data_cp.keys())
    num_colors = len(colors)
    if num_colors == 0:
        print("CP Error: No valid colors in puzzle.")
        return None, None, time.time() - start_time_total, "No Valid Colors"

    # Map colors to indices 0..N-1 for the model
    color_map = {c: i for i, c in enumerate(colors)}
    idx_to_color = {i: c for c, i in color_map.items()}

    # 1. Create the model.
    model = cp_model.CpModel()

    # 2. Create variables.
    # is_path[r, c, k] = 1 if cell (r, c) is part of path k, 0 otherwise.
    is_path = {}
    for r in range(size_cp):
        for c in range(size_cp):
            for k_idx in range(num_colors): # Use index k_idx for model
                is_path[r, c, k_idx] = model.NewBoolVar(f'p_{r}_{c}_{k_idx}')

    # 3. Add Constraints.
    # Constraint 1: Each cell must belong to exactly one path.
    for r in range(size_cp):
        for c in range(size_cp):
            model.Add(sum(is_path[r, c, k_idx] for k_idx in range(num_colors)) == 1)

    # Constraint 2: Endpoints must belong to their correct color path.
    valid_endpoints = True
    for k_color, k_idx in color_map.items():
        # Get start/end, assuming they exist from parser logic
        sr, sc = colors_data_cp[k_color]['start']
        er, ec = colors_data_cp[k_color]['end']
        # Bounds check before adding constraint
        if not (0 <= sr < size_cp and 0 <= sc < size_cp and 0 <= er < size_cp and 0 <= ec < size_cp):
            print(f"CP Setup Error: Endpoint for color {k_color} out of bounds.")
            valid_endpoints = False
            break
        model.Add(is_path[sr, sc, k_idx] == 1)
        model.Add(is_path[er, ec, k_idx] == 1)
        # (No need to explicitly constrain other colors away from endpoints due to Constraint 1)

    if not valid_endpoints:
        return None, None, time.time() - start_time_total, "Setup Error: Invalid Endpoints"

    # Constraint 3: Path Connectivity (Neighbor Counting Method).
    for k_color, k_idx in color_map.items():
        sr, sc = colors_data_cp[k_color]['start']
        er, ec = colors_data_cp[k_color]['end']
        for r in range(size_cp):
            for c in range(size_cp):
                neighbors = get_neighbors(r, c, size_cp)
                # Sum of neighbors belonging to the same color path k_idx
                sum_neighbors = sum(is_path[nr, nc, k_idx] for nr, nc in neighbors)

                # Intermediate variable: Is the current cell (r, c) an endpoint for color k?
                # Using NewConstant is efficient here.
                is_rc_endpoint_k_flag = (r == sr and c == sc) or (r == er and c == ec)
                # Link the boolean variable to the flag condition if needed for complex constraints,
                # but direct use in OnlyEnforceIf with NewConstant is often clearer.
                # cell_is_endpoint_k = model.NewBoolVar(f'is_endpoint_{r}_{c}_{k_idx}')
                # model.Add(cell_is_endpoint_k == is_rc_endpoint_k_flag)

                # Logic: Apply constraints only if the cell (r,c) belongs to path k.
                cell_is_color_k = is_path[r, c, k_idx]

                # If (r,c) is color k AND is an endpoint => must have exactly 1 neighbor of color k.
                model.Add(sum_neighbors == 1).OnlyEnforceIf(cell_is_color_k).OnlyEnforceIf(model.NewConstant(is_rc_endpoint_k_flag))

                # If (r,c) is color k AND is NOT an endpoint => must have exactly 2 neighbors of color k.
                model.Add(sum_neighbors == 2).OnlyEnforceIf(cell_is_color_k).OnlyEnforceIf(model.NewConstant(not is_rc_endpoint_k_flag))

                # If (r,c) is NOT color k => must have 0 neighbors of color k (implied by Constraint 1, but can be added).
                # model.Add(sum_neighbors == 0).OnlyEnforceIf(cell_is_color_k.Not())

    # 4. Solve the model.
    print("Solving with Constraint Programming (OR-Tools)...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False # Disable verbose logging
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    # Process Results
    solve_time_solver = solver.WallTime()
    solve_time_total = time.time() - start_time_total
    status_name = solver.StatusName(status)
    print(f"CP Solver Time: {solve_time_solver:.4f}s, Total Function Time: {solve_time_total:.4f}s")
    print(f"CP Status: {status_name}")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        status_message = "Success"
        print("CP: Solution found.")
        # Build the solution grid from the solver's variable assignments
        solution_grid = [[0] * size_cp for _ in range(size_cp)]
        grid_build_error = False
        try:
            for r in range(size_cp):
                for c in range(size_cp):
                    found_color_for_cell = False
                    for k_idx in range(num_colors):
                        if solver.Value(is_path[r, c, k_idx]):
                            actual_color = idx_to_color[k_idx]
                            # Check if it's an endpoint for *this specific color*
                            is_start = (r, c) == colors_data_cp[actual_color]['start']
                            is_end = (r, c) == colors_data_cp[actual_color]['end']
                            solution_grid[r][c] = -actual_color if (is_start or is_end) else actual_color
                            found_color_for_cell = True
                            break # Move to the next cell
                    if not found_color_for_cell:
                        # This should not happen if the model is correct and feasible
                        print(f"CRITICAL CP ERROR: Cell ({r},{c}) not assigned to any color path!")
                        solution_grid[r][c] = 99 # Mark error clearly
                        grid_build_error = True
                        status_message = "Grid Build Error"
                        # Potentially stop or just flag the error

            if grid_build_error:
                solution_grid = None # Invalidate grid if any cell failed assignment

        except Exception as e:
            print(f"CP: Error building grid from solution: {e}")
            traceback.print_exc()
            solution_grid = None # Invalidate grid on exception
            status_message = "Grid Build Exception"

        # Reconstruct paths only if grid was built successfully
        if solution_grid:
            print("CP: Reconstructing paths from solution grid...")
            # Pass the original colors_data (with start/end) for reconstruction
            solution_paths_dict = reconstruct_paths(solution_grid, colors_data_cp, size_cp)
            # Validate reconstruction
            if not solution_paths_dict or len(solution_paths_dict) != len(colors_data_cp):
                 print("CP Warning: Path reconstruction failed or incomplete.")
                 status_message = "Path Recon Failed"
                 # Decide if this invalidates the whole solution
                 solution_paths_dict = None # Invalidate paths if reconstruction fails
                 # solution_grid = None # Optionally invalidate grid too
            else:
                 print("CP: Path reconstruction successful.")


    elif status == cp_model.INFEASIBLE:
        status_message = "Infeasible"
        print("CP Solver: Model is infeasible (no solution exists).")
    elif status == cp_model.MODEL_INVALID:
        status_message = "Model Invalid"
        print("CP Solver: Model is invalid.")
    else: # Includes UNKNOWN, ABORTED (often due to timeout)
        status_message = status_name # Use the solver's status name
        print(f"CP Solver: Status is {status_name}.")
        # Refine Timeout detection
        if status_message == "UNKNOWN" and solve_time_solver >= time_limit * 0.98:
            status_message = "Timeout"
            print("(Interpreted as Timeout)")

    # Ensure consistency on failure
    if status_message != "Success":
        solution_grid = None
        solution_paths_dict = None

    return solution_grid, solution_paths_dict, solve_time_total, status_message


# --- Breadth-First Search ---
def solve_bfs(puzzle_str, time_limit=60.0, state_limit=500000):
    """Wrapper for Breadth-First Search solver."""
    start_time_total = time.time()
    solution_grid, solution_paths_dict = None, None
    states_explored = 0
    status_message = "Timeout / State Limit" # Default status

    try:
        # Parse returns grid with negative endpoints
        initial_grid, initial_colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid is None or not initial_colors_data or size <= 0:
            raise ValueError("Invalid puzzle data for BFS.")
    except ValueError as e:
        print(f"BFS Parse Error: {e}")
        return None, None, time.time() - start_time_total, f"Parse Error: {e}"

    color_order = sorted(initial_colors_data.keys()) # Fixed order for processing colors
    num_colors = len(color_order)
    if num_colors == 0:
        return None, None, time.time() - start_time_total, "No Valid Colors"

    # Initial state setup for BFS internal logic
    initial_paths_internal = {
        c: {'coords': [initial_colors_data[c]['start']], # Path starts with start point
            'target': initial_colors_data[c]['end'],
            'complete': False}
        for c in color_order
    }
    # Grid for state needs negative endpoints
    initial_grid_tuple = tuple(map(tuple, initial_grid))
    # State: (grid_tuple, paths_dict_internal, color_index_to_process)
    initial_state = (initial_grid_tuple, initial_paths_internal, 0)

    queue = deque([initial_state])
    # Visited set stores (grid_tuple, color_index_to_process)
    visited = {(initial_grid_tuple, 0)}

    print("Solving with BFS...")
    found_solution = False

    while queue:
        # --- Check Limits ---
        current_time = time.time()
        if current_time - start_time_total > time_limit:
            status_message = "Timeout"
            print(f"BFS timed out after {time_limit}s.")
            break
        if states_explored > state_limit:
            status_message = "State Limit Exceeded"
            print(f"BFS exceeded state limit ({state_limit}).")
            break

        # --- Dequeue State ---
        current_grid_tuple, current_paths, current_color_idx = queue.popleft()
        states_explored += 1

        # --- Check for Goal State ---
        # Goal: All paths are marked 'complete' AND the grid is full
        all_complete = all(data.get('complete', False) for data in current_paths.values())
        if all_complete:
            # The grid_tuple in the state still has positive path values.
            # We need the version with negative endpoints for the check.
            final_grid_list = [list(row) for row in current_grid_tuple] # Temp list grid
            # No, wait. BFS state grid ONLY has endpoints marked negative.
            # Path values are positive during the search.
            # Need to construct the final grid for checking is_grid_full.
            final_check_grid = [row[:] for row in final_grid_list]
            for c, data in current_paths.items():
                 if data.get('complete'):
                     if 'coords' in data and len(data['coords']) >= 1:
                          sr, sc = data['coords'][0]
                          if 0 <= sr < size and 0 <= sc < size: final_check_grid[sr][sc] = -c
                          if len(data['coords']) > 1:
                               er, ec = data['coords'][-1]
                               if 0 <= er < size and 0 <= ec < size: final_check_grid[er][ec] = -c

            if is_grid_full(final_check_grid, size):
                print(f"BFS: Solution found after {states_explored} states.")
                solution_grid = final_check_grid # Grid with negative endpoints is the solution
                solution_paths_dict = {c: d['coords'] for c, d in current_paths.items()}
                status_message = "Success"
                found_solution = True
                break # Exit while loop
            else:
                # All paths complete, but grid not full - this state is a dead end for *this* problem
                continue # Explore other states

        # --- State Expansion ---
        # Find the actual color to process (skip completed colors)
        actual_color_idx_to_process = current_color_idx
        while actual_color_idx_to_process < num_colors:
             active_color_check = color_order[actual_color_idx_to_process]
             # Check if color exists and is not complete
             if active_color_check in current_paths and not current_paths[active_color_check].get('complete', False):
                 break # Found an incomplete color to process
             actual_color_idx_to_process += 1
        else:
             # All remaining colors are complete, but goal check failed (grid not full)
             continue # This branch is finished

        # Process the determined active color
        active_color = color_order[actual_color_idx_to_process]
        path_data = current_paths[active_color]
        if not path_data.get('coords'): continue # Skip if path coords somehow missing
        current_head = path_data['coords'][-1]
        target = path_data['target']
        # Use a mutable list version of the grid for checking neighbors
        current_grid_list_for_check = [list(row) for row in current_grid_tuple]

        # Explore neighbors of the current head
        for nr, nc in get_neighbors(current_head[0], current_head[1], size):
            # Bounds check (redundant if get_neighbors is correct, but safe)
            if not (0 <= nr < size and 0 <= nc < size): continue

            cell_value = current_grid_list_for_check[nr][nc]
            is_target = (nr, nc) == target

            # Conditions for valid move
            is_valid_empty_cell = (cell_value == 0 and (nr, nc) not in path_data['coords'])
            is_valid_target_cell = (is_target and cell_value == -active_color)

            if is_valid_empty_cell or is_valid_target_cell:
                # Create the next state (deep copies needed)
                next_paths = get_path_data_copy(current_paths)
                next_grid_list = [row[:] for row in current_grid_list_for_check]
                next_color_idx_for_state = actual_color_idx_to_process # Assume stay on same color initially

                if is_valid_empty_cell:
                    next_grid_list[nr][nc] = active_color # Mark path with positive color
                    next_paths[active_color]['coords'].append((nr, nc))
                    # Stay processing the same color in the next state
                elif is_valid_target_cell:
                    if next_paths[active_color]['complete']: continue # Should not happen
                    next_paths[active_color]['coords'].append((nr, nc))
                    next_paths[active_color]['complete'] = True
                    # Move processing to the next color index in the next state
                    next_color_idx_for_state += 1

                # Convert grid back to tuple for hashing/visited check
                next_grid_tuple = tuple(map(tuple, next_grid_list))
                next_state_key = (next_grid_tuple, next_color_idx_for_state)

                # Add to queue only if not visited
                if next_state_key not in visited:
                    visited.add(next_state_key)
                    next_state = (next_grid_tuple, next_paths, next_color_idx_for_state)
                    queue.append(next_state)

    # --- After Loop ---
    solve_time = time.time() - start_time_total
    if not found_solution:
        print(f"BFS did not find a solution. Status: {status_message}")
    print(f"BFS Time: {solve_time:.4f}s, States Explored: {states_explored}")

    # Ensure consistency on failure
    if status_message != "Success":
        solution_grid = None
        solution_paths_dict = None

    return solution_grid, solution_paths_dict, solve_time, status_message


# --- A* Search ---
def calculate_heuristic_astar(paths_dict, color_order, size):
    """Heuristic for A*: Sum of Manhattan distances for incomplete paths."""
    h_manhattan = 0
    # Optional: Add penalty for number of incomplete paths?
    # incomplete_count = 0
    for color in color_order:
        data = paths_dict.get(color)
        # Check if path data exists and is not complete
        if data and not data.get('complete', False):
            # Check if coords exist and are not empty
            if data.get('coords'):
                current_head = data['coords'][-1]
                target = data['target'] # Assumes target exists
                h_manhattan += manhattan_distance(current_head, target)
                # incomplete_count += 1
    # return h_manhattan + incomplete_count # Heuristic with penalty
    return h_manhattan

def solve_astar(puzzle_str, time_limit=120.0, state_limit=300000):
    """Wrapper for A* Search solver."""
    start_time_total = time.time()
    solution_grid, solution_paths_dict = None, None
    states_explored = 0
    status_message = "Timeout / State Limit" # Default status

    try:
        # Parse returns grid with negative endpoints
        initial_grid, initial_colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid is None or not initial_colors_data or size <= 0:
            raise ValueError("Invalid puzzle data for A*.")
    except ValueError as e:
        print(f"A* Parse Error: {e}")
        return None, None, time.time() - start_time_total, f"Parse Error: {e}"

    color_order = sorted(initial_colors_data.keys())
    num_colors = len(color_order)
    if num_colors == 0:
        return None, None, time.time() - start_time_total, "No Valid Colors"

    # Initial state setup for A* internal logic
    initial_paths_internal = {
        c: {'coords': [initial_colors_data[c]['start']],
            'target': initial_colors_data[c]['end'],
            'complete': False}
        for c in color_order
    }
    initial_grid_tuple = tuple(map(tuple, initial_grid))
    # State: (grid_tuple, paths_dict_internal, color_index_to_process)
    initial_state = (initial_grid_tuple, initial_paths_internal, 0)

    # Priority Queue: (f_score, unique_id, g_score, state)
    state_counter = itertools.count() # Tie-breaker for equal f_scores
    initial_g = 0 # Cost (path length) to reach initial state
    initial_h = calculate_heuristic_astar(initial_paths_internal, color_order, size)
    priority_queue = [(initial_g + initial_h, next(state_counter), initial_g, initial_state)]

    # Visited set stores best g_score found so far for a state key
    # visited = { state_key = (grid_tuple, color_idx_to_process) : min_g_score }
    visited = {(initial_grid_tuple, 0): initial_g}

    print("Solving with A*...")
    found_solution = False

    while priority_queue:
        # --- Check Limits ---
        current_time = time.time()
        if current_time - start_time_total > time_limit:
            status_message = "Timeout"
            print(f"A* timed out after {time_limit}s.")
            break
        if states_explored > state_limit:
            status_message = "State Limit Exceeded"
            print(f"A* exceeded state limit ({state_limit}).")
            break

        # --- Pop lowest f-score state ---
        f_score, _, current_g, current_state = heapq.heappop(priority_queue)
        current_grid_tuple, current_paths, current_color_idx = current_state
        states_explored += 1

        # --- Check if already found a better path to this state ---
        # (This check is more efficient after popping)
        state_key = (current_grid_tuple, current_color_idx)
        if current_g > visited.get(state_key, float('inf')):
             continue # Skip if we found a shorter path previously

        # --- Check for Goal State ---
        all_complete = all(data.get('complete', False) for data in current_paths.values())
        if all_complete:
            # Construct the final grid with negative endpoints for checking fullness
            final_grid_list = [list(row) for row in current_grid_tuple]
            final_check_grid = [row[:] for row in final_grid_list]
            for c, data in current_paths.items():
                 if data.get('complete'):
                     if 'coords' in data and len(data['coords']) >= 1:
                          sr, sc = data['coords'][0]
                          if 0 <= sr < size and 0 <= sc < size: final_check_grid[sr][sc] = -c
                          if len(data['coords']) > 1:
                               er, ec = data['coords'][-1]
                               if 0 <= er < size and 0 <= ec < size: final_check_grid[er][ec] = -c

            if is_grid_full(final_check_grid, size):
                print(f"A*: Solution found after {states_explored} states.")
                solution_grid = final_check_grid # Final grid has neg endpoints
                solution_paths_dict = {c: d['coords'] for c, d in current_paths.items()}
                status_message = "Success"
                found_solution = True
                break # Exit while loop
            else:
                # All paths complete, but grid not full - prune this branch
                continue

        # --- State Expansion ---
        # Find the actual color to process (skip completed)
        actual_color_idx_to_process = current_color_idx
        while actual_color_idx_to_process < num_colors:
             active_color_check = color_order[actual_color_idx_to_process]
             if active_color_check in current_paths and not current_paths[active_color_check].get('complete', False):
                 break
             actual_color_idx_to_process += 1
        else:
             continue # All remaining complete, goal check failed

        active_color = color_order[actual_color_idx_to_process]
        path_data = current_paths[active_color]
        if not path_data.get('coords'): continue
        current_head = path_data['coords'][-1]
        target = path_data['target']
        current_grid_list_for_check = [list(row) for row in current_grid_tuple]

        # Explore neighbors
        for nr, nc in get_neighbors(current_head[0], current_head[1], size):
            if not (0 <= nr < size and 0 <= nc < size): continue

            cell_value = current_grid_list_for_check[nr][nc]
            is_target = (nr, nc) == target

            # Valid move conditions
            is_valid_empty_cell = (cell_value == 0 and (nr, nc) not in path_data['coords'])
            is_valid_target_cell = (is_target and cell_value == -active_color)

            if is_valid_empty_cell or is_valid_target_cell:
                # Create the next state candidate
                next_paths = get_path_data_copy(current_paths)
                next_grid_list = [row[:] for row in current_grid_list_for_check]
                next_color_idx_for_state = actual_color_idx_to_process
                # Calculate g_score for the next state (cost increases by 1 for the move)
                next_g = current_g + 1

                if is_valid_empty_cell:
                    next_grid_list[nr][nc] = active_color # Mark path positive
                    next_paths[active_color]['coords'].append((nr, nc))
                    # Stay on same color index
                elif is_valid_target_cell:
                    if next_paths[active_color]['complete']: continue # Safety
                    next_paths[active_color]['coords'].append((nr, nc))
                    next_paths[active_color]['complete'] = True
                    # Move to next color index
                    next_color_idx_for_state += 1

                # Convert grid to tuple for state key
                next_grid_tuple = tuple(map(tuple, next_grid_list))
                next_state_key = (next_grid_tuple, next_color_idx_for_state)

                # Check if this state is new or offers a better path (lower g_score)
                if next_g < visited.get(next_state_key, float('inf')):
                    visited[next_state_key] = next_g # Update visited with better g_score
                    # Calculate heuristic (h) and f_score for the new state
                    next_h = calculate_heuristic_astar(next_paths, color_order, size)
                    next_f = next_g + next_h
                    # Create the full state object to push
                    next_state = (next_grid_tuple, next_paths, next_color_idx_for_state)
                    # Push to priority queue
                    heapq.heappush(priority_queue, (next_f, next(state_counter), next_g, next_state))

    # --- After Loop ---
    solve_time = time.time() - start_time_total
    if not found_solution:
        print(f"A* did not find a solution. Status: {status_message}")
    print(f"A* Time: {solve_time:.4f}s, States Explored: {states_explored}")

    # Ensure consistency on failure
    if status_message != "Success":
        solution_grid = None
        solution_paths_dict = None

    return solution_grid, solution_paths_dict, solve_time, status_message


# ============================================================
# PUZZLE DATA (Keep Original)
# ============================================================
PUZZLES = {
    "Easy (5x5)": [ """ 1.2.5\n..3.4\n.....\n.2.5.\n.134. """, """ 1....\n.....\n..4..\n342.1\n2...3 """, """ .123.\n...5.\n..5..\n1..4.\n2.43. """, """ 1.2.3\n.4...\n.4.5.\n...5.\n1.2.3 """, """ ..1..\n2...3\n.4.4.\n2...3\n..1.. """ ],
    "Medium (6x6)": [ """ 1.....\n.2.3..\n.2.3..\n..45..\n..45.1\n...... """, """ 1.2..3\n......\n.4.5..\n.4.5..\n......\n1.2..3 """, """ ..1...\n.2.3.4\n.2.3.4\n..5...\n..5.6.\n..1.6. """, """ 1....2\n.3....\n.3....\n....4.\n.5.4.2\n1.5... """, """ ..12..\n.3..4.\n.3..4.\n.5..6.\n.5..6.\n..12.. """ ],
    "Hard (7x7)": [ """ 1.2....\n.3.....\n.3.4.5.\n...4.5.\n.6.....\n.6.7.2.\n1....7. """, """ .......\n.1.2.3.\n.1.2.3.\n.4.5.6.\n.4.5.6.\n.7.8.9.\n.7.8.9. """, """ 1.2.3..\n.......\n4.5.6..\n4.5.6.7\n.......\n.8.9.7.\n18293.. """ ],
    "Very Hard (8x8)": [ """ ....1...\n....2.34\n....34.2\n...65...\n....6...\n....5...\n........\n.1...... """, """ 1.2.3.4.\n........\n.5.6.7.8\n........\n.9.A.B.C\n........\n.D.E.F.G\n159D26AE """, """ ........\n.1.2.3.4\n.1.2.3.4\n.5.6.7.8\n.5.6.7.8\n.9.A.B.C\n.9.A.B.C\n........ """ ]
}

# ============================================================
# PYGAME APPLICATION CLASS (2-Column Layout)
# ============================================================
class FlowFreePygameApp:
    def __init__(self):
        pygame.init()
        # --- Window and Layout Config ---
        self.SCREEN_WIDTH = 1050 # Adjusted width for clarity
        self.SCREEN_HEIGHT = 750
        self.LEFT_PANEL_WIDTH = 300 # Slightly wider left panel
        self.PANEL_GAP = 10
        self.RIGHT_PANEL_WIDTH = self.SCREEN_WIDTH - self.LEFT_PANEL_WIDTH - self.PANEL_GAP
        self.STATUS_BAR_HEIGHT = 30
        self.PADDING = 15

        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("Flow Free Solver (Pygame) - 2 Columns")
        self.clock = pygame.time.Clock()
        # --- Fonts ---
        # Prioritize clearer fonts if available
        try:
            self.font_ui = pygame.font.SysFont("Segoe UI", 17)
            self.font_ui_label = pygame.font.SysFont("Segoe UI", 19, bold=True)
            self.font_grid = pygame.font.SysFont("Consolas", 14) # Monospace good for grids
            self.font_grid_bold = pygame.font.SysFont("Consolas", 16, bold=True)
            self.font_status = pygame.font.SysFont("Segoe UI", 14)
            self.font_result_info = pygame.font.SysFont("Segoe UI", 13)
            self.font_grid_small = None # Initialized dynamically
            self.font_grid_bold_small = None
        except pygame.error:
            print("Warning: Preferred fonts (Segoe UI, Consolas) not found, using Arial.")
            self.font_ui = pygame.font.SysFont("Arial", 16)
            self.font_ui_label = pygame.font.SysFont("Arial", 18, bold=True)
            self.font_grid = pygame.font.SysFont("Arial", 12)
            self.font_grid_bold = pygame.font.SysFont("Arial", 14, bold=True)
            self.font_status = pygame.font.SysFont("Arial", 14)
            self.font_result_info = pygame.font.SysFont("Arial", 12)
            self.font_grid_small = None
            self.font_grid_bold_small = None

        # --- Colors (Improved Palette) ---
        self.BG_COLOR = (248, 249, 250)
        self.LEFT_PANEL_BG = (233, 236, 239)
        self.RIGHT_PANEL_BG = (222, 226, 230)
        self.TEXT_COLOR = (33, 37, 41)
        self.BUTTON_COLOR = (206, 212, 218)
        self.BUTTON_HOVER_COLOR = (173, 181, 189) # Darker hover
        self.BUTTON_TEXT_COLOR = (52, 58, 64)
        self.CHECKBOX_BORDER_COLOR = (108, 117, 125)
        self.CHECKBOX_FILL_COLOR = (25, 135, 84) # Bootstrap green
        self.DISABLED_COLOR = (173, 181, 189)
        self.STATUS_BG_COLOR = (52, 58, 64)
        self.STATUS_TEXT_COLOR = (248, 249, 250)
        self.GRID_BORDER_COLOR = (173, 181, 189)
        self.SOLVING_OVERLAY_COLOR = (200, 200, 200, 170)
        self.ERROR_TEXT_COLOR = (220, 53, 69) # Bootstrap danger red

        # Flow Colors (Keep list, convert to Pygame Color)
        self.flow_colors_hex = ['#E9ECEF'] + ['#DC3545', '#0D6EFD', '#198754', '#FFC107', '#6F42C1', '#FD7E14', '#0DCAF0', '#D63384', '#6C757D', '#F8F9FA', '#20C997', '#6610F2', '#000080', '#FF69B4', '#FFD700', '#ADFF2F', '#D2691E', '#5F9EA0', '#DC143C', '#7FFF00', '#CD5C5C', '#4682B4', '#9ACD32', '#EE82EE', '#F5DEB3', '#8A2BE2', '#32CD32', '#DAA520', '#FA8072', '#7FFFD4', '#DDA0DD', '#B0E0E6', '#87CEEB', '#98FB98', '#F0E68C'] * 2 # Brighter, more distinct colors
        self.flow_colors_rgb = [pygame.Color(c) for c in self.flow_colors_hex]

        # Int to Char mapping
        self.int_to_char = {i: str(i) for i in range(1, 10)}
        for i, char_code in enumerate(range(ord('A'), ord('Z') + 1)): self.int_to_char[10 + i] = chr(char_code)

        # --- State Data ---
        self.puzzles = PUZZLES
        self.difficulty_levels = list(self.puzzles.keys())
        self.current_difficulty_index = 0
        self.current_puzzle_index = 0
        self.current_puzzle_string = ""
        self.grid_size = 0
        self.initial_grid_data = None # Grid with negative endpoints from parser
        self.colors_data = {}       # Color info {'start':(r,c), 'end':(r,c)}
        self.available_algorithms = ["Backtracking", "BFS", "A*", "CP"]
        self.selected_algorithms = { algo: (algo in ["A*", "CP"]) for algo in self.available_algorithms }
        if not ORTOOLS_AVAILABLE: self.selected_algorithms["CP"] = False
        self.solver_results = {} # Stores raw results: {algo: {'grid':..,'paths':..,'time':..,'status':..}}
        self.successful_results_list = [] # Stores tuples for display: [(algo_name, display_data)]

        # Application State
        self.app_state = "SETUP" # SETUP, SOLVING, RESULTS
        self.status_message = "Sẵn sàng."
        self.is_solving = False
        self.solve_thread = None
        self.result_queue = queue.Queue()

        # --- UI Elements ---
        self.ui_elements = {} # Dictionary to store Rects and properties
        self._create_ui_elements() # Define UI element positions and sizes

        # --- Initial Load ---
        self.load_puzzle_data() # Load the first puzzle

    def _create_ui_elements(self):
        """Defines Rects and info for UI elements in the left panel."""
        self.ui_elements = {}
        y_offset = self.PADDING
        x_offset = self.PADDING
        element_h = 32 # Standard height for buttons/rows
        label_h = 24
        button_w = 45 # Width for < > buttons
        full_width = self.LEFT_PANEL_WIDTH - 2 * self.PADDING
        checkbox_size = 18
        label_gap = 8
        element_gap = 18 # Increased vertical gap

        # --- Difficulty ---
        self.ui_elements['diff_label'] = {'type': 'label', 'text': "Độ khó:", 'pos': (x_offset, y_offset), 'font': self.font_ui_label}
        y_offset += label_h # Position below label
        self.ui_elements['diff_prev_btn'] = {'type': 'button', 'text': "<", 'rect': pygame.Rect(x_offset, y_offset, button_w, element_h), 'action': 'prev_diff'}
        diff_value_x = x_offset + button_w + label_gap
        diff_value_w = full_width - 2 * (button_w + label_gap) # Calculate width for text display
        self.ui_elements['diff_value'] = {'type': 'textdisplay', 'text': "", 'rect': pygame.Rect(diff_value_x, y_offset, diff_value_w, element_h)}
        self.ui_elements['diff_next_btn'] = {'type': 'button', 'text': ">", 'rect': pygame.Rect(diff_value_x + diff_value_w + label_gap, y_offset, button_w, element_h), 'action': 'next_diff'}
        y_offset += element_h + element_gap

        # --- Puzzle ---
        self.ui_elements['puzzle_label'] = {'type': 'label', 'text': "Puzzle:", 'pos': (x_offset, y_offset), 'font': self.font_ui_label}
        y_offset += label_h
        self.ui_elements['puzzle_prev_btn'] = {'type': 'button', 'text': "<", 'rect': pygame.Rect(x_offset, y_offset, button_w, element_h), 'action': 'prev_puzzle'}
        puzzle_value_x = x_offset + button_w + label_gap
        puzzle_value_w = full_width - 2 * (button_w + label_gap)
        self.ui_elements['puzzle_value'] = {'type': 'textdisplay', 'text': "", 'rect': pygame.Rect(puzzle_value_x, y_offset, puzzle_value_w, element_h)}
        self.ui_elements['puzzle_next_btn'] = {'type': 'button', 'text': ">", 'rect': pygame.Rect(puzzle_value_x + puzzle_value_w + label_gap, y_offset, button_w, element_h), 'action': 'next_puzzle'}
        y_offset += element_h + element_gap

        # --- Algorithms ---
        self.ui_elements['algo_label'] = {'type': 'label', 'text': "Thuật toán:", 'pos': (x_offset, y_offset), 'font': self.font_ui_label}
        y_offset += label_h + label_gap // 2

        for algo in self.available_algorithms:
            cb_rect = pygame.Rect(x_offset, y_offset, checkbox_size, checkbox_size)
            # Calculate label position to align vertically with checkbox center
            label_y_adjust = (checkbox_size - self.font_ui.get_height()) // 2
            label_pos = (x_offset + checkbox_size + label_gap, y_offset + label_y_adjust)
            text = algo
            disabled = False
            if algo == "CP":
                text = "CP (OR-Tools)"
                if not ORTOOLS_AVAILABLE:
                    disabled = True
                    self.selected_algorithms["CP"] = False # Ensure it stays false if disabled

            self.ui_elements[f'algo_cb_{algo}'] = {
                'type': 'checkbox', 'text': text, 'rect': cb_rect,
                'label_pos': label_pos, 'action': f'toggle_{algo}',
                'disabled': disabled,
                'text_rect_w': full_width - checkbox_size - label_gap # Max width for label text
            }
            y_offset += checkbox_size + label_gap + 5 # Spacing between checkboxes

        y_offset += element_gap # Space before bottom buttons

        # --- Solve / Reset Buttons (At the bottom of left panel) ---
        solve_reset_y = self.SCREEN_HEIGHT - self.STATUS_BAR_HEIGHT - element_h - self.PADDING
        solve_w = (full_width - label_gap) // 2
        self.ui_elements['solve_btn'] = {'type': 'button', 'text': "Giải", 'rect': pygame.Rect(x_offset, solve_reset_y, solve_w, element_h), 'action': 'solve'}
        self.ui_elements['reset_btn'] = {'type': 'button', 'text': "Reset", 'rect': pygame.Rect(x_offset + solve_w + label_gap, solve_reset_y, solve_w, element_h), 'action': 'reset'}

    def load_puzzle_data(self):
        """Loads data for the current puzzle selection."""
        difficulty = self.difficulty_levels[self.current_difficulty_index]
        puzzles_list = self.puzzles.get(difficulty, [])
        num_puzzles = len(puzzles_list)

        # Reset state before loading
        self.initial_grid_data = None
        self.colors_data = {}
        self.grid_size = 0
        self.current_puzzle_string = ""

        if num_puzzles == 0:
            self.current_puzzle_index = 0
            self.status_message = f"No puzzles for {difficulty}."
            print(f"Warning: No puzzles for difficulty '{difficulty}'")
        else:
            self.current_puzzle_index = self.current_puzzle_index % num_puzzles # Wrap index
            self.current_puzzle_string = puzzles_list[self.current_puzzle_index]
            self.status_message = f"Loading {difficulty} - P{self.current_puzzle_index + 1}..."
            print(self.status_message)
            try:
                grid, colors, size, _ = parse_puzzle_extended(self.current_puzzle_string)
                if grid is None or size <= 0:
                    raise ValueError("Parsed puzzle data is invalid.")
                self.initial_grid_data = grid
                self.colors_data = colors if colors is not None else {}
                self.grid_size = size
                if not self.colors_data:
                    self.status_message = f"Warning: No valid colors in {difficulty} - P{self.current_puzzle_index + 1}."
                else:
                    self.status_message = f"Loaded {difficulty} - P{self.current_puzzle_index + 1}. Ready."
            except ValueError as e:
                print(f"Error loading/parsing puzzle: {e}")
                traceback.print_exc()
                self.status_message = "Error loading puzzle."
                # Ensure state is cleared on error
                self.initial_grid_data = None
                self.colors_data = {}
                self.grid_size = 0

        # Always reset solver state after loading
        self.solver_results = {}
        self.successful_results_list = []
        self.app_state = "SETUP"
        self.is_solving = False

    def handle_input(self):
        """Processes user input events (mouse clicks in left panel)."""
        mouse_pos = pygame.mouse.get_pos()
        # Only allow interaction when in SETUP state
        interaction_enabled = (self.app_state == "SETUP")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_app()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # Left click
                # Check if click is within the left panel and interaction is enabled
                if interaction_enabled and mouse_pos[0] < self.LEFT_PANEL_WIDTH:
                    clicked_element_name = None
                    # Check UI elements in reverse order (top elements first)
                    for name, element in reversed(self.ui_elements.items()):
                        # Only check clickable types
                        if element['type'] in ['button', 'checkbox']:
                            # Check if element has a rect and is not explicitly disabled
                             if 'rect' in element and not element.get('disabled', False):
                                if element['rect'].collidepoint(mouse_pos):
                                    clicked_element_name = name
                                    break # Stop after finding the first hit

                    if clicked_element_name:
                        self.handle_click(clicked_element_name) # Process the click

    def handle_click(self, element_name):
        """Handles actions for UI element clicks."""
        element = self.ui_elements[element_name]
        action = element.get('action')
        print(f"Clicked: {element_name}, Action: {action}")

        # --- Navigation Actions ---
        if action == 'prev_diff':
            self.current_difficulty_index = (self.current_difficulty_index - 1) % len(self.difficulty_levels)
            self.current_puzzle_index = 0 # Reset puzzle index for new difficulty
            self.load_puzzle_data()
        elif action == 'next_diff':
            self.current_difficulty_index = (self.current_difficulty_index + 1) % len(self.difficulty_levels)
            self.current_puzzle_index = 0
            self.load_puzzle_data()
        elif action == 'prev_puzzle':
            difficulty = self.difficulty_levels[self.current_difficulty_index]
            num_puzzles = len(self.puzzles.get(difficulty, []))
            if num_puzzles > 0:
                self.current_puzzle_index = (self.current_puzzle_index - 1) % num_puzzles
                self.load_puzzle_data() # Reload the new puzzle
        elif action == 'next_puzzle':
            difficulty = self.difficulty_levels[self.current_difficulty_index]
            num_puzzles = len(self.puzzles.get(difficulty, []))
            if num_puzzles > 0:
                self.current_puzzle_index = (self.current_puzzle_index + 1) % num_puzzles
                self.load_puzzle_data() # Reload the new puzzle

        # --- Algorithm Selection ---
        elif action and action.startswith('toggle_'):
            algo_name = action.split('_')[1]
            if algo_name in self.selected_algorithms:
                 # Toggle the boolean value
                 self.selected_algorithms[algo_name] = not self.selected_algorithms[algo_name]
                 print(f"Algorithm {algo_name} selected: {self.selected_algorithms[algo_name]}")

        # --- Control Actions ---
        elif action == 'solve':
            self.start_solving()
        elif action == 'reset':
            self.reset_puzzle()

    def start_solving(self):
        """Initiates the solving process."""
        if self.is_solving:
            print("Solver already running.")
            return
        if not self.initial_grid_data or not self.colors_data:
            self.status_message = "Error: Load a valid puzzle first."
            print(self.status_message)
            # Consider showing a popup message here
            return

        selected_algos = [algo for algo, selected in self.selected_algorithms.items() if selected]
        if not selected_algos:
            self.status_message = "Error: Select at least one algorithm."
            print(self.status_message)
            # Consider showing a popup message here
            return

        print(f"Starting solve process for: {', '.join(selected_algos)}")
        self.is_solving = True
        self.solver_results = {} # Clear previous raw results
        self.successful_results_list = [] # Clear previous display list
        self.app_state = "SOLVING"
        self.status_message = f"Preparing to solve with {len(selected_algos)} algorithm(s)..."

        puzzle_string_copy = self.current_puzzle_string
        # Start the sequential solver in a separate thread
        self.solve_thread = threading.Thread(target=self.run_solvers_pygame,
                                             args=(selected_algos, puzzle_string_copy),
                                             daemon=True)
        self.solve_thread.start()

    def run_solvers_pygame(self, algorithms_to_run, puzzle_string_copy):
        """Runs solvers sequentially in the background thread."""
        num_algos = len(algorithms_to_run)
        for i, algo_name in enumerate(algorithms_to_run):
            # Optional: Check self.is_solving flag here to allow early exit if reset clicked
            # if not self.is_solving:
            #     print(f"Solver thread interrupted.")
            #     self.result_queue.put({'type': 'interrupted'})
            #     return

            # Update status via queue
            status_update = f"({i+1}/{num_algos}) Solving: {algo_name}..."
            self.result_queue.put({'type': 'status', 'message': status_update})
            print(status_update) # Also print to console

            solution_grid, solution_paths, solve_time, status = None, None, 0.0, "Execution Error"
            try:
                start_time = time.time()
                # Call the appropriate solver function
                if algo_name == "Backtracking": solution_grid, solution_paths, solve_time, status = solve_backtracking(puzzle_string_copy)
                elif algo_name == "BFS": solution_grid, solution_paths, solve_time, status = solve_bfs(puzzle_string_copy)
                elif algo_name == "A*": solution_grid, solution_paths, solve_time, status = solve_astar(puzzle_string_copy)
                elif algo_name == "CP":
                    if ORTOOLS_AVAILABLE: solution_grid, solution_paths, solve_time, status = solve_cp(puzzle_string_copy)
                    else: solve_time, status = 0.0, "OR-Tools N/A"
                else: status = "Invalid Algorithm"

                # Recalculate solve_time if solver didn't return it but finished
                current_solve_time = time.time() - start_time
                if solve_time < 0.0001 and status not in ["OR-Tools N/A", "Execution Error", "Invalid Algorithm", "Parse Error", "Setup Error"]:
                    solve_time = current_solve_time


            except Exception as e:
                print(f"CRITICAL ERROR during {algo_name} execution: {e}")
                traceback.print_exc()
                status = "Runtime Error"
                solve_time = time.time() - start_time # Time until error

            # Put result onto the queue for the main thread
            result_data = {
                'type': 'result', 'algo_name': algo_name,
                'grid': solution_grid, 'paths': solution_paths,
                'time': solve_time, 'status': status
            }
            self.result_queue.put(result_data)

        # Signal that all requested algorithms have finished
        self.result_queue.put({'type': 'finished'})
        print("Solver thread finished execution.")

    def check_solver_results(self):
        """Checks the queue for messages from the solver thread."""
        try:
            while True: # Process all available messages
                result = self.result_queue.get_nowait() # Non-blocking get

                msg_type = result.get('type')

                if msg_type == 'status':
                    self.status_message = result.get('message', 'Status update...')
                    # print(self.status_message) # Optional console log

                elif msg_type == 'result':
                    algo = result.get('algo_name', 'UnknownAlgo')
                    # Store the raw result regardless of success
                    self.solver_results[algo] = result
                    status = result.get('status', 'Unknown')
                    time_taken = result.get('time', 0.0)
                    print(f"Received result: {algo} - Status: {status} ({time_taken:.3f}s)")

                    # Add to display list only if successful
                    if status == "Success" and result.get('grid') is not None and result.get('paths') is not None:
                        # Prepare data needed for display
                        display_data = {
                            'grid': result['grid'],
                            'paths': result['paths'],
                            'time': time_taken,
                            'status': status
                        }
                        self.successful_results_list.append((algo, display_data))
                        # Keep sorted by time for consistent display order
                        self.successful_results_list.sort(key=lambda item: item[1]['time'])
                    elif status != "Success":
                        print(f"Solver {algo} did not succeed ({status}).")

                elif msg_type == 'finished':
                    self.is_solving = False # Solver thread is done
                    # Transition state based on results
                    self.app_state = "RESULTS" if self.successful_results_list else "SETUP"
                    # Generate final status message
                    success_count = len(self.successful_results_list)
                    total_run = len(self.solver_results)
                    fail_details = [ f"{name}({data.get('status', '?')})" for name, data in self.solver_results.items() if data.get('status') != 'Success' ]
                    self.status_message = f"Finished: {success_count}/{total_run} successful."
                    if fail_details:
                        self.status_message += f" Failed: {', '.join(fail_details)}"
                    print("All selected solvers finished processing.")

                # Optional: Handle other message types like 'interrupted' if needed

        except queue.Empty:
            pass # No messages in the queue right now

    def reset_puzzle(self):
        """Resets the application state to the currently selected puzzle."""
        if self.is_solving:
            print("Cannot reset while solving.")
            # Optionally add logic to signal the thread to stop if desired
            return

        print("Resetting puzzle display and state...")
        self.solver_results = {}
        self.successful_results_list = []
        self.app_state = "SETUP"
        # Force reload of the current puzzle to ensure clean state
        self.load_puzzle_data()
        # Update status after reload
        difficulty = self.difficulty_levels[self.current_difficulty_index]
        num_puzzles = len(self.puzzles.get(difficulty, []))
        if self.initial_grid_data: # Check if load was successful
             self.status_message = f"Reset to {difficulty} - P{self.current_puzzle_index + 1}."
        else:
             self.status_message = f"Reset. Error loading {difficulty} - P{self.current_puzzle_index + 1}."


    def draw_grid(self, surface, base_grid_data, paths_data, grid_rect, grid_size, is_small=False):
        """Draws the FlowFree grid onto a specified surface and rect."""
        # Basic validation
        if grid_size <= 0 or not base_grid_data or grid_rect.width < 2 or grid_rect.height < 2:
            return

        # Calculate cell size based on available space in the rect
        cell_w = grid_rect.width // grid_size
        cell_h = grid_rect.height // grid_size
        cell_size = min(cell_w, cell_h)
        if cell_size <= 0: return # Cannot draw cells if size is zero or less

        # --- Font Selection ---
        # Initialize small fonts dynamically if needed and cell size is small enough
        if is_small:
            # Reinitialize fonts if they don't exist or are too large for the cell
            if self.font_grid_small is None or self.font_grid_small.get_height() > cell_size * 0.7:
                font_size = max(6, int(cell_size * 0.45)) # Adjust multiplier as needed
                bold_font_size = max(7, int(cell_size * 0.55))
                try:
                    self.font_grid_small = pygame.font.SysFont("Consolas" if "Consolas" in pygame.font.get_fonts() else "Arial", font_size)
                    self.font_grid_bold_small = pygame.font.SysFont("Consolas" if "Consolas" in pygame.font.get_fonts() else "Arial", bold_font_size, bold=True)
                except pygame.error: # Fallback if dynamic init fails
                     self.font_grid_small = self.font_grid
                     self.font_grid_bold_small = self.font_grid_bold
            font_g, font_g_bold = self.font_grid_small, self.font_grid_bold_small
        else:
            font_g, font_g_bold = self.font_grid, self.font_grid_bold

        # Calculate rendering area centered within grid_rect
        grid_render_w = cell_size * grid_size
        grid_render_h = cell_size * grid_size
        start_x = grid_rect.left + max(0, (grid_rect.width - grid_render_w) // 2)
        start_y = grid_rect.top + max(0, (grid_rect.height - grid_render_h) // 2)

        # --- Prepare Display Grid ---
        # Start with the base grid (should have negative endpoints)
        display_grid = [row[:] for row in base_grid_data]
        # Overlay paths if provided
        if paths_data:
            for color, path_coords in paths_data.items():
                if isinstance(path_coords, (list, tuple)) and len(path_coords) > 1:
                    for i, coord in enumerate(path_coords):
                        if isinstance(coord, (list, tuple)) and len(coord) == 2:
                            r, c = coord
                            # Only draw path segment if cell is empty (0)
                            # And it's not the start/end point (which are negative)
                            if 0 < i < len(path_coords) - 1: # Intermediate points
                                if 0 <= r < grid_size and 0 <= c < grid_size and display_grid[r][c] == 0:
                                    display_grid[r][c] = color # Use positive color value
                        # else: Silently ignore invalid coords in path data

        # --- Draw Cells ---
        for r in range(grid_size):
            for c in range(grid_size):
                cell_rect = pygame.Rect(start_x + c * cell_size, start_y + r * cell_size, cell_size, cell_size)
                val = display_grid[r][c]
                abs_val = abs(val)
                text_char = ""
                # Default appearance (empty cell)
                bg_color_rgb = self.flow_colors_rgb[0]
                fg_color = self.TEXT_COLOR
                font_to_use = font_g
                border_width = 1
                border_color = self.GRID_BORDER_COLOR

                is_endpoint = val < 0
                is_path = val > 0

                if is_endpoint or is_path:
                    # Determine color index, handle potential modulo issues
                    color_index = abs_val % (len(self.flow_colors_rgb) - 1) + 1 if abs_val > 0 else 0
                    safe_idx = color_index % len(self.flow_colors_rgb)
                    # Ensure path/endpoints don't get the default background color index (0)
                    if abs_val > 0 and safe_idx == 0: safe_idx = 1
                    bg_color_rgb = self.flow_colors_rgb[safe_idx]

                    # Determine contrasting text color (simple brightness check)
                    brightness = (0.299 * bg_color_rgb.r + 0.587 * bg_color_rgb.g + 0.114 * bg_color_rgb.b)
                    fg_color = (255, 255, 255) if brightness < 135 else (0, 0, 0) # Adjusted threshold

                    if is_endpoint:
                        text_char = self.int_to_char.get(abs_val, '?') # Get '1', 'A', etc.
                        font_to_use = font_g_bold
                        border_width = 2 if not is_small and cell_size > 15 else 1 # Thicker border on larger endpoints

                # Draw cell background and border
                pygame.draw.rect(surface, bg_color_rgb, cell_rect)
                pygame.draw.rect(surface, border_color, cell_rect, border_width)

                # Draw endpoint character if present and cell is large enough
                if text_char and cell_size > 8:
                    try:
                        text_surf = font_to_use.render(text_char, True, fg_color)
                        text_rect = text_surf.get_rect(center=cell_rect.center)
                        # Optional: check if text fits before blitting
                        if text_rect.width < cell_rect.width - border_width*2 and text_rect.height < cell_rect.height - border_width*2 :
                             surface.blit(text_surf, text_rect)
                    except pygame.error as font_error:
                        print(f"Font render error: {font_error} (char: '{text_char}', size: {font_to_use.get_height()})")
                    except Exception as render_err:
                        print(f"Unexpected text render error: {render_err}")


    def draw_left_panel(self):
        """Draws the UI elements in the left panel."""
        panel_rect = pygame.Rect(0, 0, self.LEFT_PANEL_WIDTH, self.SCREEN_HEIGHT - self.STATUS_BAR_HEIGHT)
        pygame.draw.rect(self.screen, self.LEFT_PANEL_BG, panel_rect)
        # Vertical separator line
        pygame.draw.line(self.screen, self.GRID_BORDER_COLOR, (self.LEFT_PANEL_WIDTH - 1, 0), (self.LEFT_PANEL_WIDTH - 1, panel_rect.height), 1)

        mouse_pos = pygame.mouse.get_pos()
        can_interact = (self.app_state == "SETUP") # Interaction only in SETUP state

        # Draw all UI elements defined in _create_ui_elements
        for name, element in self.ui_elements.items():
            element_type = element['type']
            # Determine if element should be visually disabled
            disabled = element.get('disabled', False) or not can_interact

            if element_type == 'label':
                font = element.get('font', self.font_ui) # Use specified font or default
                text_color = self.TEXT_COLOR if not disabled else self.DISABLED_COLOR
                text_surf = font.render(element['text'], True, text_color)
                self.screen.blit(text_surf, element['pos'])

            elif element_type == 'textdisplay':
                rect = element['rect']
                 # Update dynamic text content
                if name == 'diff_value':
                    element['text'] = self.difficulty_levels[self.current_difficulty_index]
                elif name == 'puzzle_value':
                    difficulty = self.difficulty_levels[self.current_difficulty_index]
                    num_puzzles = len(self.puzzles.get(difficulty, []))
                    element['text'] = f"{self.current_puzzle_index + 1} / {num_puzzles}" if num_puzzles > 0 else "N/A"

                # Draw background/border
                pygame.draw.rect(self.screen, self.BG_COLOR, rect, border_radius=3) # Slightly different bg
                pygame.draw.rect(self.screen, self.GRID_BORDER_COLOR, rect, width=1, border_radius=3)
                # Draw centered text
                text_color = self.TEXT_COLOR # Text display doesn't get disabled look
                text_surf = self.font_ui.render(element['text'], True, text_color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

            elif element_type == 'button':
                rect = element['rect']; text = element['text']
                text_color = self.BUTTON_TEXT_COLOR if not disabled else self.DISABLED_COLOR
                border_col = self.GRID_BORDER_COLOR if not disabled else self.DISABLED_COLOR
                # Check hover only if enabled and mouse is within left panel
                is_hover = not disabled and rect.collidepoint(mouse_pos) and mouse_pos[0] < self.LEFT_PANEL_WIDTH
                current_bg = self.BUTTON_HOVER_COLOR if is_hover else self.BUTTON_COLOR

                pygame.draw.rect(self.screen, current_bg, rect, border_radius=4) # Rounded buttons
                pygame.draw.rect(self.screen, border_col, rect, width=1, border_radius=4)
                # Draw centered text
                text_surf = self.font_ui.render(text, True, text_color)
                text_rect = text_surf.get_rect(center=rect.center)
                self.screen.blit(text_surf, text_rect)

            elif element_type == 'checkbox':
                rect = element['rect']; text = element['text']; label_pos = element['label_pos']
                text_color = self.TEXT_COLOR if not disabled else self.DISABLED_COLOR
                border_col = self.CHECKBOX_BORDER_COLOR if not disabled else self.DISABLED_COLOR
                fill_col = self.CHECKBOX_FILL_COLOR if not disabled else self.DISABLED_COLOR

                # Draw checkbox border
                pygame.draw.rect(self.screen, border_col, rect, width=1, border_radius=2)
                # Draw check if selected
                algo_name = element['action'].split('_')[1] # Extract algo name from action
                if algo_name in self.selected_algorithms and self.selected_algorithms[algo_name]:
                    # Draw smaller inner rectangle as checkmark
                    inner_rect = rect.inflate(-8, -8) # Adjust inflation for check size
                    pygame.draw.rect(self.screen, fill_col, inner_rect, border_radius=1)
                # Draw label text
                label_surf = self.font_ui.render(text, True, text_color)
                # Simple clip if too wide (can be improved)
                max_w = element.get('text_rect_w', self.LEFT_PANEL_WIDTH)
                if label_surf.get_width() > max_w:
                     try: # Protect against potential division by zero or tiny widths
                         clip_ratio = max_w / label_surf.get_width()
                         clip_len = max(1, int(len(text) * clip_ratio) - 3) # Ensure at least 1 char + ...
                         clipped_text = text[:clip_len] + "..."
                         label_surf = self.font_ui.render(clipped_text, True, text_color)
                     except Exception: pass # Ignore clipping errors

                label_rect = label_surf.get_rect(midleft=label_pos) # Align left edge vertically centered
                self.screen.blit(label_surf, label_rect)

    def draw_right_panel(self):
        """Draws the content of the right panel (grid or results)."""
        panel_rect = pygame.Rect(
            self.LEFT_PANEL_WIDTH + self.PANEL_GAP, 0,
            self.RIGHT_PANEL_WIDTH, self.SCREEN_HEIGHT - self.STATUS_BAR_HEIGHT
        )
        pygame.draw.rect(self.screen, self.RIGHT_PANEL_BG, panel_rect)

        # Define the main drawing area within the right panel
        content_area_rect = panel_rect.inflate(-self.PADDING * 2, -self.PADDING * 2)

        if self.app_state == "SETUP":
            if self.initial_grid_data and self.grid_size > 0:
                # Draw the large initial puzzle grid
                self.draw_grid(self.screen, self.initial_grid_data, None, content_area_rect, self.grid_size, is_small=False)
            else:
                 # Display a message if no puzzle is loaded
                 placeholder_text = self.font_ui_label.render("Load Puzzle", True, self.TEXT_COLOR)
                 placeholder_rect = placeholder_text.get_rect(center=content_area_rect.center)
                 self.screen.blit(placeholder_text, placeholder_rect)

        elif self.app_state == "SOLVING":
            # Show the initial grid dimly with "Solving..." overlay
            if self.initial_grid_data and self.grid_size > 0:
                self.draw_grid(self.screen, self.initial_grid_data, None, content_area_rect, self.grid_size, is_small=False)

            # Draw semi-transparent overlay
            overlay_surface = pygame.Surface(content_area_rect.size, pygame.SRCALPHA)
            overlay_surface.fill(self.SOLVING_OVERLAY_COLOR)
            self.screen.blit(overlay_surface, content_area_rect.topleft)

            # Draw "Solving..." text
            solve_text = self.font_ui_label.render("ĐANG GIẢI...", True, self.ERROR_TEXT_COLOR)
            solve_rect = solve_text.get_rect(center=content_area_rect.center)
            self.screen.blit(solve_text, solve_rect)

        elif self.app_state == "RESULTS":
            if self.successful_results_list:
                # Draw multiple small result grids
                self.draw_multiple_results(content_area_rect)
            else:
                 # If no successful results, show initial grid + error message
                 if self.initial_grid_data and self.grid_size > 0:
                     self.draw_grid(self.screen, self.initial_grid_data, None, content_area_rect, self.grid_size, is_small=False)

                 no_res_text = self.font_ui_label.render("Không tìm thấy lời giải hợp lệ", True, self.ERROR_TEXT_COLOR)
                 no_res_rect = no_res_text.get_rect(center=content_area_rect.center)
                 # Optional: Draw a faint background behind the text for readability
                 # text_bg_rect = no_res_rect.inflate(10, 5)
                 # pygame.draw.rect(self.screen, self.RIGHT_PANEL_BG, text_bg_rect, border_radius=3)
                 self.screen.blit(no_res_text, no_res_rect)

    def draw_multiple_results(self, area_rect):
        """Calculates layout and draws multiple small result grids."""
        results_to_draw = self.successful_results_list
        num_results = len(results_to_draw)
        if num_results == 0 or self.grid_size <= 0 or area_rect.width < 10 or area_rect.height < 10:
             return # Cannot draw

        # --- Calculate Grid Layout (Attempt to fit) ---
        cols, rows = 1, 1
        if num_results > 1:
            # Heuristic to determine cols/rows based on area aspect ratio
            aspect_ratio = area_rect.width / area_rect.height
            cols = max(1, int(round(math.sqrt(num_results * aspect_ratio))))
            rows = max(1, math.ceil(num_results / cols))

            # Adjust if layout is too sparse or too cramped (optional refinement)
            # Example: if rows * cols > num_results + cols: try fewer cols?

        # --- Calculate Cell Dimensions ---
        # Total space used by gaps between cells
        total_gap_x = max(0, (cols - 1)) * self.PADDING
        total_gap_y = max(0, (rows - 1)) * self.PADDING

        # Available width/height per cell (grid + label + internal padding)
        cell_container_w = (area_rect.width - total_gap_x) / cols
        cell_container_h = (area_rect.height - total_gap_y) / rows

        # Estimate fixed height for the 2-line info label
        info_label_h = self.font_result_info.get_height() * 2 + 8 # Height of 2 lines + small gap

        # Calculate available space *just for the grid visualization*
        grid_max_w = max(1, cell_container_w - 10) # Allow small padding within cell
        grid_max_h = max(1, cell_container_h - info_label_h - 10) # Space above label

        if grid_max_w <= 1 or grid_max_h <= 1:
             print("Warning: Not enough space per cell to draw result grids.")
             # Optionally draw a message instead of trying to draw tiny grids
             warn_text = self.font_ui.render(f"Không đủ chỗ để vẽ {num_results} kết quả.", True, self.TEXT_COLOR)
             warn_rect = warn_text.get_rect(center=area_rect.center)
             self.screen.blit(warn_text, warn_rect)
             return

        # --- Draw Each Result Grid and Info ---
        for index, (algo_name, result_data) in enumerate(results_to_draw):
            current_col = index % cols
            current_row = index // cols

            # Calculate top-left corner of the *container* for this result
            cell_x = area_rect.left + current_col * (cell_container_w + self.PADDING)
            cell_y = area_rect.top + current_row * (cell_container_h + self.PADDING)

            # Define the Rect where the small grid itself will be drawn
            # Center the grid drawing area within the available space above the label
            grid_draw_area_x = cell_x + (cell_container_w - grid_max_w) // 2
            grid_draw_area_y = cell_y # Start grid at the top of its vertical space
            small_grid_rect = pygame.Rect(grid_draw_area_x, grid_draw_area_y, grid_max_w, grid_max_h)

            # Draw the grid using the solution data
            # Pass the base grid (with negative endpoints) and the paths dict
            self.draw_grid(self.screen,
                           result_data['grid'],
                           result_data['paths'],
                           small_grid_rect,
                           self.grid_size,
                           is_small=True) # Signal to use smaller fonts/borders

            # Draw Algorithm Name and Time/Status below the grid
            info_y_start = small_grid_rect.bottom + 6 # Position below the drawn grid
            info_text_line1 = f"{algo_name}"
            status_str = str(result_data.get('status', 'N/A'))
            info_text_line2 = f"({result_data.get('time', 0.0):.3f}s) {status_str}"

            try:
                info_surf1 = self.font_result_info.render(info_text_line1, True, self.TEXT_COLOR)
                info_surf2 = self.font_result_info.render(info_text_line2, True, self.TEXT_COLOR)

                # Center the text lines horizontally within the cell container width
                info_center_x = cell_x + cell_container_w / 2
                info_rect1 = info_surf1.get_rect(midtop=(info_center_x, info_y_start))
                info_rect2 = info_surf2.get_rect(midtop=(info_center_x, info_rect1.bottom + 2)) # Small gap

                # Only draw if it fits vertically within the cell container
                if info_rect2.bottom < cell_y + cell_container_h:
                     self.screen.blit(info_surf1, info_rect1)
                     self.screen.blit(info_surf2, info_rect2)
            except pygame.error as font_err:
                 print(f"Error rendering result info font: {font_err}")
            except Exception as e:
                 print(f"Error drawing result info: {e}")


    def draw_status_bar(self):
        """Draws the bottom status bar."""
        status_rect = pygame.Rect(0, self.SCREEN_HEIGHT - self.STATUS_BAR_HEIGHT, self.SCREEN_WIDTH, self.STATUS_BAR_HEIGHT)
        pygame.draw.rect(self.screen, self.STATUS_BG_COLOR, status_rect)
        # Divider line
        pygame.draw.line(self.screen, (80, 80, 80), (0, status_rect.top), (self.SCREEN_WIDTH, status_rect.top), 1)

        # Render status text, handle potential truncation if too long
        status_text = self.status_message
        try:
            status_surf = self.font_status.render(status_text, True, self.STATUS_TEXT_COLOR)
            max_status_width = self.SCREEN_WIDTH - 2 * self.PADDING
            if status_surf.get_width() > max_status_width:
                # Basic truncation logic
                avg_char_width = status_surf.get_width() / max(1, len(status_text))
                max_chars = int(max_status_width / max(1, avg_char_width)) - 3 # Account for "..."
                if max_chars < 1: max_chars = 1
                status_text = status_text[:max_chars] + "..."
                status_surf = self.font_status.render(status_text, True, self.STATUS_TEXT_COLOR)

            status_text_rect = status_surf.get_rect(centery=status_rect.centery, left=self.PADDING)
            self.screen.blit(status_surf, status_text_rect)
        except pygame.error as font_err:
            print(f"Error rendering status font: {font_err}")
        except Exception as e:
            print(f"Error drawing status bar: {e}")


    def draw(self):
        """Main drawing function, orchestrates drawing panels and status bar."""
        self.screen.fill(self.BG_COLOR) # Clear screen
        self.draw_left_panel()
        self.draw_right_panel()
        self.draw_status_bar()
        pygame.display.flip() # Update the display

    def run(self):
        """Main application loop."""
        running = True
        while running:
            # 1. Handle user input
            self.handle_input()

            # 2. Check for results from background thread if solving
            if self.is_solving:
                self.check_solver_results()

            # 3. Update game state (if needed, e.g., animations - none currently)

            # 4. Draw everything based on current state
            self.draw()

            # 5. Control frame rate
            self.clock.tick(30) # Aim for 30 FPS

            # Check if the display is still active (handles closing the window)
            if not pygame.display.get_init():
                running = False # Exit loop if display closed

    def quit_app(self):
        """Cleans up and exits the application."""
        print("Exiting Flow Free Solver...")
        self.is_solving = False # Attempt to signal thread, though daemon=True handles exit
        pygame.quit()
        sys.exit()

# ============================================================
# RUN APPLICATION
# ============================================================
if __name__ == "__main__":
    # Ensure all required solver functions are defined above this point!
    print("Initializing Flow Free Solver (Pygame)...")
    app = FlowFreePygameApp()
    app.run()