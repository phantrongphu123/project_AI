import tkinter as tk
from tkinter import ttk, messagebox
import copy
import time
import threading
import traceback
import heapq
from collections import deque
import itertools
import argparse
import csv
import os
import random
import sys
import math

# --- Kiểm tra và Import Matplotlib ---
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    FigureCanvasTkAgg = None
    plt = None

# --- Kiểm tra và Import OR-Tools ---
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    cp_model = None
    ORTOOLS_AVAILABLE = False

BENCHMARK_MODE = False

# ============================================================
# HÀM TIỆN ÍCH VÀ PARSE
# ============================================================
def get_neighbors(r, c, size):
    neighbors = []
    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < size and 0 <= nc < size:
            neighbors.append((nr, nc))
    return neighbors

def parse_puzzle_extended(puzzle_str):
    global BENCHMARK_MODE
    lines = [line.strip() for line in puzzle_str.strip().split('\n') if line.strip()]
    if not lines: return None, None, 0, 0
    try:
        size = len(lines[0])
        if size == 0: return None, None, 0, 0
    except IndexError: return None, None, 0, 0

    grid = [[0] * size for _ in range(size)]
    points = {}
    max_color_id = 0
    char_to_int = {str(i): i for i in range(1, 10)}
    for i, char_code in enumerate(range(ord('A'), ord('Z') + 1)):
        char_to_int[chr(char_code)] = 10 + i

    for r, row_str in enumerate(lines):
        if len(row_str) != size: return None, None, 0, 0
        for c, char in enumerate(row_str):
            if char == '.': grid[r][c] = 0
            elif char in char_to_int:
                color = char_to_int[char]
                if color <= 0: grid[r][c] = 0; continue
                if color not in points: points[color] = []
                points[color].append((r, c))
                grid[r][c] = -color
                max_color_id = max(max_color_id, color)
            else: grid[r][c] = 0

    colors_data = {}
    valid_colors_found = False
    invalid_colors = []
    for color, coords in points.items():
        if len(coords) == 2:
            if coords[0] == coords[1]: invalid_colors.append(color)
            else:
                colors_data[color] = {'start': coords[0], 'end': coords[1], 'path': [coords[0]], 'complete': False}
                valid_colors_found = True
        else: invalid_colors.append(color)

    for color in invalid_colors:
        if color in points:
             for r_err, c_err in points[color]:
                 if 0 <= r_err < size and 0 <= c_err < size and grid[r_err][c_err] == -color:
                     grid[r_err][c_err] = 0

    if not valid_colors_found and points and not BENCHMARK_MODE:
        if not BENCHMARK_MODE: print("Cảnh báo: Không tìm thấy cặp điểm màu hợp lệ nào.")

    final_max_color_id = max(colors_data.keys()) if colors_data else 0
    return grid, colors_data, size, final_max_color_id


def reconstruct_paths(solution_grid, colors_data, size):
    global BENCHMARK_MODE
    if not solution_grid or not colors_data or size <= 0: return {}
    reconstructed = {}
    grid_copy = [row[:] for row in solution_grid]

    max_abs_val_in_grid = 0
    if any(solution_grid):
        for row in solution_grid:
            for cell in row:
                if cell != 0:
                    max_abs_val_in_grid = max(max_abs_val_in_grid, abs(cell))
    marker_base = max_abs_val_in_grid + 100


    for color, data in colors_data.items():
        start_node = data['start']
        end_node = data['end']
        if not (0 <= start_node[0] < size and 0 <= start_node[1] < size and \
                0 <= end_node[0] < size and 0 <= end_node[1] < size):
            if not BENCHMARK_MODE: print(f"Lỗi reconstruct: Endpoint màu {color} ngoài grid.")
            continue

        path = [start_node]
        current_r, current_c = start_node
        marker = marker_base + color
        if 0 <= current_r < size and 0 <= current_c < size:
             grid_copy[current_r][current_c] = marker
        else:
            if not BENCHMARK_MODE: print(f"Lỗi reconstruct: Startpoint màu {color} {start_node} ngoài grid.")
            continue

        found_path = False
        path_construction_attempts = 0
        max_attempts = size * size * 2

        while (current_r, current_c) != end_node and path_construction_attempts < max_attempts :
            path_construction_attempts +=1
            found_next = False
            neighbors = get_neighbors(current_r, current_c, size)
            neighbors.sort(key=lambda pos: manhattan_distance(pos, end_node))

            for nr, nc in neighbors:
                cell_val = grid_copy[nr][nc]
                is_correct_color_path = (cell_val == color)
                is_correct_color_end = (cell_val == -color and (nr,nc) == end_node)
                is_not_visited_marker = (cell_val != marker)

                if (is_correct_color_path or is_correct_color_end) and is_not_visited_marker:
                    path.append((nr, nc))
                    grid_copy[nr][nc] = marker
                    current_r, current_c = nr, nc
                    found_next = True
                    if (current_r, current_c) == end_node: found_path = True
                    break
            if not found_next:
                break

        if path_construction_attempts >= max_attempts and not found_path:
             if not BENCHMARK_MODE: print(f"Cảnh báo reconstruct: Màu {color} vượt quá số lần thử xây dựng path.")


        if found_path:
            reconstructed[color] = path

    if len(reconstructed) != len(colors_data):
        if not BENCHMARK_MODE:
            pass
    return reconstructed

def is_grid_full(grid, size):
    if not grid or size == 0: return False
    for r in range(size):
        for c in range(size):
            if grid[r][c] == 0:
                return False
    return True

def get_path_data_copy(paths_dict):
    return copy.deepcopy(paths_dict)

def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# ============================================================
# HEURISTIC FUNCTIONS FOR A*
# ============================================================
def h_manhattan_sum(paths_dict, color_order, size, grid_tuple=None):
    h = 0
    for color in color_order:
        data = paths_dict.get(color)
        if data and not data['complete']:
            current_head = data['coords'][-1]
            target = data['target']
            h += manhattan_distance(current_head, target)
    return h

def h_manhattan_max(paths_dict, color_order, size, grid_tuple=None):
    h = 0
    for color in color_order:
        data = paths_dict.get(color)
        if data and not data['complete']:
            current_head = data['coords'][-1]
            target = data['target']
            h = max(h, manhattan_distance(current_head, target))
    return h

def h_manhattan_avg_plus_incomplete(paths_dict, color_order, size, grid_tuple=None):
    total_manhattan = 0
    incomplete_count = 0
    active_colors = 0
    for color in color_order:
        data = paths_dict.get(color)
        if data and not data['complete']:
            current_head = data['coords'][-1]
            target = data['target']
            total_manhattan += manhattan_distance(current_head, target)
            incomplete_count += 1
            active_colors +=1
    avg_manhattan = (total_manhattan / active_colors) if active_colors > 0 else 0
    return avg_manhattan + incomplete_count * (size)

AVAILABLE_HEURISTICS = {
    "Manhattan Sum": h_manhattan_sum,
    "Manhattan Max": h_manhattan_max,
    "Manhattan Avg + Incomplete Penalty": h_manhattan_avg_plus_incomplete,
}
AVAILABLE_QLEARNING_CONFIGS = {
    "Default": {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 0.3, "episodes": 500},
    "Exploratory": {"learning_rate": 0.1, "discount_factor": 0.9, "exploration_rate": 0.5, "episodes": 750},
    "Conservative": {"learning_rate": 0.05, "discount_factor": 0.95, "exploration_rate": 0.2, "episodes": 1000},
    "Aggressive": {"learning_rate": 0.2, "discount_factor": 0.85, "exploration_rate": 0.4, "episodes": 400},
    "Balanced": {"learning_rate": 0.15, "discount_factor": 0.9, "exploration_rate": 0.35, "episodes": 600},
}
# ============================================================
# THUẬT TOÁN GIẢI
# ============================================================
def solve_simulated_annealing(puzzle_str, time_limit=60.0, initial_temp=100.0, cooling_rate=0.95, iterations_per_temp=100):
    global BENCHMARK_MODE
    start_time_total = time.time()
    solution_grid = None
    solution_paths_dict = None
    states_explored = 0

    try:
        initial_grid, colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid is None or not colors_data or size == 0:
            raise ValueError("Invalid puzzle data")
    except ValueError as e:
        if not BENCHMARK_MODE: print(f"Lỗi parse puzzle cho Simulated Annealing: {e}")
        return None, None, time.time() - start_time_total, 0

    clean_grid = [[0] * size for _ in range(size)]
    for color, data in colors_data.items():
        sr, sc = data['start']
        er, ec = data['end']
        clean_grid[sr][sc] = -color
        clean_grid[er][ec] = -color

    current_grid = [row[:] for row in clean_grid]
    color_order = sorted(colors_data.keys())
    current_paths = {}

    for color in color_order:
        start = colors_data[color]['start']
        end = colors_data[color]['end']
        current_paths[color] = {'coords': [start], 'target': end, 'complete': False}

    def calculate_energy(grid, paths):
        energy = 0
        for r in range(size):
            for c in range(size):
                if grid[r][c] == 0:
                    energy += 1
        for color, path_data in paths.items():
            if not path_data['complete']:
                current_head = path_data['coords'][-1]
                target = path_data['target']
                energy += manhattan_distance(current_head, target) * 2
        return energy

    def generate_neighbor(grid, paths):
        new_grid = [row[:] for row in grid]
        new_paths = get_path_data_copy(paths)
        incomplete_colors = [c for c, p in new_paths.items() if not p['complete']]

        if not incomplete_colors:
            if random.random() < 0.3:
                color = random.choice(list(new_paths.keys()))
                path = new_paths[color]['coords']
                if len(path) > 2:
                    idx = random.randint(1, len(path) - 2)
                    removed_point = path[idx]
                    new_grid[removed_point[0]][removed_point[1]] = 0
                    new_paths[color]['coords'].pop(idx)
            return new_grid, new_paths

        color = random.choice(incomplete_colors)
        path_data = new_paths[color]
        current_head = path_data['coords'][-1]
        target = path_data['target']
        neighbors = get_neighbors(current_head[0], current_head[1], size)
        valid_moves = []

        for nr, nc in neighbors:
            is_target = (nr, nc) == target
            cell_value = new_grid[nr][nc]
            is_valid = (cell_value == 0 or (is_target and cell_value == -color))
            if (nr, nc) not in path_data['coords'] and is_valid:
                valid_moves.append((nr, nc))

        if not valid_moves:
            if len(path_data['coords']) > 1:
                last_r, last_c = path_data['coords'].pop()
                if not ((last_r, last_c) == path_data['target'] or (last_r, last_c) == path_data['coords'][0]):
                    new_grid[last_r][last_c] = 0
            return new_grid, new_paths

        next_r, next_c = random.choice(valid_moves)
        is_target_move = (next_r, next_c) == target
        path_data['coords'].append((next_r, next_c))
        if is_target_move:
            path_data['complete'] = True
        else:
            new_grid[next_r][next_c] = color
        return new_grid, new_paths

    current_energy = calculate_energy(current_grid, current_paths)
    best_energy = current_energy
    best_grid = [row[:] for row in current_grid]
    best_paths = get_path_data_copy(current_paths)
    temperature = initial_temp

    if not BENCHMARK_MODE: print("Giải bằng Simulated Annealing...")

    while temperature > 0.1 and time.time() - start_time_total < time_limit:
        for i in range(iterations_per_temp):
            states_explored += 1
            if time.time() - start_time_total >= time_limit:
                break
            neighbor_grid, neighbor_paths = generate_neighbor(current_grid, current_paths)
            neighbor_energy = calculate_energy(neighbor_grid, neighbor_paths)
            delta_energy = neighbor_energy - current_energy

            if delta_energy < 0:
                current_grid = neighbor_grid
                current_paths = neighbor_paths
                current_energy = neighbor_energy
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_grid = [row[:] for row in current_grid]
                    best_paths = get_path_data_copy(current_paths)
                    if best_energy == 0 or (is_grid_full(best_grid, size) and
                                          all(path['complete'] for path in best_paths.values())):
                        if not BENCHMARK_MODE:
                            print(f"SA: Found complete solution with energy {best_energy}")
                        temperature = 0
                        break
            else:
                acceptance_prob = math.exp(-delta_energy / temperature)
                if random.random() < acceptance_prob:
                    current_grid = neighbor_grid
                    current_paths = neighbor_paths
                    current_energy = neighbor_energy
        temperature *= cooling_rate
        if not BENCHMARK_MODE and states_explored % 1000 == 0:
            print(f"SA: Temp={temperature:.2f}, Energy={current_energy}, Best={best_energy}, States={states_explored}")

    solve_time = time.time() - start_time_total
    if best_energy == 0 or (is_grid_full(best_grid, size) and all(path['complete'] for path in best_paths.values())):
        solution_grid = best_grid
        solution_paths_dict = {color: data['coords'] for color, data in best_paths.items()}
        if not BENCHMARK_MODE:
            print(f"SA: Solution found in {solve_time:.4f} seconds with {states_explored} states explored")
    else:
        if all(path['complete'] for path in best_paths.values()):
            solution_grid = best_grid
            solution_paths_dict = {color: data['coords'] for color, data in best_paths.items()}
            if not BENCHMARK_MODE:
                print(f"SA: Partial solution found in {solve_time:.4f} seconds with {states_explored} states")
        else:
            if not BENCHMARK_MODE:
                print(f"SA: No solution found in {solve_time:.4f} seconds")
    return solution_grid, solution_paths_dict, solve_time, states_explored
def solve_and_or_search(puzzle_str, time_limit=60.0):
    global BENCHMARK_MODE
    start_time_total = time.time()
    solution_grid = None
    solution_paths_dict = None
    states_explored = 0
    
    try:
        initial_grid, colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid is None or not colors_data or size == 0:
            raise ValueError("Invalid puzzle data")
    except ValueError as e:
        if not BENCHMARK_MODE: print(f"Lỗi parse puzzle cho AND-OR Search: {e}")
        return None, None, time.time() - start_time_total, 0
    
    color_order = list(colors_data.keys())
    memo = {}  # Memoization cache
    
    # Khởi tạo paths
    initial_paths = {c: {'coords': [colors_data[c]['start']], 
                         'target': colors_data[c]['end'], 
                         'complete': False} 
                    for c in color_order}
    
    def and_or_search(grid, unfinished_colors, current_paths):
        nonlocal states_explored, start_time_total
        states_explored += 1
        
        if time.time() - start_time_total > time_limit:
            return None, None
        
        # Nếu tất cả màu đã hoàn thành
        if not unfinished_colors:
            if is_grid_full(grid, size):
                return grid, current_paths
            return None, None
        
        # Memoization key
        grid_tuple = tuple(map(tuple, grid))
        unfinished_tuple = tuple(sorted(unfinished_colors))
        memo_key = (grid_tuple, unfinished_tuple)
        
        if memo_key in memo:
            return memo[memo_key]
        
        # Nút OR: Chọn màu để mở rộng
        # Sắp xếp theo khoảng cách Manhattan còn lại
        colors_to_try = sorted(unfinished_colors, 
                              key=lambda c: manhattan_distance(current_paths[c]['coords'][-1], 
                                                            current_paths[c]['target']))
        
        for color in colors_to_try:
            path_data = current_paths[color]
            current_head = path_data['coords'][-1]
            target = path_data['target']
            
            # Nút OR: Thử các hướng di chuyển có thể
            neighbors = sorted(
                get_neighbors(current_head[0], current_head[1], size),
                key=lambda pos: manhattan_distance(pos, target)
            )
            
            for nr, nc in neighbors:
                # Kiểm tra điều kiện di chuyển hợp lệ
                is_target = (nr, nc) == target
                cell_value = grid[nr][nc]
                is_valid_empty = (cell_value == 0)
                is_valid_target = (is_target and cell_value == -color)
                is_valid_move = is_valid_empty or is_valid_target
                already_visited = (nr, nc) in path_data['coords']
                
                if is_valid_move and not already_visited:
                    # Tạo trạng thái mới
                    new_grid = [row[:] for row in grid]
                    new_paths = copy.deepcopy(current_paths)
                    
                    # Cập nhật grid và path
                    if is_valid_empty:
                        new_grid[nr][nc] = color
                    new_paths[color]['coords'].append((nr, nc))
                    
                    # Nếu đã đến đích, đánh dấu màu này đã hoàn thành
                    new_unfinished = set(unfinished_colors)
                    if is_target:
                        new_paths[color]['complete'] = True
                        new_unfinished.remove(color)
                    
                    # Nút AND: Tất cả các trạng thái con phải thành công
                    result_grid, result_paths = and_or_search(new_grid, new_unfinished, new_paths)
                    
                    if result_grid:
                        memo[memo_key] = (result_grid, result_paths)
                        return result_grid, result_paths
        
        # Không tìm thấy giải pháp
        memo[memo_key] = (None, None)
        return None, None
    
    # Bắt đầu thuật toán AND-OR Search
    if not BENCHMARK_MODE: print("Giải bằng AND-OR Search...")
    solution_grid, solution_paths_internal = and_or_search(initial_grid, set(color_order), initial_paths)
    
    # Xử lý kết quả
    if solution_grid and solution_paths_internal:
        solution_paths_dict = {color: data['coords'] for color, data in solution_paths_internal.items()}
    
    solve_time = time.time() - start_time_total
    if not BENCHMARK_MODE: print(f"Thời gian AND-OR Search: {solve_time:.4f} giây, Trạng thái: {states_explored}")
    
    return solution_grid, solution_paths_dict, solve_time, states_explored
def solve_qlearning(puzzle_str, time_limit=60.0, episodes=500, config=None):
    global BENCHMARK_MODE
    start_time_total = time.time()
    solution_grid = None
    solution_paths_dict = None
    states_explored = 0

    try:
        initial_grid, colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid is None or not colors_data or size == 0:
            raise ValueError("Invalid puzzle data")
    except ValueError as e:
        if not BENCHMARK_MODE: print(f"Lỗi parse puzzle cho Q-learning: {e}")
        return None, None, time.time() - start_time_total, 0

    effective_config = {}
    if config is None:
        effective_config = AVAILABLE_QLEARNING_CONFIGS["Default"]
    else:
        effective_config = config

    learning_rate = effective_config.get("learning_rate", 0.1)
    discount_factor = effective_config.get("discount_factor", 0.9)
    exploration_rate = effective_config.get("exploration_rate", 0.3)
    episodes_to_run = effective_config.get("episodes", episodes)
    color_order = sorted(colors_data.keys())
    q_table = {}

    def state_to_key(grid, active_color_idx):
        grid_flat = tuple(cell for row in grid for cell in row)
        return (grid_flat, active_color_idx)

    def get_possible_actions(grid, paths_dict, active_color):
        if active_color not in paths_dict or paths_dict[active_color]['complete']:
            return []
        current_head = paths_dict[active_color]['coords'][-1]
        target = paths_dict[active_color]['target']
        neighbors = get_neighbors(current_head[0], current_head[1], size)
        valid_actions = []
        for nr, nc in neighbors:
            is_target = (nr, nc) == target
            is_valid_empty_cell = (grid[nr][nc] == 0)
            is_valid_target_cell = (is_target and (grid[nr][nc] == -active_color or grid[nr][nc] == 0) )
            is_valid_move_to_empty = (is_valid_empty_cell and (nr, nc) not in paths_dict[active_color]['coords'])
            if is_valid_target_cell or is_valid_move_to_empty:
                 valid_actions.append((nr, nc))
        return valid_actions

    def compute_reward(grid, paths_dict, active_color, next_r, next_c, is_final_move_for_color, all_colors_done):
        reward = -0.1
        if is_final_move_for_color:
            reward += 10.0
            if all_colors_done and is_grid_full(grid, size):
                reward += 100.0
            elif all_colors_done and not is_grid_full(grid,size):
                reward -= 20
        elif (next_r, next_c) in paths_dict[active_color]['coords'][:-1]:
            reward -= 5.0
        return reward

    def select_action(state_key, possible_actions):
        if not possible_actions:
            return None
        if random.random() < exploration_rate or state_key not in q_table:
            return random.choice(possible_actions)
        else:
            for action in possible_actions:
                if action not in q_table[state_key]:
                    q_table[state_key][action] = 0.0
            action_values = {action: q_table[state_key].get(action, 0.0) for action in possible_actions}
            max_q_val = max(action_values.values())
            best_actions = [action for action, q_val in action_values.items() if q_val == max_q_val]
            return random.choice(best_actions) if best_actions else random.choice(possible_actions)

    if not BENCHMARK_MODE: print(f"Giải bằng Q-Learning (LR={learning_rate}, DF={discount_factor}, ER={exploration_rate}, EP={episodes_to_run})...")

    initial_q_grid = [[0] * size for _ in range(size)]
    for c_id, data in colors_data.items():
        sr, sc = data['start']
        er, ec = data['end']
        initial_q_grid[sr][sc] = -c_id
        initial_q_grid[er][ec] = -c_id

    best_grid_overall = None
    best_paths_overall = None
    max_reward_achieved_overall = float('-inf')
    best_episode_num = -1

    for episode in range(episodes_to_run):
        if time.time() - start_time_total > time_limit:
            if not BENCHMARK_MODE:
                print(f"Q-Learning: Đã vượt quá giới hạn thời gian ({time_limit}s) ở episode {episode+1}.")
            break
        current_grid = [row[:] for row in initial_q_grid]
        current_paths = {c: {'coords': [colors_data[c]['start']],
                             'target': colors_data[c]['end'],
                             'complete': False}
                         for c in color_order}
        active_color_idx = 0
        total_reward_episode = 0
        episode_path_valid = True
        num_steps_in_episode = 0
        max_steps_per_episode = size * size * len(color_order)

        while active_color_idx < len(color_order) and num_steps_in_episode < max_steps_per_episode :
            states_explored += 1
            num_steps_in_episode +=1
            active_color = color_order[active_color_idx]
            if current_paths[active_color]['complete']:
                active_color_idx += 1
                continue
            state_key = state_to_key(current_grid, active_color_idx)
            possible_actions = get_possible_actions(current_grid, current_paths, active_color)
            if not possible_actions:
                episode_path_valid = False
                break
            action = select_action(state_key, possible_actions)
            next_r, next_c = action
            is_target_move = (next_r, next_c) == current_paths[active_color]['target']
            all_colors_will_be_complete = False

            if is_target_move:
                current_paths[active_color]['coords'].append((next_r, next_c))
                current_paths[active_color]['complete'] = True
                all_colors_will_be_complete = all(path['complete'] for path_idx, path in enumerate(current_paths.values()) if path_idx <= active_color_idx) and \
                                              (active_color_idx == len(color_order) -1)
            else:
                current_grid[next_r][next_c] = active_color
                current_paths[active_color]['coords'].append((next_r, next_c))

            reward = compute_reward(current_grid, current_paths, active_color, next_r, next_c, is_target_move, all_colors_will_be_complete)
            total_reward_episode += reward
            next_active_color_idx_for_q = active_color_idx
            if is_target_move:
                next_active_color_idx_for_q +=1
            next_state_key = state_to_key(current_grid, next_active_color_idx_for_q)
            if state_key not in q_table: q_table[state_key] = {}
            if action not in q_table[state_key]: q_table[state_key][action] = 0.0
            max_next_q = 0.0
            if next_active_color_idx_for_q < len(color_order):
                next_color_for_q = color_order[next_active_color_idx_for_q]
                next_possible_actions = get_possible_actions(current_grid, current_paths, next_color_for_q)
                if next_state_key in q_table and next_possible_actions:
                    for next_act in next_possible_actions:
                        if next_act not in q_table[next_state_key]:
                            q_table[next_state_key][next_act] = 0.0
                    max_next_q = max([q_table[next_state_key].get(a, 0.0) for a in next_possible_actions], default=0.0)
            old_q = q_table[state_key][action]
            q_table[state_key][action] = old_q + learning_rate * (reward + discount_factor * max_next_q - old_q)
            if is_target_move:
                active_color_idx +=1

        all_paths_complete_this_episode = all(path['complete'] for path in current_paths.values())
        grid_full_this_episode = is_grid_full(current_grid, size)
        if episode_path_valid and all_paths_complete_this_episode and grid_full_this_episode:
            if total_reward_episode > max_reward_achieved_overall:
                max_reward_achieved_overall = total_reward_episode
                best_grid_overall = [row[:] for row in current_grid]
                best_paths_overall = {c: path['coords'] for c, path in current_paths.items()}
                best_episode_num = episode + 1
                if not BENCHMARK_MODE:
                    print(f"Q-Learning: New best solution in episode {episode+1} with reward {total_reward_episode:.2f}")
        exploration_rate = max(0.05, exploration_rate * 0.995)

    solve_time = time.time() - start_time_total
    if best_grid_overall and best_paths_overall:
        solution_grid = best_grid_overall
        solution_paths_dict = best_paths_overall
        if not BENCHMARK_MODE:
            print(f"Q-Learning: Solution found from episode {best_episode_num} in {solve_time:.4f}s, {states_explored} states, Max Reward: {max_reward_achieved_overall:.2f}")
    else:
        if not BENCHMARK_MODE:
            print(f"Q-Learning: No solution found after {episodes_to_run} episodes in {solve_time:.4f}s, {states_explored} states.")
    return solution_grid, solution_paths_dict, solve_time, states_explored

def backtrack_solver(grid, current_paths, size, color_order, state_counter):
    state_counter[0] += 1
    active_color = -1
    all_complete = True
    for color in color_order:
        if color in current_paths and not current_paths[color]['complete']:
            active_color = color
            all_complete = False
            break

    if all_complete:
        return (grid, copy.deepcopy(current_paths)) if is_grid_full(grid, size) else (None, None)

    if active_color == -1:
        return None, None

    path_data = current_paths[active_color]
    current_head = path_data['coords'][-1]
    target = path_data['target']

    neighbors_sorted = sorted(get_neighbors(current_head[0], current_head[1], size),
                            key=lambda p: manhattan_distance(p, target))

    for nr, nc in neighbors_sorted:
        is_target = (nr, nc) == target
        is_valid_empty_cell = (grid[nr][nc] == 0)
        is_valid_target_cell = (is_target and grid[nr][nc] == -active_color)
        can_move_to = is_target or (is_valid_empty_cell and (nr, nc) not in path_data['coords'])

        if can_move_to:
            if is_valid_empty_cell:
                grid[nr][nc] = active_color
                path_data['coords'].append((nr, nc))
                result_grid, result_paths = backtrack_solver(grid, current_paths, size, color_order, state_counter)
                if result_grid: return result_grid, result_paths
                path_data['coords'].pop()
                grid[nr][nc] = 0
            elif is_valid_target_cell:
                path_data['coords'].append((nr, nc))
                path_data['complete'] = True
                result_grid, result_paths = backtrack_solver(grid, current_paths, size, color_order, state_counter)
                if result_grid: return result_grid, result_paths
                path_data['complete'] = False
                path_data['coords'].pop()
    return None, None

def solve_backtracking(puzzle_str, time_limit=60.0):
    global BENCHMARK_MODE
    start_time_total = time.time()
    solution_paths_dict = None
    solution_grid = None
    states_explored_counter = [0]

    try:
        initial_grid, initial_colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid is None or not initial_colors_data or size == 0: raise ValueError("Invalid puzzle data")
    except ValueError as e:
        if not BENCHMARK_MODE: print(f"Lỗi parse puzzle cho Backtracking: {e}")
        return None, None, time.time() - start_time_total, states_explored_counter[0]

    clean_grid = [[0] * size for _ in range(size)]
    current_paths_internal = {}
    color_order = sorted(initial_colors_data.keys(),
                         key=lambda c: manhattan_distance(initial_colors_data[c]['start'], initial_colors_data[c]['end']))

    for color, data in initial_colors_data.items():
        sr, sc = data['start']; er, ec = data['end']
        clean_grid[sr][sc] = -color; clean_grid[er][ec] = -color
        current_paths_internal[color] = {'coords': [data['start']], 'target': data['end'], 'complete': False}

    if not BENCHMARK_MODE: print("Giải bằng Backtracking (Optimized)...")
    grid_copy = [row[:] for row in clean_grid]
    paths_copy = copy.deepcopy(current_paths_internal)

    solved_paths_internal = None

    solver_thread_result = {}
    def solver_task():
        try:
            grid_res, paths_res = backtrack_solver(grid_copy, paths_copy, size, color_order, states_explored_counter)
            solver_thread_result['grid'] = grid_res
            solver_thread_result['paths'] = paths_res
            solver_thread_result['error'] = None
        except Exception as e:
            solver_thread_result['error'] = e
            solver_thread_result['traceback'] = traceback.format_exc()

    solver_thread = threading.Thread(target=solver_task, daemon=True)
    solver_thread.start()
    solver_thread.join(timeout=time_limit)

    timed_out = False
    if solver_thread.is_alive():
        if not BENCHMARK_MODE: print(f"Backtracking: Đã vượt quá giới hạn thời gian ({time_limit}s).")
        timed_out = True
        solution_grid = None
    elif solver_thread_result.get('error'):
        if not BENCHMARK_MODE:
            print(f"Lỗi trong thread Backtracking: {solver_thread_result['error']}")
        solution_grid = None
    else:
        solution_grid = solver_thread_result.get('grid')
        solved_paths_internal = solver_thread_result.get('paths')

    solve_time = time.time() - start_time_total
    if not BENCHMARK_MODE: print(f"Thời gian Backtracking: {solve_time:.4f} giây, Trạng thái: {states_explored_counter[0]}")

    if timed_out:
        return None, None, solve_time, states_explored_counter[0]

    if solution_grid and solved_paths_internal:
        solution_paths_dict = {color: data['coords'] for color, data in solved_paths_internal.items() if data and 'coords' in data}
        if len(solution_paths_dict) != len(initial_colors_data):
             if not BENCHMARK_MODE: print("Lỗi Backtracking: Số path trả về không khớp số màu.")
             return None, None, solve_time, states_explored_counter[0]
        if not is_grid_full(solution_grid, size):
             if not BENCHMARK_MODE: print("Lỗi Backtracking: Grid không đầy đủ dù các path đã hoàn thành.")
             return None, None, solve_time, states_explored_counter[0]
    else:
        solution_grid = None
        solution_paths_dict = None

    return solution_grid, solution_paths_dict, solve_time, states_explored_counter[0]

def solve_cp(puzzle_str, time_limit=60.0):
    global BENCHMARK_MODE
    if not ORTOOLS_AVAILABLE:
        if not BENCHMARK_MODE: print("Lỗi: OR-Tools không khả dụng.")
        return None, None, 0.0, 0

    start_time_total = time.time()
    solution_grid = None
    solution_paths_dict = None

    try:
        initial_grid_cp, colors_data_cp, size_cp, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid_cp is None or not colors_data_cp or size_cp == 0: raise ValueError("Invalid puzzle data")
    except ValueError as e:
        if not BENCHMARK_MODE: print(f"Lỗi parse puzzle cho CP: {e}")
        return None, None, time.time() - start_time_total, 0

    colors = list(colors_data_cp.keys())
    num_colors = len(colors)
    if num_colors == 0:
        if not BENCHMARK_MODE: print("CP: Không có màu hợp lệ.")
        return None, None, time.time() - start_time_total, 0

    color_map = {c: i for i, c in enumerate(colors)}
    idx_to_color = {i: c for c, i in color_map.items()}
    model = cp_model.CpModel()
    is_path = {}
    for r in range(size_cp):
        for c in range(size_cp):
            for k in range(num_colors):
                is_path[r, c, k] = model.NewBoolVar(f'p_{r}_{c}_{k}')

    for r in range(size_cp):
        for c in range(size_cp):
            model.Add(sum(is_path[r, c, k] for k in range(num_colors)) == 1)

    for k_color, k_idx in color_map.items():
        sr, sc = colors_data_cp[k_color]['start']
        er, ec = colors_data_cp[k_color]['end']
        if not (0 <= sr < size_cp and 0 <= sc < size_cp and 0 <= er < size_cp and 0 <= ec < size_cp):
            if not BENCHMARK_MODE: print(f"Lỗi CP: Endpoint màu {k_color} không hợp lệ.")
            return None, None, time.time() - start_time_total, 0
        model.Add(is_path[sr, sc, k_idx] == 1)
        model.Add(is_path[er, ec, k_idx] == 1)

    for k_color, k_idx in color_map.items():
        sr, sc = colors_data_cp[k_color]['start']
        er, ec = colors_data_cp[k_color]['end']
        for r in range(size_cp):
            for c in range(size_cp):
                neighbors = get_neighbors(r, c, size_cp)
                sum_neighbors = sum(is_path[nr, nc, k_idx] for nr, nc in neighbors)
                is_rc_endpoint_k = (r == sr and c == sc) or (r == er and c == ec)
                b_is_path = is_path[r,c,k_idx]
                b_is_endpoint = model.NewConstant(int(is_rc_endpoint_k))

                model.Add(sum_neighbors == 1).OnlyEnforceIf([b_is_path, b_is_endpoint])
                model.Add(sum_neighbors == 2).OnlyEnforceIf([b_is_path, b_is_endpoint.Not()])

    if not BENCHMARK_MODE: print("Giải bằng Constraint Programming (OR-Tools)...")
    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = False
    solver.parameters.max_time_in_seconds = time_limit
    status = solver.Solve(model)

    solve_time = time.time() - start_time_total
    actual_solver_time = solver.WallTime()

    if not BENCHMARK_MODE:
        print(f"Thời gian CP Solver: {actual_solver_time:.4f} giây (Tổng thời gian hàm: {solve_time:.4f} giây)")
        print(f"Trạng thái CP: {solver.StatusName(status)}")

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        if not BENCHMARK_MODE: print("CP: Tìm thấy lời giải.")
        solution_grid_temp = [[0] * size_cp for _ in range(size_cp)]
        try:
            for r_idx in range(size_cp):
                for c_idx in range(size_cp):
                    for k_idx_loop in range(num_colors):
                        if solver.Value(is_path[r_idx, c_idx, k_idx_loop]):
                            actual_color = idx_to_color[k_idx_loop]
                            is_start = (r_idx,c_idx) == colors_data_cp[actual_color]['start']
                            is_end = (r_idx,c_idx) == colors_data_cp[actual_color]['end']
                            solution_grid_temp[r_idx][c_idx] = -actual_color if (is_start or is_end) else actual_color
                            break
            solution_grid = solution_grid_temp
        except Exception as e:
            if not BENCHMARK_MODE: print(f"Lỗi khi xây dựng grid từ kết quả CP: {e}")
            return None, None, solve_time, 0

        if not BENCHMARK_MODE: print("CP: Tái tạo đường đi từ grid kết quả...")
        solution_paths_dict = reconstruct_paths(solution_grid, colors_data_cp, size_cp)
        if not solution_paths_dict or len(solution_paths_dict) != len(colors_data_cp):
             if not BENCHMARK_MODE: print("Cảnh báo: Tái tạo paths từ kết quả CP thất bại hoặc không đủ.")
             solution_paths_dict = None
             solution_grid = None

    if not (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE):
        solution_grid = None
        solution_paths_dict = None

    return solution_grid, solution_paths_dict, solve_time, 0

def solve_bfs(puzzle_str, time_limit=60.0, state_limit=500000):
    global BENCHMARK_MODE
    start_time_total = time.time()
    solution_paths_dict = None
    states_explored = 0
    solution_grid = None

    try:
        initial_grid_parsed, initial_colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid_parsed is None or not initial_colors_data or size == 0: raise ValueError("Invalid puzzle data")
    except ValueError as e:
        if not BENCHMARK_MODE: print(f"Lỗi parse puzzle cho BFS: {e}")
        return None, None, time.time() - start_time_total, 0

    color_order = sorted(initial_colors_data.keys())
    num_colors = len(color_order)

    initial_paths = {c: {'coords': [initial_colors_data[c]['start']],
                         'target': initial_colors_data[c]['end'],
                         'complete': False}
                     for c in color_order}
    initial_grid_tuple = tuple(map(tuple, initial_grid_parsed))
    initial_state = (initial_grid_tuple, initial_paths, 0)

    queue = deque([initial_state])
    visited = set()
    visited.add((initial_grid_tuple, 0))

    if not BENCHMARK_MODE: print("Giải bằng BFS (Simple Visited Set)...")

    timed_out, state_limited_out = False, False

    while queue:
        if time.time() - start_time_total > time_limit:
            if not BENCHMARK_MODE: print(f"BFS: Đã vượt quá giới hạn thời gian ({time_limit}s).")
            timed_out = True; break
        if states_explored > state_limit:
            if not BENCHMARK_MODE: print(f"BFS: Đã vượt quá giới hạn trạng thái ({state_limit}).")
            state_limited_out = True; break

        current_grid_tuple, current_paths, current_color_idx = queue.popleft()
        states_explored += 1

        all_complete = all(data['complete'] for data in current_paths.values())
        if all_complete:
            final_grid_list = [list(row) for row in current_grid_tuple]
            if is_grid_full(final_grid_list, size):
                if not BENCHMARK_MODE: print(f"BFS: Tìm thấy lời giải sau {states_explored} trạng thái.")
                solution_grid = final_grid_list
                solution_paths_dict = {c: d['coords'] for c, d in current_paths.items()}
                break
            else: continue

        if current_color_idx >= num_colors: continue
        active_color = color_order[current_color_idx]
        path_data = current_paths.get(active_color)

        if not path_data or path_data['complete']:
            next_color_idx = current_color_idx + 1
            next_state_key = (current_grid_tuple, next_color_idx)
            if next_state_key not in visited:
                visited.add(next_state_key)
                queue.append((current_grid_tuple, current_paths, next_color_idx))
            continue

        current_head = path_data['coords'][-1]
        target = path_data['target']
        current_grid_list_for_check = [list(row) for row in current_grid_tuple]

        for nr, nc in get_neighbors(current_head[0], current_head[1], size):
            is_target = (nr, nc) == target
            cell_value = current_grid_list_for_check[nr][nc]
            is_valid_empty_cell = (cell_value == 0)
            is_valid_target_cell = (is_target and cell_value == -active_color)
            can_move_to = is_target or (is_valid_empty_cell and (nr, nc) not in path_data['coords'])

            if can_move_to:
                next_paths = get_path_data_copy(current_paths)
                next_grid_list = [row[:] for row in current_grid_list_for_check]
                next_color_idx_for_state = current_color_idx; moved = False
                if is_valid_empty_cell:
                    next_grid_list[nr][nc] = active_color
                    next_paths[active_color]['coords'].append((nr, nc)); moved = True
                elif is_valid_target_cell:
                    next_paths[active_color]['coords'].append((nr, nc))
                    next_paths[active_color]['complete'] = True; moved = True
                    next_color_idx_for_state += 1
                if moved:
                    next_grid_tuple = tuple(map(tuple, next_grid_list))
                    next_state_key = (next_grid_tuple, next_color_idx_for_state)
                    if next_state_key not in visited:
                        visited.add(next_state_key)
                        queue.append((next_grid_tuple, next_paths, next_color_idx_for_state))

    solve_time = time.time() - start_time_total
    if not solution_grid and not BENCHMARK_MODE and not timed_out and not state_limited_out:
        print(f"BFS: Không tìm thấy lời giải.")
    if not BENCHMARK_MODE:
        print(f"Thời gian BFS: {solve_time:.4f} giây, Trạng thái khám phá: {states_explored}")

    if not solution_grid: solution_paths_dict = None
    return solution_grid, solution_paths_dict, solve_time, states_explored


def solve_astar(puzzle_str, heuristic_func, time_limit=120.0, state_limit=300000):
    global BENCHMARK_MODE
    start_time_total = time.time()
    solution_paths_dict, solution_grid = None, None
    states_explored = 0

    try:
        initial_grid_parsed, initial_colors_data, size, _ = parse_puzzle_extended(puzzle_str)
        if initial_grid_parsed is None or not initial_colors_data or size == 0: raise ValueError("Invalid puzzle data")
    except ValueError as e:
        if not BENCHMARK_MODE: print(f"Lỗi parse puzzle cho A*: {e}")
        return None, None, time.time() - start_time_total, 0

    color_order = sorted(initial_colors_data.keys())
    num_colors = len(color_order)

    initial_paths = {c: {'coords': [initial_colors_data[c]['start']],
                         'target': initial_colors_data[c]['end'],
                         'complete': False}
                     for c in color_order}
    initial_grid_tuple = tuple(map(tuple, initial_grid_parsed))
    initial_state = (initial_grid_tuple, initial_paths, 0)

    state_counter = itertools.count(); initial_g = 0
    initial_h = heuristic_func(initial_paths, color_order, size, initial_grid_tuple)
    priority_queue = [(initial_g + initial_h, next(state_counter), initial_state)]
    visited_states = {}; visited_states[(initial_grid_tuple, 0)] = initial_g

    if not BENCHMARK_MODE: print(f"Giải bằng A* (Heuristic: {heuristic_func.__name__})...")

    timed_out, state_limited_out = False, False

    while priority_queue:
        if time.time() - start_time_total > time_limit:
            if not BENCHMARK_MODE: print(f"A*: Đã vượt quá giới hạn thời gian ({time_limit}s).")
            timed_out = True; break
        if states_explored > state_limit:
            if not BENCHMARK_MODE: print(f"A*: Đã vượt quá giới hạn trạng thái ({state_limit}).")
            state_limited_out = True; break

        f_score, _, current_state_tuple = heapq.heappop(priority_queue)
        current_grid_tuple, current_paths, current_color_idx = current_state_tuple
        states_explored += 1
        current_g = sum(len(data['coords']) - 1 for data in current_paths.values())
        state_key = (current_grid_tuple, current_color_idx)
        if state_key in visited_states and visited_states[state_key] < current_g: continue

        all_complete = all(data['complete'] for data in current_paths.values())
        if all_complete:
            final_grid_list = [list(row) for row in current_grid_tuple]
            if is_grid_full(final_grid_list, size):
                if not BENCHMARK_MODE: print(f"A*: Tìm thấy lời giải sau {states_explored} trạng thái.")
                solution_grid = final_grid_list
                solution_paths_dict = {c: d['coords'] for c, d in current_paths.items()}
                break
            else: continue

        if current_color_idx >= num_colors: continue
        active_color = color_order[current_color_idx]
        path_data = current_paths.get(active_color)

        if not path_data or path_data['complete']:
            next_color_idx = current_color_idx + 1; next_state_key = (current_grid_tuple, next_color_idx)
            next_g = current_g; next_h = heuristic_func(current_paths, color_order, size, current_grid_tuple)
            if next_state_key not in visited_states or visited_states[next_state_key] > next_g:
                visited_states[next_state_key] = next_g
                heapq.heappush(priority_queue, (next_g + next_h, next(state_counter), (current_grid_tuple, current_paths, next_color_idx)))
            continue

        current_head = path_data['coords'][-1]; target = path_data['target']
        current_grid_list_for_check = [list(row) for row in current_grid_tuple]

        for nr, nc in get_neighbors(current_head[0], current_head[1], size):
            is_target = (nr, nc) == target; cell_value = current_grid_list_for_check[nr][nc]
            is_valid_empty_cell = (cell_value == 0)
            is_valid_target_cell = (is_target and cell_value == -active_color)
            can_move_to = is_target or (is_valid_empty_cell and (nr, nc) not in path_data['coords'])

            if can_move_to:
                next_paths = get_path_data_copy(current_paths)
                next_grid_list = [row[:] for row in current_grid_list_for_check]
                next_color_idx_for_state = current_color_idx; moved = False
                if is_valid_empty_cell:
                    next_grid_list[nr][nc] = active_color
                    next_paths[active_color]['coords'].append((nr, nc)); moved = True
                elif is_valid_target_cell:
                    next_paths[active_color]['coords'].append((nr, nc))
                    next_paths[active_color]['complete'] = True; moved = True
                    next_color_idx_for_state += 1
                if moved:
                    next_grid_tuple = tuple(map(tuple, next_grid_list))
                    next_g_val = current_g + 1
                    next_h_val = heuristic_func(next_paths, color_order, size, next_grid_tuple)
                    next_state_key = (next_grid_tuple, next_color_idx_for_state)
                    if next_state_key not in visited_states or visited_states[next_state_key] > next_g_val:
                        visited_states[next_state_key] = next_g_val
                        heapq.heappush(priority_queue, (next_g_val + next_h_val, next(state_counter), (next_grid_tuple, next_paths, next_color_idx_for_state)))

    solve_time = time.time() - start_time_total
    if not solution_grid and not BENCHMARK_MODE and not timed_out and not state_limited_out:
        print(f"A*: Không tìm thấy lời giải.")
    if not BENCHMARK_MODE:
        print(f"Thời gian A*: {solve_time:.4f} giây, Trạng thái khám phá: {states_explored}")

    if not solution_grid: solution_paths_dict = None
    return solution_grid, solution_paths_dict, solve_time, states_explored
# ============================================================
# DỮ LIỆU PUZZLE MẪU
# ============================================================
PUZZLES = {
    "Tiny (3x3)": [
        """
123
...
123
""",
"""
1.1
2.2
3.3
"""
,
"""
112
2..
3.3
"""
,
"""
1.3
2..
213
"""
,
"""
1..
231
2.3
"""
,
"""
12.
.1.
332
"""
,
"""
133
.22
..1
"""
,
"""
1..
221
3.3
"""
,
"""
1.2
3.3
2.1
"""
,
"""
123
123
...
"""

    ],
    "Easy (5x5)": [
        """
1.2.5
..3.4
.....
.2.5.
.134.
""", """
1....
.....
..4..
342.1
2...3
""", """
.123.
...5.
..5..
1..4.
2.43.
"""
, """
...12
1....
..4..
...3.
234..
"""
, """
...12
.432.
.....
....3
..1.4
"""
, """
1.234
.....
...4.
.2..3
....1
"""
, """
1..2.
3....
2.31.
.4.4.
.....
"""
, """
1..23
...4.
..4..
.23.5
.15..
"""
, """
...1.
.432.
..4..
.....
321..
"""
, """
.....
1..3.
2324.
.....
4...1
"""

    ],
    "Medium (6x6)": [
        """
123.45
....6.
..3...
..4...
1.6...
2.5...
""", """
.12..3
......
..45.3
...6.2
.54.61
......
""", """
......
.1..4.
...3..
......
..124.
..23..
"""
, """
1.....
......
......
2343..
14.2..
......
"""
, """
1.....
......
4..34.
....2.
.2..3.
....1.
"""
, """
...1.2
.4.3..
....12
..6354
..5...
6.....
"""
, """
..12..
.54..2
...1.3
..5...
3...4.
......
"""
, """
....12
..4...
..5...
..32..
.4..1.
.53...
"""
, """
13...3
..2..2
5....1
.4...4
.6...6
.....5
"""
, """
1.....
...421
...5.3
..3...
.54.2.
......
"""
    ],
    "Hard (7x7)": [
         """
......1
.....54
.5.....
...62..
..6.3..
....43.
.....12
""", """
.12....
..35.5.
..4....
..67.4.
....76.
..3....
..12...
"""
, """
1......
2.....1
3.345.2
7.6...5
8....46
.7.....
......8
"""
, """
.......
1......
..5....
....5..
..3.4..
24..32.
1......
"""
, """
123....
.....6.
.4.3...
.5..2..
..4.6..
.......
.1....5
"""
    ],
     "Very Hard (8x8)": [
        """
....1...
....2.34
....34.2
...65...
....6...
....5...
........
.1......
""",
"""
126.....
......3.
........
4.45....
........
5....2..
........
13.6....
""",
"""
.1......
...2..2.
..3.....
........
...16..4
...5...3
.6......
......54
"""
,
"""
1.......
......67
...5....
.7....65
....3..4
..4....2
..32...1
........
"""
,
"""
.......1
.23.....
.......3
...1.4..
.....652
.......6
..5.....
4.......
"""
    ]
}

# ============================================================
# LỚP ỨNG DỤNG TKINTER
# ============================================================
class FlowFreeApp:
    ANIMATION_DELAY = 100

    def __init__(self, root):
        self.root = root
        self.root.title("Flow Free Solver & Benchmarker - Galaxy Edition")

        self.GALAXY_BG = "#0d001a"
        self.GALAXY_FG = "#e0e0ff"
        self.GALAXY_ACCENT1 = "#7a00cc"
        self.GALAXY_ACCENT2 = "#c900ff"
        self.GALAXY_STAR = "#ffffcc"
        self.GALAXY_BUTTON_BG = "#301a4d"
        self.GALAXY_BUTTON_FG = self.GALAXY_STAR
        self.GALAXY_LABELFRAME_BG = self.GALAXY_BG
        self.GALAXY_LABELFRAME_FG = self.GALAXY_STAR

        self.root.configure(bg=self.GALAXY_BG)

        style = ttk.Style(self.root)
        style.theme_use('clam')

        style.configure("TFrame", background=self.GALAXY_BG)
        style.configure("TLabel", background=self.GALAXY_BG, foreground=self.GALAXY_FG, font=("Segoe UI", 10))
        style.configure("TLabelframe", background=self.GALAXY_LABELFRAME_BG, bordercolor=self.GALAXY_ACCENT1)
        style.configure("TLabelframe.Label", background=self.GALAXY_LABELFRAME_BG, foreground=self.GALAXY_LABELFRAME_FG, font=("Segoe UI", 11, "bold"))

        style.configure("TButton", background=self.GALAXY_BUTTON_BG, foreground=self.GALAXY_BUTTON_FG,
                        bordercolor=self.GALAXY_ACCENT1, lightcolor=self.GALAXY_ACCENT1, darkcolor=self.GALAXY_BUTTON_BG,
                        font=("Segoe UI", 10, "bold"), padding=5)
        style.map("TButton",
                  background=[('active', self.GALAXY_ACCENT1), ('disabled', '#444455')],
                  foreground=[('active', self.GALAXY_STAR), ('disabled', '#888899')])

        style.configure("TRadiobutton", background=self.GALAXY_BG, foreground=self.GALAXY_FG,
                        indicatorcolor=self.GALAXY_ACCENT2, font=("Segoe UI", 10))
        style.map("TRadiobutton",
                  background=[('active', self.GALAXY_BUTTON_BG)],
                  indicatorcolor=[('selected', self.GALAXY_STAR), ('pressed', self.GALAXY_ACCENT1)])

        style.configure("TMenubutton", background=self.GALAXY_BUTTON_BG, foreground=self.GALAXY_BUTTON_FG,
                        arrowcolor=self.GALAXY_STAR, borderwidth=1, relief="raised",
                        font=("Segoe UI", 10))
        style.map("TMenubutton", background=[('active', self.GALAXY_ACCENT1)])

        style.configure("Treeview",
                        background="#1c0033", foreground=self.GALAXY_FG,
                        fieldbackground="#1c0033", font=("Segoe UI", 9))
        style.map("Treeview", background=[('selected', self.GALAXY_ACCENT1)], foreground=[('selected', self.GALAXY_STAR)])
        style.configure("Treeview.Heading",
                        background=self.GALAXY_BUTTON_BG, foreground=self.GALAXY_STAR,
                        font=("Segoe UI", 10, "bold"), relief="raised")
        style.map("Treeview.Heading", background=[('active', self.GALAXY_ACCENT1)])
        style.configure("GalaxyTitle.TLabel", background=self.GALAXY_BG, foreground=self.GALAXY_STAR)
        style.configure("Galaxy.TCheckbutton", background=self.GALAXY_BG, foreground=self.GALAXY_FG, indicatorcolor=self.GALAXY_STAR)
        style.map("Galaxy.TCheckbutton", indicatorcolor=[('selected', self.GALAXY_ACCENT2)])


        self.puzzles = PUZZLES
        def get_size_from_difficulty(difficulty_name):
            if self.puzzles[difficulty_name]:
                first_puzzle_str = self.puzzles[difficulty_name][0]
                lines = [line.strip() for line in first_puzzle_str.strip().split('\n') if line.strip()]
                if lines: return len(lines[0])
            return float('inf')
        self.difficulty_levels = sorted(list(self.puzzles.keys()), key=get_size_from_difficulty)
        self.current_difficulty = tk.StringVar(value=self.difficulty_levels[0] if self.difficulty_levels else "")

        self.current_puzzle_index = tk.IntVar(value=-1)
        self.current_puzzle_string = ""
        self.current_puzzle_display_var = tk.StringVar(value="N/A")

        self.grid_size = 0; self.grid_data = []; self.initial_grid_data = []
        self.colors_data = {}; self.grid_labels = []

        self.selected_algorithm = tk.StringVar(value="CP" if ORTOOLS_AVAILABLE else "A*")
        self.available_algorithms = ["Backtracking", "BFS", "A*", "CP", "Q-Learning", "Simulated Annealing", "AND-OR Search"]
        self.last_solve_algorithm = ""; self.last_solve_time = 0.0
        self.benchmark_results_data = None

        self.selected_heuristic_name = tk.StringVar(value=list(AVAILABLE_HEURISTICS.keys())[0])
        self.heuristic_to_use = AVAILABLE_HEURISTICS[self.selected_heuristic_name.get()]
        self.heuristic_optimizer_results = []
        self.flow_colors = ['#330044'] + [ # Màu nền mặc định của ô
            # Các màu sắc nổi bật và đa dạng hơn
            '#FF33CC',  # Hồng magenta sáng
            '#33FFFF',  # Cyan sáng
            '#FFFF33',  # Vàng sáng
            '#FF6600',  # Cam
            '#33CC33',  # Xanh lá cây tươi
            '#6633FF',  # Tím violet
            '#FF3333',  # Đỏ tươi
            '#33FF99',  # Xanh mint
            '#FF9933',  # Cam vàng
            '#0099FF',  # Xanh dương sáng
            '#CC33FF',  # Tím hồng
            '#A0A0A0',  # Xám bạc
            '#FFD700',  # Gold
            '#FF69B4',  # HotPink
            '#ADFF2F',  # GreenYellow
            '#7B68EE',  # MediumSlateBlue
            '#FFA07A',  # LightSalmon
            '#20B2AA',  # LightSeaGreen
            '#F08080',  # LightCoral
            '#DA70D6',  # Orchid
            '#B0E0E6'   # PowderBlue
        ] * 2
        self.is_animating = False; self._animation_job_id = None
        self._paths_to_animate = {}; self._animation_color_order = []
        self._current_animation_color_idx = 0; self._current_animation_step = 0

        top_controls_frame = ttk.Frame(self.root)
        top_controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10,5))

        left_panel = ttk.Frame(top_controls_frame)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10), anchor='nw')

        right_panel = ttk.Frame(top_controls_frame)
        right_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False, anchor='ne')

        self.grid_outer_frame = ttk.Frame(self.root, borderwidth=2, relief="sunken", style="Galaxy.TFrame")
        style.configure("Galaxy.TFrame", background=self.GALAXY_ACCENT1)
        self.grid_outer_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.grid_frame = ttk.Frame(self.grid_outer_frame)

        self.status_var = tk.StringVar(value="Sẵn sàng.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding="3 5",
                               background="#100020", foreground=self.GALAXY_STAR, font=("Segoe UI", 9, "italic"))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.create_puzzle_selection_controls(left_panel)
        self.create_algorithm_controls(left_panel)
        self.create_action_buttons(right_panel)

        if self.difficulty_levels:
            self.update_puzzle_options()
        else:
            self.status_var.set("Không có puzzle nào được định nghĩa.")

        self.toggle_heuristic_subframe_visibility()
        self._update_button_states()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)


    def on_closing(self):
        self.stop_animation()
        if self.root.winfo_exists():
            self.root.destroy()

    def create_puzzle_selection_controls(self, parent_frame):
        frame = ttk.LabelFrame(parent_frame, text="Chọn Puzzle", padding=5)
        frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5), anchor='n')

        ttk.Label(frame, text="Độ khó:").grid(row=0, column=0, padx=5, pady=3, sticky=tk.W)
        self.difficulty_menu = ttk.OptionMenu(frame, self.current_difficulty,
                                         self.current_difficulty.get(), *self.difficulty_levels,
                                         command=self.on_difficulty_change, style="TMenubutton")
        self.difficulty_menu.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)


        ttk.Label(frame, text="Puzzle:").grid(row=1, column=0, padx=5, pady=3, sticky=tk.W)
        self.puzzle_option_menu = ttk.OptionMenu(frame, self.current_puzzle_display_var, "N/A", style="TMenubutton")
        self.puzzle_option_menu.grid(row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        self.puzzle_option_menu.config(state=tk.DISABLED)
        frame.columnconfigure(1, weight=1)

    def create_algorithm_controls(self, parent_frame):
        frame = ttk.LabelFrame(parent_frame, text="Cấu hình Thuật toán", padding=5)
        frame.pack(side=tk.TOP, fill=tk.X, pady=5, anchor='n')

        algo_radio_frame = ttk.Frame(frame)
        algo_radio_frame.pack(fill=tk.X, pady=(0,5))
        ttk.Label(algo_radio_frame, text="Thuật toán:").pack(side=tk.LEFT, padx=5)
        for algo in self.available_algorithms:
            rb = ttk.Radiobutton(algo_radio_frame, text=algo, variable=self.selected_algorithm, value=algo)
            if algo == "CP":
                 rb.config(text="CP")
                 if not ORTOOLS_AVAILABLE: rb.config(state=tk.DISABLED);
                 if self.selected_algorithm.get() == "CP" and not ORTOOLS_AVAILABLE: self.selected_algorithm.set("A*")
            rb.pack(side=tk.LEFT, padx=(0,5))

        self.selected_algorithm.trace_add("write", self.toggle_heuristic_subframe_visibility)

        self.heuristic_subframe = ttk.Frame(frame)
        self.heuristic_subframe.pack(fill=tk.X, pady=(5,0))

        self.heuristic_label = ttk.Label(self.heuristic_subframe, text="Heuristic (A*):")
        self.heuristic_menu = ttk.OptionMenu(self.heuristic_subframe, self.selected_heuristic_name,
                                             self.selected_heuristic_name.get(),
                                             *AVAILABLE_HEURISTICS.keys(),
                                             command=self.on_heuristic_change, style="TMenubutton")
        self.optimize_heuristic_button = ttk.Button(self.heuristic_subframe, text="Đánh giá Heuristics",
                                                    command=self.run_heuristic_optimization_threaded)
        self.qlearning_subframe = ttk.Frame(frame)
        self.qlearning_subframe.pack(fill=tk.X, pady=(5,0))

        self.selected_qlearning_config = tk.StringVar(value=list(AVAILABLE_QLEARNING_CONFIGS.keys())[0])
        self.qlearning_label = ttk.Label(self.qlearning_subframe, text="Config (Q-Learning):")
        self.qlearning_menu = ttk.OptionMenu(self.qlearning_subframe, self.selected_qlearning_config,
                                            self.selected_qlearning_config.get(),
                                            *AVAILABLE_QLEARNING_CONFIGS.keys(),
                                            style="TMenubutton")
        self.optimize_qlearning_button = ttk.Button(self.qlearning_subframe, text="Đánh giá Configs",
                                                command=self.run_qlearning_optimization_threaded)
        self.heuristic_subframe.columnconfigure(1, weight=1)
        self.qlearning_subframe.columnconfigure(1, weight=1)

    def toggle_heuristic_subframe_visibility(self, *args):
        is_astar = self.selected_algorithm.get() == "A*"
        is_qlearning = self.selected_algorithm.get() == "Q-Learning"

        if hasattr(self, 'heuristic_label') and self.heuristic_label.winfo_exists():
            if is_astar: self.heuristic_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            else: self.heuristic_label.grid_remove()

        if hasattr(self, 'heuristic_menu') and self.heuristic_menu.winfo_exists():
            if is_astar: self.heuristic_menu.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
            else: self.heuristic_menu.grid_remove()

        if hasattr(self, 'optimize_heuristic_button') and self.optimize_heuristic_button.winfo_exists():
            if is_astar: self.optimize_heuristic_button.grid(row=0, column=2, padx=(10,0), pady=2, sticky=tk.E)
            else: self.optimize_heuristic_button.grid_remove()

        if hasattr(self, 'qlearning_label') and self.qlearning_label.winfo_exists():
            if is_qlearning: self.qlearning_label.grid(row=0, column=0, padx=5, pady=2, sticky=tk.W)
            else: self.qlearning_label.grid_remove()

        if hasattr(self, 'qlearning_menu') and self.qlearning_menu.winfo_exists():
            if is_qlearning: self.qlearning_menu.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
            else: self.qlearning_menu.grid_remove()

        if hasattr(self, 'optimize_qlearning_button') and self.optimize_qlearning_button.winfo_exists():
            if is_qlearning: self.optimize_qlearning_button.grid(row=0, column=2, padx=(10,0), pady=2, sticky=tk.E)
            else: self.optimize_qlearning_button.grid_remove()

        if hasattr(self, '_update_button_states'):
            self._update_button_states()

        if self.root.winfo_exists(): self.root.update_idletasks()

    def create_action_buttons(self, parent_frame):
        frame = ttk.LabelFrame(parent_frame, text="Hành động", padding="10")
        frame.pack(side=tk.TOP, fill=tk.X, pady=5, anchor='n')

        self.solve_button = ttk.Button(frame, text="Giải & Vẽ", command=self.solve_puzzle_threaded, state=tk.DISABLED)
        self.solve_button.pack(pady=3, fill=tk.X)
        self.reset_button = ttk.Button(frame, text="Reset Puzzle", command=self.reset_grid, state=tk.DISABLED)
        self.reset_button.pack(pady=3, fill=tk.X)
        self.benchmark_button = ttk.Button(frame, text="Chạy Benchmark Suite", command=self.run_gui_benchmark_suite)
        self.benchmark_button.pack(pady=3, fill=tk.X)
        self.charts_button = ttk.Button(frame, text="Hiển thị Biểu đồ", command=self.show_benchmark_charts, state=tk.DISABLED)
        self.charts_button.pack(pady=3, fill=tk.X)

    def _update_button_states(self):
        if not self.root.winfo_exists(): return
        can_solve = bool(self.colors_data) and not self.is_animating
        can_reset = bool(self.initial_grid_data) and not self.is_animating

        if hasattr(self, 'solve_button') and self.solve_button.winfo_exists():
            self.solve_button.config(state=tk.NORMAL if can_solve else tk.DISABLED)
        if hasattr(self, 'reset_button') and self.reset_button.winfo_exists():
            self.reset_button.config(state=tk.NORMAL if can_reset else tk.DISABLED)
        if hasattr(self, 'benchmark_button') and self.benchmark_button.winfo_exists():
            self.benchmark_button.config(state=tk.NORMAL if not self.is_animating else tk.DISABLED)

        can_optimize_heuristic = (self.selected_algorithm.get() == "A*" and
                                  bool(self.colors_data) and
                                  not self.is_animating)
        if hasattr(self, 'optimize_heuristic_button') and self.optimize_heuristic_button.winfo_exists():
            self.optimize_heuristic_button.config(state=tk.NORMAL if can_optimize_heuristic else tk.DISABLED)
            can_optimize_qlearning = (self.selected_algorithm.get() == "Q-Learning" and
                              bool(self.colors_data) and
                              not self.is_animating)
        if hasattr(self, 'optimize_qlearning_button') and self.optimize_qlearning_button.winfo_exists():
            self.optimize_qlearning_button.config(state=tk.NORMAL if can_optimize_qlearning else tk.DISABLED)
        if hasattr(self, 'charts_button') and self.charts_button.winfo_exists():
             self.charts_button.config(state=tk.NORMAL if self.benchmark_results_data and MATPLOTLIB_AVAILABLE else tk.DISABLED)

        if hasattr(self, 'puzzle_option_menu') and self.puzzle_option_menu.winfo_exists():
             self.puzzle_option_menu.config(state=tk.DISABLED if self.is_animating else tk.NORMAL if self.puzzles.get(self.current_difficulty.get()) else tk.DISABLED)
        if hasattr(self, 'difficulty_menu') and self.difficulty_menu.winfo_exists():
             self.difficulty_menu.config(state=tk.DISABLED if self.is_animating else tk.NORMAL)
        if hasattr(self, 'heuristic_menu') and self.heuristic_menu.winfo_exists():
             self.heuristic_menu.config(state=tk.DISABLED if self.is_animating or self.selected_algorithm.get() != "A*" else tk.NORMAL)

    def update_puzzle_options(self, event=None):
        self.stop_animation()
        difficulty = self.current_difficulty.get()
        if not difficulty:
            if hasattr(self, 'puzzle_option_menu') and self.puzzle_option_menu.winfo_exists():
                self.puzzle_option_menu.config(state=tk.DISABLED)
            self.current_puzzle_index.set(-1); self.current_puzzle_display_var.set("N/A")
            self.clear_grid_display(); self._clear_puzzle_state()
            if hasattr(self, 'status_var'): self.status_var.set(f"Không có độ khó nào được chọn.")
            self._update_button_states()
            return

        puzzles_list = self.puzzles.get(difficulty, [])
        puzzle_display_names = [f"Puzzle {i+1}" for i in range(len(puzzles_list))]

        if hasattr(self, 'puzzle_option_menu') and self.puzzle_option_menu.winfo_exists():
            menu = self.puzzle_option_menu["menu"]
            if menu is None:
                self.puzzle_option_menu["menu"] = tk.Menu(self.puzzle_option_menu, tearoff=0)
                menu = self.puzzle_option_menu["menu"]
            menu.configure(background=self.GALAXY_BUTTON_BG, foreground=self.GALAXY_STAR,
                           activebackground=self.GALAXY_ACCENT1, activeforeground=self.GALAXY_STAR,
                           font=("Segoe UI", 10))
            menu.delete(0, "end")

            if puzzle_display_names:
                self.puzzle_option_menu.config(state=tk.NORMAL)
                for i, name in enumerate(puzzle_display_names):
                    menu.add_command(label=name, command=lambda idx=i, disp_name=name: (
                        self.current_puzzle_display_var.set(disp_name),
                        self.set_puzzle_index_and_load(idx)
                    ))
                self.current_puzzle_display_var.set(puzzle_display_names[0])
                self.set_puzzle_index_and_load(0)
            else:
                self.puzzle_option_menu.config(state=tk.DISABLED)
                self.current_puzzle_index.set(-1); self.current_puzzle_display_var.set("N/A")
                self.clear_grid_display(); self._clear_puzzle_state()
                if hasattr(self, 'status_var'): self.status_var.set(f"Không có puzzle cho độ khó '{difficulty}'.")
                self._update_button_states()
        else:
            if not BENCHMARK_MODE: print("Lỗi: puzzle_option_menu không tồn tại khi update_puzzle_options")


    def _clear_puzzle_state(self):
        self.grid_data = []; self.initial_grid_data = []; self.colors_data = {}
        self.grid_size = 0; self.current_puzzle_string = ""; self.is_animating = False


    def set_puzzle_index_and_load(self, index):
        self.stop_animation()
        self.current_puzzle_index.set(index)
        self.load_puzzle()

    def on_difficulty_change(self, *args):
        if not BENCHMARK_MODE: print(f"Đổi độ khó thành: {self.current_difficulty.get()}")
        if hasattr(self, 'difficulty_menu') and self.difficulty_menu.winfo_exists():
            menu = self.difficulty_menu["menu"]
            if menu:
                 menu.config(background=self.GALAXY_BUTTON_BG, foreground=self.GALAXY_STAR,
                             activebackground=self.GALAXY_ACCENT1, activeforeground=self.GALAXY_STAR,
                             font=("Segoe UI", 10))
        self.update_puzzle_options()


    def on_heuristic_change(self, *args):
        selected_name = self.selected_heuristic_name.get()
        if selected_name in AVAILABLE_HEURISTICS:
            self.heuristic_to_use = AVAILABLE_HEURISTICS[selected_name]
            if not BENCHMARK_MODE:
                print(f"Đã chọn Heuristic cho A*: {selected_name}")
        else:
             default_h_name = list(AVAILABLE_HEURISTICS.keys())[0]
             self.selected_heuristic_name.set(default_h_name)
             self.heuristic_to_use = AVAILABLE_HEURISTICS[default_h_name]
        if hasattr(self, 'heuristic_menu') and self.heuristic_menu.winfo_exists():
            menu = self.heuristic_menu["menu"]
            if menu:
                 menu.config(background=self.GALAXY_BUTTON_BG, foreground=self.GALAXY_STAR,
                             activebackground=self.GALAXY_ACCENT1, activeforeground=self.GALAXY_STAR,
                             font=("Segoe UI", 10))


    def load_puzzle(self):
        self.stop_animation()
        difficulty = self.current_difficulty.get(); index = self.current_puzzle_index.get()
        if not BENCHMARK_MODE: print(f"Đang tải: {difficulty} - Puzzle Index {index}")
        if index < 0:
             self.clear_grid_display(); self._clear_puzzle_state()
             if hasattr(self, 'status_var'): self.status_var.set("Chưa chọn puzzle.")
             self._update_button_states(); return
        puzzles_list = self.puzzles.get(difficulty, [])
        if 0 <= index < len(puzzles_list):
            self.current_puzzle_string = puzzles_list[index]
            if hasattr(self, 'status_var'): self.status_var.set(f"Phân tích {difficulty} - P{index+1}...");
            if self.root.winfo_exists(): self.root.update_idletasks()
            try:
                grid, colors, size, _ = parse_puzzle_extended(self.current_puzzle_string)
                if grid is None or size == 0: raise ValueError("Dữ liệu puzzle không hợp lệ hoặc rỗng.")
                self.grid_data = grid; self.initial_grid_data = [row[:] for row in grid]
                self.colors_data = colors if colors is not None else {}; self.grid_size = size
                if self.colors_data:
                     if not BENCHMARK_MODE: print(f"Parse thành công: {size}x{size}, {len(self.colors_data)} màu.")
                     self.update_grid_display(show_paths=False)
                     if hasattr(self, 'status_var'): self.status_var.set(f"Đã tải {difficulty} - P{index+1}. Sẵn sàng giải.")
                else:
                     if not BENCHMARK_MODE: print(f"Cảnh báo: Puzzle {difficulty} - P{index+1} không có cặp điểm màu hợp lệ.")
                     self.update_grid_display(show_paths=False)
                     if hasattr(self, 'status_var'): self.status_var.set(f"P{index+1} không có cặp điểm màu hợp lệ.")
                self.is_animating = False; self._update_button_states()
            except ValueError as e:
                if not BENCHMARK_MODE: print(f"Lỗi khi tải hoặc phân tích puzzle: {e}");
                if self.root.winfo_exists(): messagebox.showerror("Lỗi tải Puzzle", f"Không thể đọc hoặc phân tích puzzle:\n{e}")
                self.clear_grid_display(); self._clear_puzzle_state()
                if hasattr(self, 'status_var'): self.status_var.set("Lỗi tải puzzle.")
                self._update_button_states()
        else:
            if not BENCHMARK_MODE: print(f"Lỗi index puzzle không hợp lệ: {index}");
            self.clear_grid_display(); self._clear_puzzle_state()
            if hasattr(self, 'status_var'): self.status_var.set("Index puzzle không hợp lệ.")
            self._update_button_states()

    def clear_grid_display(self):
         if hasattr(self, 'grid_frame') and self.grid_frame.winfo_exists():
             for widget in self.grid_frame.winfo_children():
                 widget.destroy()
         self.grid_labels = []

    def update_grid_display(self, show_paths=True):
        if hasattr(self, 'clear_grid_display'):
            self.clear_grid_display()
        else:
            if hasattr(self, 'grid_frame') and self.grid_frame.winfo_exists():
                for widget in self.grid_frame.winfo_children(): widget.destroy()
            self.grid_labels = []

        if not self.grid_data or self.grid_size == 0:
            if hasattr(self, '_update_button_states'): self._update_button_states()
            return

        if self.root.winfo_exists(): self.root.update_idletasks()
        container_width = self.grid_outer_frame.winfo_width() - 4
        container_height = self.grid_outer_frame.winfo_height() - 4

        if self.grid_size == 3:
            target_cell_size = 75
        elif self.grid_size == 5:
            target_cell_size = 60
        elif self.grid_size == 6:
            target_cell_size = 50
        elif self.grid_size == 7:
            target_cell_size = 45
        elif self.grid_size == 8:
            target_cell_size = 40
        else:
            cell_size_w = container_width // self.grid_size if self.grid_size > 0 else 25
            cell_size_h = container_height // self.grid_size if self.grid_size > 0 else 25
            target_cell_size = max(20, min(cell_size_w, cell_size_h, 70))

        cell_size = target_cell_size
        font_families = tk.font.families()
        custom_font_name = "Orbitron" if "Orbitron" in font_families else "Segoe UI Semibold"

        font_size = max(9, cell_size // 3)
        endpoint_font_size = max(12, cell_size // 2)
        cell_font = (custom_font_name, font_size)
        endpoint_font = (custom_font_name, endpoint_font_size, "bold")

        self.grid_labels = [[None] * self.grid_size for _ in range(self.grid_size)]
        int_to_char = {i: str(i) for i in range(1, 10)}
        for i, char_code in enumerate(range(ord('A'), ord('Z') + 1)): int_to_char[10 + i] = chr(char_code)

        inner_frame = ttk.Frame(self.grid_frame, style="TFrame")

        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                val = self.grid_data[r_idx][c_idx]; abs_val = abs(val); text_val = ""
                bg_color_val = self.flow_colors[0]
                fg_color_val = self.GALAXY_FG
                font_val = cell_font; relief_val = "flat"; border_val = 1
                border_color = self.GALAXY_ACCENT1

                is_endpoint = val < 0; is_path_segment = val > 0 and show_paths
                if is_endpoint or is_path_segment:
                    color_idx_raw = abs_val % (len(self.flow_colors) -1) +1 if abs_val >0 else 0
                    safe_idx = color_idx_raw if 0 <= color_idx_raw < len(self.flow_colors) else 1
                    if abs_val > 0 and safe_idx == 0: safe_idx = 1
                    try:
                        bg_color_val = self.flow_colors[safe_idx]
                        rgb = self.root.winfo_rgb(bg_color_val)
                        brightness = (0.2126*rgb[0]/65535 + 0.7152*rgb[1]/65535 + 0.0722*rgb[2]/65535)
                        fg_color_val = "#000033" if brightness > 0.6 else self.GALAXY_STAR
                    except tk.TclError: bg_color_val = self.flow_colors[1]; fg_color_val = "#000033";
                    except Exception: bg_color_val = "magenta"; fg_color_val = "white";

                    if is_endpoint:
                        text_val = int_to_char.get(abs_val, '?'); font_val = endpoint_font
                        relief_val = "raised"; border_val = 2; border_color = fg_color_val
                    elif is_path_segment:
                        relief_val = "sunken"; border_val = 1; border_color = bg_color_val

                cell_f = tk.Frame(inner_frame, bg=bg_color_val, borderwidth=0, relief=relief_val,
                                      width=cell_size, height=cell_size,
                                      highlightbackground=border_color, highlightthickness=border_val)
                cell_f.grid(row=r_idx, column=c_idx, sticky="nsew", padx=1, pady=1)
                inner_frame.grid_rowconfigure(r_idx, weight=1, minsize=cell_size)
                inner_frame.grid_columnconfigure(c_idx, weight=1, minsize=cell_size)

                if text_val:
                    lbl = tk.Label(cell_f, text=text_val, font=font_val, fg=fg_color_val, bg=bg_color_val)
                    lbl.pack(expand=True, fill=tk.BOTH)
                self.grid_labels[r_idx][c_idx] = cell_f

        self.grid_frame.place(in_=self.grid_outer_frame, anchor="c", relx=.5, rely=.5)
        inner_frame.pack(expand=True, padx=2, pady=2)

        if self.root.winfo_exists(): self.root.update_idletasks()
        if hasattr(self, '_update_button_states'): self._update_button_states()

    def reset_grid(self):
        self.stop_animation()
        if self.initial_grid_data and self.grid_size > 0:
            if not BENCHMARK_MODE: print("Resetting grid...")
            self.grid_data = [row[:] for row in self.initial_grid_data]
            try:
                _, self.colors_data, self.grid_size, _ = parse_puzzle_extended(self.current_puzzle_string)
                if not self.colors_data or self.grid_size == 0: raise ValueError("Parse lại khi reset thất bại")
                self.update_grid_display(show_paths=False)
                if hasattr(self, 'status_var'): self.status_var.set(f"Đã reset. Sẵn sàng giải {self.current_difficulty.get()} - P{self.current_puzzle_index.get()+1}.")
            except ValueError as e:
                 if not BENCHMARK_MODE: print(f"Lỗi parse lại khi reset: {e}")
                 if hasattr(self, 'status_var'): self.status_var.set("Lỗi khi reset puzzle.")
                 self._clear_puzzle_state()
            self.is_animating = False; self._update_button_states()
        else:
            if not BENCHMARK_MODE: print("Không có dữ liệu ban đầu để reset.")
            if hasattr(self, 'status_var'): self.status_var.set("Không có dữ liệu để reset.")
            self._update_button_states()

    def solve_puzzle_threaded(self):
         if self.is_animating:
             if self.root.winfo_exists(): messagebox.showwarning("Đang bận", "Vui lòng đợi hoạt ảnh vẽ xong!"); return
         if not self.colors_data:
             if self.root.winfo_exists(): messagebox.showwarning("Thiếu dữ liệu", "Vui lòng chọn một puzzle hợp lệ trước."); return
         if not self.initial_grid_data:
             if self.root.winfo_exists(): messagebox.showerror("Lỗi", "Thiếu dữ liệu grid ban đầu. Hãy thử tải lại puzzle."); return

         self.stop_animation(); self.is_animating = False
         self._update_button_states()

         algo = self.selected_algorithm.get(); self.last_solve_algorithm = algo
         if hasattr(self, 'status_var'): self.status_var.set(f"Đang giải bằng {algo}...");
         if not BENCHMARK_MODE: print(f"Bắt đầu giải bằng {algo}...")
         if self.root.winfo_exists(): self.root.update_idletasks()
         puzzle_string_copy = self.current_puzzle_string
         thread = threading.Thread(target=self.run_solver, args=(algo, puzzle_string_copy), daemon=True)
         thread.start()
    def run_solver(self, algorithm_name, puzzle_string_copy):
        solution_grid, solution_paths, solve_time, states_explored = None, None, 0.0, 0
        solver_func = None
        try:
            if algorithm_name == "Backtracking": solver_func = solve_backtracking
            elif algorithm_name == "BFS": solver_func = solve_bfs
            elif algorithm_name == "A*":
                solution_grid, solution_paths, solve_time, states_explored = solve_astar(puzzle_string_copy, self.heuristic_to_use)
            elif algorithm_name == "CP":
                if ORTOOLS_AVAILABLE: solver_func = solve_cp
                else:
                    if self.root.winfo_exists(): self.root.after(0, lambda: messagebox.showerror("Lỗi", "Thư viện OR-Tools chưa được cài đặt."))
                    if self.root.winfo_exists(): self.root.after(0, self.solver_finished, None, None, 0.0, 0, "Lỗi CP")
                    return
            elif algorithm_name == "Q-Learning":
                config_name = self.selected_qlearning_config.get()
                config = AVAILABLE_QLEARNING_CONFIGS.get(config_name, AVAILABLE_QLEARNING_CONFIGS["Default"])
                solution_grid, solution_paths, solve_time, states_explored = solve_qlearning(
                    puzzle_string_copy, config=config)
                solver_func = None
            elif algorithm_name == "Simulated Annealing": solver_func = solve_simulated_annealing
            elif algorithm_name == "AND-OR Search": solver_func = solve_and_or_search
            if solver_func and algorithm_name != "Q-Learning":
                solution_grid, solution_paths, solve_time, states_explored = solver_func(puzzle_string_copy)
            elif algorithm_name != "A*" and algorithm_name != "Q-Learning":
                if not BENCHMARK_MODE: print(f"Lỗi: Thuật toán không xác định: {algorithm_name}")
                if self.root.winfo_exists(): self.root.after(0, self.solver_finished, None, None, 0.0, 0, "Lỗi thuật toán")
                return

            self.last_solve_time = solve_time
            if self.root.winfo_exists():
                self.root.after(10, self.solver_finished, solution_grid, solution_paths, solve_time, states_explored, algorithm_name)
        except Exception as e:
            if not BENCHMARK_MODE: print(f"Lỗi nghiêm trọng khi giải ({algorithm_name}): {e}")
            msg = f"Lỗi nghiêm trọng khi giải ({algorithm_name}):\n{str(e)[:100]}"
            if self.root.winfo_exists():
                self.root.after(0, lambda m=msg: messagebox.showerror("Lỗi Solver", m))
                self.root.after(0, self.solver_finished, None, None, 0.0, 0, f"Lỗi {algorithm_name}")


    def solver_finished(self, solution_grid, solution_paths, solve_time, states_explored, algorithm_name):
         if not self.root.winfo_exists(): return

         if not BENCHMARK_MODE: print(f"Solver ({algorithm_name}) hoàn thành. Thời gian: {solve_time:.4f} giây. Trạng thái: {states_explored}")

         self.is_animating = False
         valid_paths = isinstance(solution_paths, dict) and bool(solution_paths)
         if valid_paths and self.colors_data:
             valid_paths = len(solution_paths) == len(self.colors_data) and \
                           all(isinstance(p, list) and len(p) > 0 for p in solution_paths.values())

         if solution_grid and valid_paths and is_grid_full(solution_grid, self.grid_size):
             if not BENCHMARK_MODE: print("Tìm thấy lời giải hợp lệ. Bắt đầu hoạt ảnh vẽ.")
             self.grid_data = [row[:] for row in self.initial_grid_data]
             self.update_grid_display(show_paths=False)
             if hasattr(self, 'status_var'): self.status_var.set(f"Đang vẽ lời giải ({algorithm_name})...")
             self.start_animation(solution_paths)
         else:
             msg = ""
             if "Lỗi" in algorithm_name:
                 if not BENCHMARK_MODE: print(f"Giải thất bại do lỗi đã báo cáo ({algorithm_name}).")
                 msg = f"Giải thất bại: {algorithm_name}"
             else:
                 msg = f"Không tìm thấy lời giải ({algorithm_name}) trong {solve_time:.2f}s."
                 if solution_grid and (not valid_paths or not is_grid_full(solution_grid, self.grid_size)):
                      msg = f"Giải pháp không hoàn chỉnh ({algorithm_name})."
                 if not BENCHMARK_MODE: print(msg)
             if hasattr(self, 'status_var'): self.status_var.set(msg)
             self.reset_grid()

    def start_animation(self, solved_paths):
        if not solved_paths or self.is_animating:
             if not self.is_animating: self._update_button_states()
             return
        if not BENCHMARK_MODE: print("Bắt đầu hoạt ảnh vẽ đường đi...")
        self.is_animating = True; self._update_button_states()
        self._paths_to_animate = solved_paths
        self._animation_color_order = sorted(self._paths_to_animate.keys(), key=lambda c: len(self._paths_to_animate[c]))
        self._current_animation_color_idx = 0; self._current_animation_step = 1
        self.stop_animation()
        if self.root.winfo_exists(): self._animation_job_id = self.root.after(self.ANIMATION_DELAY, self._animate_step)

    def stop_animation(self):
        if self._animation_job_id:
            try:
                if self.root.winfo_exists(): self.root.after_cancel(self._animation_job_id)
            except tk.TclError: pass
            except Exception as e:
                if not BENCHMARK_MODE: print(f"Lỗi khi hủy job animation: {e}")
            finally: self._animation_job_id = None

    def _animate_step(self):
        if not self.is_animating or not self._paths_to_animate or not self._animation_color_order or not self.root.winfo_exists():
             self._finalize_animation(); return
        try:
            if self._current_animation_color_idx >= len(self._animation_color_order):
                self._finalize_animation(); return
            current_color = self._animation_color_order[self._current_animation_color_idx]
            current_path = self._paths_to_animate.get(current_color)
            if not current_path or self._current_animation_step >= len(current_path):
                self._current_animation_color_idx += 1; self._current_animation_step = 1
                if self.root.winfo_exists(): self._animation_job_id = self.root.after(1, self._animate_step);
                return

            r, c = current_path[self._current_animation_step]
            if not (0 <= r < self.grid_size and 0 <= c < self.grid_size and self.grid_labels and
                    r < len(self.grid_labels) and c < len(self.grid_labels[r]) and
                    isinstance(self.grid_labels[r][c], tk.Frame) and self.grid_labels[r][c].winfo_exists()):
                 if not BENCHMARK_MODE: print(f"Lỗi animation: Tọa độ ({r},{c}) hoặc widget không hợp lệ/đã hủy.")
                 self._finalize_animation(); return

            target_frame = self.grid_labels[r][c]
            color_idx_raw = current_color % (len(self.flow_colors) - 1) + 1 if current_color > 0 else 0
            safe_idx = color_idx_raw if 0 <= color_idx_raw < len(self.flow_colors) else 1
            if current_color > 0 and safe_idx == 0: safe_idx = 1
            bg_color = self.flow_colors[safe_idx]

            if self.initial_grid_data and 0 <= r < len(self.initial_grid_data) and 0 <= c < len(self.initial_grid_data[0]):
                is_initial_empty = self.initial_grid_data[r][c] == 0
                if is_initial_empty:
                    for widget in target_frame.winfo_children(): widget.destroy()
                    target_frame.config(bg=bg_color, relief="sunken", borderwidth=1, highlightthickness=1, highlightbackground=bg_color) # Viền cùng màu

            self._current_animation_step += 1
            if self.root.winfo_exists(): self._animation_job_id = self.root.after(self.ANIMATION_DELAY, self._animate_step)
        except Exception as e:
            if not BENCHMARK_MODE: print(f"Lỗi trong quá trình animate step: {e}");
            self._finalize_animation()

    def _finalize_animation(self):
        if not BENCHMARK_MODE: print("Hoàn tất hoạt ảnh vẽ.")
        self.stop_animation(); self.is_animating = False
        self._paths_to_animate = {}; self._animation_color_order = []
        self._current_animation_color_idx = 0; self._current_animation_step = 0
        try:
             if self.root.winfo_exists():
                  if hasattr(self, 'status_var'): self.status_var.set(f"Đã vẽ xong ({self.last_solve_algorithm}) - {self.last_solve_time:.4f} giây.")
                  self._update_button_states()
        except tk.TclError: pass
        except Exception as e:
            if not BENCHMARK_MODE: print(f"Lỗi không xác định khi finalize animation: {e}")
    def run_qlearning_optimization_threaded(self):
        if not self.current_puzzle_string:
            if self.root.winfo_exists(): messagebox.showwarning("Thiếu Puzzle", "Vui lòng tải một puzzle trước khi đánh giá Q-Learning configs.")
            return
        if self.is_animating:
            if self.root.winfo_exists(): messagebox.showwarning("Đang bận", "Vui lòng đợi các tác vụ khác hoàn thành.")
            return

        if hasattr(self, 'status_var'): self.status_var.set("Đang đánh giá các cấu hình cho Q-Learning...")
        self._update_button_states()

        thread = threading.Thread(target=self._qlearning_optimization_task,
                                args=(self.current_puzzle_string,), daemon=True)
        thread.start()

    def run_heuristic_optimization_threaded(self):
        if not self.current_puzzle_string:
            if self.root.winfo_exists(): messagebox.showwarning("Thiếu Puzzle", "Vui lòng tải một puzzle trước khi đánh giá heuristic.")
            return
        if self.is_animating:
            if self.root.winfo_exists(): messagebox.showwarning("Đang bận", "Vui lòng đợi các tác vụ khác hoàn thành.")
            return

        if hasattr(self, 'status_var'): self.status_var.set("Đang đánh giá các heuristic cho A*...")
        self._update_button_states()

        thread = threading.Thread(target=self._heuristic_optimization_task,
                                   args=(self.current_puzzle_string,), daemon=True)
        thread.start()
    def _qlearning_optimization_task(self, puzzle_str_copy):
        results = []
        global BENCHMARK_MODE; original_benchmark_mode = BENCHMARK_MODE; BENCHMARK_MODE = True

        for name, config in AVAILABLE_QLEARNING_CONFIGS.items():
            if not self.root.winfo_exists(): break
            if self.root.winfo_exists() and hasattr(self, 'status_var'):
                self.root.after(0, lambda n=name: self.status_var.set(f"Đang thử Q-Learning với config: {n}..."))
            try:
                grid, paths, solve_time, states = solve_qlearning(puzzle_str_copy, time_limit=30, config=config)
                is_solved = grid is not None and paths is not None and len(paths) > 0 and is_grid_full(grid, self.grid_size)

                score = float('inf')
                if is_solved: score = solve_time + states * 0.00001
                elif solve_time < 29.9: score = 1000000 + solve_time + states * 0.00001

                results.append({
                    "name": name,
                    "time": solve_time,
                    "states": states,
                    "score": score,
                    "solved": is_solved,
                    "config": config
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "time": float('inf'),
                    "states": float('inf'),
                    "score": float('inf'),
                    "solved": False,
                    "config": config
                })
                if not original_benchmark_mode:
                    print(f"Lỗi khi thử Q-Learning config {name}: {e}")

        BENCHMARK_MODE = original_benchmark_mode
        if self.root.winfo_exists():
            self.root.after(0, self.display_qlearning_optimization_results, results)

    def _heuristic_optimization_task(self, puzzle_str_copy):
        results = []
        global BENCHMARK_MODE; original_benchmark_mode = BENCHMARK_MODE; BENCHMARK_MODE = True

        for name, func in AVAILABLE_HEURISTICS.items():
            if not self.root.winfo_exists(): break
            if self.root.winfo_exists() and hasattr(self, 'status_var'):
                self.root.after(0, lambda n=name: self.status_var.set(f"Đang thử A* với heuristic: {n}..."))
            try:
                grid, paths, solve_time, states = solve_astar(puzzle_str_copy, func, time_limit=20, state_limit=150000)
                is_solved = grid is not None and paths is not None and len(paths) > 0 and is_grid_full(grid, self.grid_size)

                score = float('inf')
                if is_solved: score = solve_time + states * 0.00001
                elif solve_time < 19.9: score = 1000000 + solve_time + states * 0.00001

                results.append({"name": name, "time": solve_time, "states": states, "score": score, "solved": is_solved})
            except Exception as e:
                results.append({"name": name, "time": float('inf'), "states": float('inf'), "score": float('inf'), "solved": False})
                if not original_benchmark_mode: print(f"Lỗi khi thử heuristic {name}: {e}")

        BENCHMARK_MODE = original_benchmark_mode
        if self.root.winfo_exists():
            self.root.after(0, self.display_heuristic_optimization_results, results)

    def display_qlearning_optimization_results(self, results):
        if not self.root.winfo_exists(): return

        qlearning_results = sorted(results, key=lambda x: x["score"])
        self._update_button_states()

        if not qlearning_results:
            if hasattr(self, 'status_var'): self.status_var.set("Đánh giá Q-Learning configs thất bại.");
            return

        best_config = qlearning_results[0]
        if hasattr(self, 'status_var'):
            self.status_var.set(f"Đánh giá xong. Đề xuất: {best_config['name']}" if best_config['solved'] else
                            "Đánh giá xong (không config nào giải được)")

        top = tk.Toplevel(self.root)
        top.title("Kết quả đánh giá Configs cho Q-Learning")
        top.geometry("800x400")
        top.configure(bg=self.GALAXY_BG)
        top.transient(self.root)
        top.grab_set()

        cols = ["Config", "Learning Rate", "Discount Factor", "Exploration Rate", "Time (s)", "States", "Solved?"]
        tree_frame = ttk.Frame(top, style="TFrame")
        tree_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", style="Galaxy.Treeview")

        style_top = ttk.Style(top)
        style_top.configure("Galaxy.Treeview.Heading",
                        background=self.GALAXY_BUTTON_BG,
                        foreground=self.GALAXY_STAR,
                        font=("Segoe UI", 10, "bold"),
                        relief="raised")
        style_top.map("Galaxy.Treeview.Heading", background=[('active', self.GALAXY_ACCENT1)])
        tree.configure(style="Galaxy.Treeview")

        col_widths = {
            "Config": 150,
            "Learning Rate": 100,
            "Discount Factor": 120,
            "Exploration Rate": 120,
            "Time (s)": 80,
            "States": 100,
            "Solved?": 80
        }

        for col_name in cols:
            anchor = tk.W if col_name == "Config" else tk.CENTER
            tree.heading(col_name, text=col_name, anchor=anchor)
            tree.column(col_name, width=col_widths.get(col_name, 100), anchor=anchor, stretch=tk.YES)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        for res in qlearning_results:
            tree.insert("", tk.END, values=(
                res['name'],
                res['config']['learning_rate'],
                res['config']['discount_factor'],
                res['config']['exploration_rate'],
                f"{res['time']:.3f}" if res['time'] != float('inf') else "Inf",
                res['states'],
                str(res['solved'])
            ))

        button_frame_top = ttk.Frame(top, style="TFrame")
        button_frame_top.pack(pady=(0,10))

        if best_config['solved']:
            def apply_best_config():
                self.selected_qlearning_config.set(best_config['name'])
                if self.root.winfo_exists():
                    messagebox.showinfo("Đã áp dụng", f"Đã chọn config '{best_config['name']}' cho Q-Learning.", parent=top)
                top.destroy()
            apply_button = ttk.Button(button_frame_top, text=f"Áp dụng Config '{best_config['name']}'", command=apply_best_config)
            apply_button.pack(side=tk.LEFT, padx=5)
            self.selected_qlearning_config.set(best_config['name'])
        else:
            if self.root.winfo_exists():
                messagebox.showwarning("Không giải được", "Không config nào giải được puzzle trong thời gian giới hạn.", parent=top)

        ttk.Button(button_frame_top, text="Đóng", command=top.destroy).pack(side=tk.LEFT, padx=5)
        top.wait_window()

    def display_heuristic_optimization_results(self, results):
        if not self.root.winfo_exists(): return

        self.heuristic_optimizer_results = sorted(results, key=lambda x: x["score"])
        self._update_button_states()

        if not self.heuristic_optimizer_results:
            if hasattr(self, 'status_var'): self.status_var.set("Đánh giá heuristic thất bại."); return

        best_h = self.heuristic_optimizer_results[0]
        if hasattr(self, 'status_var'): self.status_var.set(f"Đánh giá xong. Đề xuất: {best_h['name']}" if best_h['solved'] else "Đánh giá xong (không heuristic nào giải được)")

        top = tk.Toplevel(self.root); top.title("Kết quả đánh giá Heuristic cho A*"); top.geometry("700x350")
        top.configure(bg=self.GALAXY_BG)
        top.transient(self.root); top.grab_set()

        cols = ["Heuristic", "Time (s)", "States", "Score", "Solved?"]
        tree_frame = ttk.Frame(top, style="TFrame"); tree_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        tree = ttk.Treeview(tree_frame, columns=cols, show="headings", style="Galaxy.Treeview")

        style_top_heuristic = ttk.Style(top)
        style_top_heuristic.configure("Galaxy.Treeview.Heading",
                           background=self.GALAXY_BUTTON_BG,
                           foreground=self.GALAXY_STAR,
                           font=("Segoe UI", 10, "bold"),
                           relief="raised")
        style_top_heuristic.map("Galaxy.Treeview.Heading", background=[('active', self.GALAXY_ACCENT1)])
        tree.configure(style="Galaxy.Treeview")


        col_widths_h = {"Heuristic": 200, "Time (s)": 100, "States": 100, "Score": 100, "Solved?": 80}
        for col_name in cols:
            anchor = tk.W if col_name == "Heuristic" else tk.CENTER
            tree.heading(col_name, text=col_name, anchor=anchor)
            tree.column(col_name, width=col_widths_h.get(col_name, 100), anchor=anchor, stretch=tk.YES)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y); hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        for res in self.heuristic_optimizer_results:
            tree.insert("", tk.END, values=(res['name'], f"{res['time']:.3f}", res['states'], f"{res['score']:.3f}" if res['score'] != float('inf') else "Inf", str(res['solved'])))

        button_frame_top = ttk.Frame(top, style="TFrame")
        button_frame_top.pack(pady=(0,10))

        if best_h['solved']:
            def apply_best_heuristic():
                self.selected_heuristic_name.set(best_h['name'])
                self.on_heuristic_change()
                if self.root.winfo_exists(): messagebox.showinfo("Đã áp dụng", f"Đã chọn heuristic '{best_h['name']}' cho A*.", parent=top)
                top.destroy()
            apply_button = ttk.Button(button_frame_top, text=f"Áp dụng Heuristic '{best_h['name']}'", command=apply_best_heuristic)
            apply_button.pack(side=tk.LEFT, padx=5)
            self.selected_heuristic_name.set(best_h['name']); self.on_heuristic_change()
        else:
            if self.root.winfo_exists(): messagebox.showwarning("Không giải được", "Không heuristic nào giải được puzzle trong thời gian giới hạn.", parent=top)

        ttk.Button(button_frame_top, text="Đóng", command=top.destroy).pack(side=tk.LEFT, padx=5)
        top.wait_window()

    def run_gui_benchmark_suite(self):
        if self.is_animating:
            if self.root.winfo_exists(): messagebox.showwarning("Đang bận", "Vui lòng đợi."); return

        dialog = tk.Toplevel(self.root); dialog.title("Cấu hình Benchmark Suite"); dialog.geometry("520x450")
        dialog.configure(bg=self.GALAXY_BG)
        dialog.transient(self.root); dialog.grab_set()
        ttk.Label(dialog, text="Cấu hình Benchmark Suite:", font=("Segoe UI", 12, "bold"),
                style="GalaxyTitle.TLabel").pack(pady=(10,5))

        main_config_frame = ttk.Frame(dialog, padding="10")
        main_config_frame.pack(expand=True, fill=tk.BOTH)

        limits_frame = ttk.Frame(main_config_frame)
        limits_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0,10))

        lf_time = ttk.LabelFrame(limits_frame, text="Giới hạn thời gian", padding=5)
        lf_time.pack(fill=tk.X, pady=5)
        ttk.Label(lf_time, text="Time Limit / Puzzle (s):").pack(side=tk.TOP, anchor=tk.W, padx=5)
        time_limit_var = tk.DoubleVar(value=15.0); time_limit_entry = ttk.Entry(lf_time, textvariable=time_limit_var, width=15, font=("Segoe UI", 9))
        time_limit_entry.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=(0,5), fill=tk.X)

        lf_state = ttk.LabelFrame(limits_frame, text="Giới hạn trạng thái", padding=5)
        lf_state.pack(fill=tk.X, pady=5)
        ttk.Label(lf_state, text="State Limit (BFS/A*):").pack(side=tk.TOP, anchor=tk.W, padx=5)
        state_limit_var = tk.IntVar(value=50000); state_limit_entry = ttk.Entry(lf_state, textvariable=state_limit_var, width=15, font=("Segoe UI", 9))
        state_limit_entry.pack(side=tk.TOP, anchor=tk.W, padx=5, pady=(0,5), fill=tk.X)

        selection_frame = ttk.Frame(main_config_frame)
        selection_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        lf_algos = ttk.LabelFrame(selection_frame, text="Chọn Thuật toán", padding=5)
        lf_algos.pack(fill=tk.X, pady=5)
        algo_vars = {}; algo_options_frame = ttk.Frame(lf_algos)
        algo_options_frame.pack(fill=tk.X, padx=5, pady=5)
        all_algos_gui = ["Backtracking", "BFS", "A*", "Q-Learning", "Simulated Annealing", "AND-OR Search"] + (["CP"] if ORTOOLS_AVAILABLE else [])
        for idx, name in enumerate(all_algos_gui):
            var = tk.BooleanVar(value=(name in ["A*"] or (name=="CP" and ORTOOLS_AVAILABLE)))
            chk = ttk.Checkbutton(algo_options_frame, text=name, variable=var, style="Galaxy.TCheckbutton")

            chk.grid(row=idx // 2, column=idx % 2, sticky=tk.W, padx=5, pady=2)
            if name == "CP" and not ORTOOLS_AVAILABLE: chk.config(state=tk.DISABLED); var.set(False)
            algo_vars[name] = var

        lf_puzzles = ttk.LabelFrame(selection_frame, text="Chọn Độ khó Puzzle", padding=5)
        lf_puzzles.pack(fill=tk.BOTH, expand=True, pady=5)
        puzzle_vars = {};

        canvas_puzzles = tk.Canvas(lf_puzzles, borderwidth=0, height=100, bg=self.GALAXY_BG, highlightthickness=0)
        scrollbar_puzzles = ttk.Scrollbar(lf_puzzles, orient="vertical", command=canvas_puzzles.yview)
        frame_puzzles_scrollable = ttk.Frame(canvas_puzzles, style="TFrame")
        canvas_puzzles.configure(yscrollcommand=scrollbar_puzzles.set)

        scrollbar_puzzles.pack(side="right", fill="y")
        canvas_puzzles.pack(side="left", fill="both", expand=True)
        canvas_puzzles.create_window((0,0), window=frame_puzzles_scrollable, anchor="nw", tags="frame_puzzles_scrollable")

        def on_frame_configure_puzzles(event): canvas_puzzles.configure(scrollregion=canvas_puzzles.bbox("all"))
        frame_puzzles_scrollable.bind("<Configure>", on_frame_configure_puzzles)

        sorted_diffs = self.difficulty_levels
        for idx, name in enumerate(sorted_diffs):
            var = tk.BooleanVar(value=("Tiny" in name or "Easy" in name))
            chk = ttk.Checkbutton(frame_puzzles_scrollable, text=name, variable=var, style="Galaxy.TCheckbutton")
            chk.pack(anchor=tk.W, padx=5)
            puzzle_vars[name] = var

        def start_benchmark_from_dialog():
            algos = [name for name, var in algo_vars.items() if var.get()]
            puzzles_cfg = {name: self.puzzles[name] for name, var in puzzle_vars.items() if var.get()}
            if not algos: messagebox.showerror("Lỗi", "Chọn ít nhất một thuật toán.", parent=dialog); return
            if not puzzles_cfg: messagebox.showerror("Lỗi", "Chọn ít nhất một độ khó.", parent=dialog); return

            tl = time_limit_var.get(); sl = state_limit_var.get()
            dialog.destroy()
            if hasattr(self, 'status_var'): self.status_var.set("Đang chạy Benchmark Suite từ GUI...");
            if self.root.winfo_exists(): self.root.update_idletasks()
            threading.Thread(target=self._run_benchmark_task_gui,
                               args=(puzzles_cfg, algos, tl, sl), daemon=True).start()

        btn_frame_dialog = ttk.Frame(dialog, style="TFrame"); btn_frame_dialog.pack(pady=10, fill=tk.X)
        ttk.Button(btn_frame_dialog, text="Bắt đầu Benchmark", command=start_benchmark_from_dialog).pack(side=tk.RIGHT, padx=10)
        ttk.Button(btn_frame_dialog, text="Hủy", command=dialog.destroy).pack(side=tk.RIGHT)

        dialog.update_idletasks()
        canvas_puzzles.configure(scrollregion=canvas_puzzles.bbox("all"))
        dialog.wait_window()


    def _run_benchmark_task_gui(self, puzzles_to_test, algorithms_to_run, time_limit, state_limit):
        if not BENCHMARK_MODE: print("Bắt đầu tác vụ benchmark từ GUI...")

        original_states = {}
        button_attribute_names = [
            'solve_button', 'reset_button', 'benchmark_button',
            'optimize_heuristic_button', 'charts_button'
        ]

        if self.root.winfo_exists():
            for attr_name in button_attribute_names:
                if hasattr(self, attr_name):
                    btn = getattr(self, attr_name)
                    if btn and btn.winfo_exists():
                        original_states[attr_name] = btn.cget('state')
                        btn.config(state=tk.DISABLED)

        self.benchmark_results_data = run_benchmark_suite(puzzles_to_test, algorithms_to_run, time_limit, state_limit)

        def benchmark_gui_finish():
            if not self.root.winfo_exists(): return
            for attr_name, state in original_states.items():
                if hasattr(self, attr_name):
                    btn = getattr(self, attr_name)
                    try:
                        if btn and btn.winfo_exists(): btn.config(state=state)
                    except tk.TclError: pass

            self._update_button_states()

            if self.benchmark_results_data:
                self.display_benchmark_suite_results_toplevel(self.benchmark_results_data, time_limit, state_limit)
                if hasattr(self, 'status_var'): self.status_var.set(f"Benchmark hoàn tất. Xem/Lưu kết quả hoặc Vẽ biểu đồ.")
            else:
                if hasattr(self, 'status_var'): self.status_var.set("Benchmark hoàn tất (không có kết quả).")
                if self.root.winfo_exists(): messagebox.showinfo("Benchmark Hoàn Tất", "Benchmark chạy xong nhưng không có kết quả.", parent=self.root)
            if not BENCHMARK_MODE: print(f"Benchmark từ GUI hoàn tất.")

        if self.root.winfo_exists(): self.root.after(0, benchmark_gui_finish)

    def display_benchmark_suite_results_toplevel(self, results, time_limit_cfg, state_limit_cfg):
        if not self.root.winfo_exists(): return

        top = tk.Toplevel(self.root)
        top.title(f"Kết quả Benchmark Suite (Time Limit: {time_limit_cfg}s, State Limit: {state_limit_cfg})")
        top.geometry("950x550")
        top.configure(bg=self.GALAXY_BG)
        top.transient(self.root); top.grab_set()

        tree_frame_benchmark = ttk.Frame(top, style="TFrame"); tree_frame_benchmark.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)
        cols = ["Difficulty", "Puzz", "Size", "Colors", "Algorithm", "Time(s)", "States", "Found", "Status"]
        tree = ttk.Treeview(tree_frame_benchmark, columns=cols, show="headings", style="Galaxy.Treeview")

        style_top = ttk.Style(top)
        style_top.configure("Galaxy.Treeview.Heading",
                           background=self.GALAXY_BUTTON_BG,
                           foreground=self.GALAXY_STAR,
                           font=("Segoe UI", 10, "bold"),
                           relief="raised")
        style_top.map("Galaxy.Treeview.Heading", background=[('active', self.GALAXY_ACCENT1)])
        tree.configure(style="Galaxy.Treeview")


        col_widths = {"Difficulty": 150, "Puzz": 50, "Size": 60, "Colors": 60,
                      "Algorithm": 110, "Time(s)": 80, "States": 90, "Found": 60, "Status": 150}
        for col_name in cols:
            anchor = tk.W if col_name in ["Difficulty", "Algorithm", "Status"] else tk.CENTER
            tree.heading(col_name, text=col_name, anchor=anchor)
            tree.column(col_name, width=col_widths.get(col_name, 80), anchor=anchor, stretch=tk.YES)

        vsb = ttk.Scrollbar(tree_frame_benchmark, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame_benchmark, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y); hsb.pack(side=tk.BOTTOM, fill=tk.X)
        tree.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)

        for row_data in results:
            values_to_print = [
                row_data["difficulty"], row_data["puzzle_index"], row_data["puzzle_size"],
                row_data["num_colors"], row_data["algorithm"],
                f"{row_data['time_taken']:.3f}" if row_data['time_taken'] != -1 else "N/A",
                str(row_data["states_explored"]) if row_data['states_explored'] != -1 else "N/A",
                str(row_data["solution_found"]), row_data["status"]]
            tree.insert("", tk.END, values=values_to_print)

        button_frame_top = ttk.Frame(top, style="TFrame")
        button_frame_top.pack(pady=(5,10))
        def save_results():
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"gui_benchmark_results_{timestamp}.csv"
            save_benchmark_results_csv(results, filename)
            if self.root.winfo_exists(): messagebox.showinfo("Đã lưu", f"Kết quả đã được lưu vào:\n{filename}", parent=top)
        save_button = ttk.Button(button_frame_top, text="Lưu kết quả (CSV)", command=save_results)
        save_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame_top, text="Đóng", command=top.destroy).pack(side=tk.LEFT, padx=5)

        top.wait_window()

    def show_benchmark_charts(self):
            if not MATPLOTLIB_AVAILABLE:
                if self.root.winfo_exists(): messagebox.showerror("Lỗi", "Thư viện Matplotlib chưa được cài đặt để vẽ biểu đồ.")
                return
            if not self.benchmark_results_data:
                if self.root.winfo_exists(): messagebox.showinfo("Thiếu dữ liệu", "Vui lòng chạy Benchmark Suite trước để có dữ liệu vẽ biểu đồ.")
                return

            chart_window = tk.Toplevel(self.root)
            chart_window.title("Biểu đồ Benchmark")
            chart_window.geometry("1000x750")
            chart_window.configure(bg=self.GALAXY_BG)
            chart_window.transient(self.root); chart_window.grab_set()

            style_chart = ttk.Style(chart_window)
            style_chart.configure("Chart.TNotebook", tabmargins=[2, 5, 2, 0], background=self.GALAXY_BG)
            style_chart.configure("Chart.TNotebook.Tab", padding=[10, 5], font=("Segoe UI", 10, "bold"),
                                background=self.GALAXY_BUTTON_BG, foreground=self.GALAXY_STAR)
            style_chart.map("Chart.TNotebook.Tab",
                        background=[("selected", self.GALAXY_ACCENT1)],
                        foreground=[("selected", self.GALAXY_STAR)],
                        expand=[("selected", [1, 1, 1, 0])])


            notebook = ttk.Notebook(chart_window, style="Chart.TNotebook")
            notebook.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

            def get_bar_colors(num_bars):
                if not self.flow_colors: return ["#CCCCCC"] * num_bars
                plotting_colors = self.flow_colors[1:]
                if not plotting_colors: plotting_colors = [self.flow_colors[0]]
                if not plotting_colors: return ["#AAAAAA"] * num_bars
                bar_colors_list = [plotting_colors[i % len(plotting_colors)] for i in range(num_bars)]
                return bar_colors_list

            # --- Tab 1: Thời gian giải cho TẤT CẢ PUZZLE trong TẤT CẢ MỨC ĐỘ ĐÃ BENCHMARK ---
            tab1_frame = ttk.Frame(notebook, style="TFrame")
            tab1_title = "TG Giải / Puzzle (Các mức độ đã Benchmark)"
            notebook.add(tab1_frame, text=tab1_title)

            if not self.benchmark_results_data:
                 ttk.Label(tab1_frame, text="Không có dữ liệu benchmark để hiển thị.",
                           style="GalaxyTitle.TLabel", justify=tk.CENTER).pack(padx=20, pady=20, expand=True, fill=tk.BOTH)
            else:
                # Dữ liệu: { "Difficulty_Name": { "Puzzle X": {algo: time}, ... }, ...}
                all_puzzles_times_across_difficulties = {}
                for res in self.benchmark_results_data:
                    if res['solution_found']:
                        difficulty_name = res['difficulty']
                        puzzle_display_name = f"Puzzle {res['puzzle_index']}"
                        algo = res['algorithm']
                        time_val = res['time_taken']

                        if difficulty_name not in all_puzzles_times_across_difficulties:
                            all_puzzles_times_across_difficulties[difficulty_name] = {}
                        if puzzle_display_name not in all_puzzles_times_across_difficulties[difficulty_name]:
                            all_puzzles_times_across_difficulties[difficulty_name][puzzle_display_name] = {}
                        all_puzzles_times_across_difficulties[difficulty_name][puzzle_display_name][algo] = time_val

                if not all_puzzles_times_across_difficulties:
                    ttk.Label(tab1_frame, text="Không có dữ liệu giải thành công cho bất kỳ puzzle nào.",
                              style="GalaxyTitle.TLabel", justify=tk.CENTER).pack(padx=20, pady=20, expand=True, fill=tk.BOTH)
                else:
                    canvas_scroll_tab1 = tk.Canvas(tab1_frame, bg=self.GALAXY_BG, highlightthickness=0)
                    scrollbar_y_tab1 = ttk.Scrollbar(tab1_frame, orient="vertical", command=canvas_scroll_tab1.yview)
                    scrollable_frame_tab1 = ttk.Frame(canvas_scroll_tab1, style="TFrame")

                    scrollable_frame_tab1.bind(
                        "<Configure>",
                        lambda e: canvas_scroll_tab1.configure(scrollregion=canvas_scroll_tab1.bbox("all"))
                    )
                    canvas_scroll_tab1.create_window((0, 0), window=scrollable_frame_tab1, anchor="nw")
                    canvas_scroll_tab1.configure(yscrollcommand=scrollbar_y_tab1.set)
                    canvas_scroll_tab1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    scrollbar_y_tab1.pack(side=tk.RIGHT, fill=tk.Y)

                    # Sắp xếp các mức độ khó theo thứ tự trong self.difficulty_levels
                    sorted_difficulties_in_data = sorted(
                        all_puzzles_times_across_difficulties.keys(),
                        key=lambda d_name: self.difficulty_levels.index(d_name) if d_name in self.difficulty_levels else float('inf')
                    )

                    for difficulty_name in sorted_difficulties_in_data:
                        puzzles_in_this_difficulty = all_puzzles_times_across_difficulties[difficulty_name]
                        if not puzzles_in_this_difficulty:
                            continue

                        # Nhãn cho nhóm mức độ khó
                        ttk.Label(scrollable_frame_tab1, text=f"Mức Độ: {difficulty_name}",
                                  style="GalaxyTitle.TLabel", font=("Segoe UI", 14, "bold")).pack(pady=(15, 5), fill=tk.X, padx=10)

                        sorted_puzzle_names_in_difficulty = sorted(
                            puzzles_in_this_difficulty.keys(),
                            key=lambda name: int(name.split(" ")[1])
                        )

                        for puzzle_name in sorted_puzzle_names_in_difficulty:
                            algo_times_for_puzzle = puzzles_in_this_difficulty[puzzle_name]
                            if not algo_times_for_puzzle:
                                continue

                            puzzle_chart_frame = ttk.LabelFrame(scrollable_frame_tab1, text=puzzle_name, style="TLabelframe")
                            puzzle_chart_frame.pack(pady=10, padx=10, fill=tk.X, expand=False)

                            fig_single_puzzle, ax_single_puzzle = plt.subplots(figsize=(8, 4.5), facecolor=self.GALAXY_BG)
                            ax_single_puzzle.set_facecolor("#1c0033")

                            algos = list(algo_times_for_puzzle.keys())
                            times_vals = list(algo_times_for_puzzle.values())
                            bar_colors = get_bar_colors(len(algos))
                            bars = ax_single_puzzle.bar(algos, times_vals, color=bar_colors, width=0.5)

                            ax_single_puzzle.set_ylabel('Thời gian giải (giây)', color=self.GALAXY_FG, fontsize=9)
                            ax_single_puzzle.tick_params(axis='x', colors=self.GALAXY_FG, rotation=20, labelsize=8)
                            if plt: plt.setp(ax_single_puzzle.get_xticklabels(), ha="right")
                            ax_single_puzzle.tick_params(axis='y', colors=self.GALAXY_FG, labelsize=8)

                            for spine_pos in ['top', 'bottom', 'left', 'right']:
                                ax_single_puzzle.spines[spine_pos].set_color(self.GALAXY_ACCENT1)

                            max_time_val = max(times_vals) if times_vals else 1
                            for bar in bars:
                                yval = bar.get_height()
                                ax_single_puzzle.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02 * max_time_val,
                                        f'{yval:.2f}s', ha='center', va='bottom', color=self.GALAXY_STAR, fontsize=7)

                            plt.tight_layout(pad=1.0)
                            canvas_single = FigureCanvasTkAgg(fig_single_puzzle, master=puzzle_chart_frame)
                            canvas_single.draw()
                            canvas_single.get_tk_widget().pack(side=tk.TOP, fill=tk.X, expand=True, pady=(5,0))
                            plt.close(fig_single_puzzle)

            # --- Tab 2: Số thuật toán giải được / Puzzle (TẤT CẢ MỨC ĐỘ ĐÃ BENCHMARK) ---
            tab2_frame = ttk.Frame(notebook, style="TFrame")
            notebook.add(tab2_frame, text="Số Algo Giải Được / Puzzle (Tất cả)")

            if not self.benchmark_results_data:
                 ttk.Label(tab2_frame, text="Không có dữ liệu benchmark để hiển thị.",
                           style="GalaxyTitle.TLabel", justify=tk.CENTER).pack(padx=20, pady=20, expand=True, fill=tk.BOTH)
            else:
                # Dữ liệu: { (difficulty, puzzle_index): count, ... }
                solved_counts_all_puzzles = {}
                for res in self.benchmark_results_data:
                    if res['solution_found']:
                        # Tạo một key duy nhất cho mỗi puzzle trên tất cả các mức độ
                        puzzle_key = (res['difficulty'], res['puzzle_index'])
                        if puzzle_key not in solved_counts_all_puzzles:
                            solved_counts_all_puzzles[puzzle_key] = 0
                        solved_counts_all_puzzles[puzzle_key] += 1

                if solved_counts_all_puzzles:
                    # Sắp xếp các puzzle để hiển thị theo thứ tự
                    sorted_puzzle_keys = sorted(
                        solved_counts_all_puzzles.keys(),
                        key=lambda pk: (self.difficulty_levels.index(pk[0]) if pk[0] in self.difficulty_levels else float('inf'), pk[1])
                    )
                    
                    puzzle_labels_tab2 = [f"{pk[0]} P{pk[1]}" for pk in sorted_puzzle_keys]
                    solve_counts_tab2 = [solved_counts_all_puzzles[pk] for pk in sorted_puzzle_keys]
                    
                    # Có thể cần điều chỉnh figsize nếu số lượng puzzle quá lớn
                    fig_height_tab2 = max(5, len(puzzle_labels_tab2) * 0.3) # Chiều cao động
                    fig_width_tab2 = 10

                    fig2, ax2 = plt.subplots(figsize=(fig_width_tab2, fig_height_tab2), facecolor=self.GALAXY_BG)
                    ax2.set_facecolor("#1c0033")

                    bar_colors2 = get_bar_colors(len(puzzle_labels_tab2))
                    # Vẽ biểu đồ ngang nếu có nhiều puzzle
                    if len(puzzle_labels_tab2) > 15: # Ngưỡng để chuyển sang biểu đồ ngang
                        bars2 = ax2.barh(puzzle_labels_tab2, solve_counts_tab2, color=bar_colors2)
                        ax2.set_xlabel('Số thuật toán giải được', color=self.GALAXY_FG)
                        ax2.set_ylabel('Puzzle', color=self.GALAXY_FG)
                        ax2.tick_params(axis='y', colors=self.GALAXY_FG, labelsize=8)
                        ax2.tick_params(axis='x', colors=self.GALAXY_FG, labelsize=8)
                        if plt: ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
                        for bar in bars2: # Add text cho barh
                            xval = bar.get_width()
                            ax2.text(xval + 0.05, bar.get_y() + bar.get_height()/2.0, int(xval), ha='left', va='center', color=self.GALAXY_STAR, fontsize=7)

                    else: # Biểu đồ đứng
                        bars2 = ax2.bar(puzzle_labels_tab2, solve_counts_tab2, color=bar_colors2)
                        ax2.set_xlabel('Puzzle', color=self.GALAXY_FG)
                        ax2.set_ylabel('Số thuật toán giải được', color=self.GALAXY_FG)
                        ax2.tick_params(axis='x', colors=self.GALAXY_FG, rotation=30, labelsize=8, ha="right")
                        ax2.tick_params(axis='y', colors=self.GALAXY_FG, labelsize=8)
                        if plt: ax2.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                        for bar in bars2: # Add text cho bar
                            yval = bar.get_height()
                            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', va='bottom', color=self.GALAXY_STAR, fontsize=8)


                    ax2.set_title(f'Số thuật toán giải được cho tất cả puzzle đã benchmark', color=self.GALAXY_STAR, fontsize=12)
                    for spine_pos in ['top', 'bottom', 'left', 'right']:
                        ax2.spines[spine_pos].set_color(self.GALAXY_ACCENT1)
                    
                    plt.tight_layout(pad=1.5)
                    canvas2 = FigureCanvasTkAgg(fig2, master=tab2_frame)
                    canvas2.draw()
                    canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    plt.close(fig2)
                else:
                    ttk.Label(tab2_frame, text="Không có dữ liệu giải thành công cho bất kỳ puzzle nào.",
                              style="GalaxyTitle.TLabel", justify=tk.CENTER).pack(padx=20, pady=20, expand=True, fill=tk.BOTH)

            # --- Tab 3: Số Puzzle Giải Được / Thuật Toán ---
            tab3_frame = ttk.Frame(notebook, style="TFrame")
            notebook.add(tab3_frame, text="Số Puzzle Giải Được / Thuật Toán (Tổng)")

            if not self.benchmark_results_data:
                 ttk.Label(tab3_frame, text="Không có dữ liệu benchmark để hiển thị.",
                           style="GalaxyTitle.TLabel", justify=tk.CENTER).pack(padx=20, pady=20, expand=True, fill=tk.BOTH)
            else:
                solved_puzzles_per_algorithm = {}
                for res in self.benchmark_results_data:
                    algo = res['algorithm']
                    if algo not in solved_puzzles_per_algorithm:
                        solved_puzzles_per_algorithm[algo] = set() # Dùng set để tránh đếm trùng puzzle
                    if res['solution_found']:
                        puzzle_identifier = (res['difficulty'], res['puzzle_index'])
                        solved_puzzles_per_algorithm[algo].add(puzzle_identifier)

                final_counts_per_algorithm = {
                    algo: len(puzzle_set) for algo, puzzle_set in solved_puzzles_per_algorithm.items()
                }
                sorted_final_counts = dict(sorted(
                    filter(lambda item: item[1] > 0, final_counts_per_algorithm.items()),
                    key=lambda item: item[1],
                    reverse=True
                ))

                if sorted_final_counts:
                    fig3, ax3 = plt.subplots(figsize=(8, 5), facecolor=self.GALAXY_BG)
                    ax3.set_facecolor("#1c0033")
                    algorithms_tab3 = list(sorted_final_counts.keys())
                    counts_tab3 = list(sorted_final_counts.values())
                    bar_colors_tab3 = get_bar_colors(len(algorithms_tab3))
                    bars_tab3 = ax3.bar(algorithms_tab3, counts_tab3, color=bar_colors_tab3)

                    ax3.set_ylabel('Tổng số Puzzle giải được', color=self.GALAXY_FG)
                    ax3.set_title('Tổng số Puzzle giải được bởi mỗi Thuật toán', color=self.GALAXY_STAR, fontsize=14)
                    ax3.tick_params(axis='x', colors=self.GALAXY_FG, rotation=25, labelsize=9)
                    if plt: plt.setp(ax3.get_xticklabels(), ha="right")
                    ax3.tick_params(axis='y', colors=self.GALAXY_FG)
                    for spine_pos in ['top', 'bottom', 'left', 'right']:
                        ax3.spines[spine_pos].set_color(self.GALAXY_ACCENT1)
                    if plt: ax3.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
                    max_counts_tab3 = max(counts_tab3) if counts_tab3 else 1
                    for bar in bars_tab3:
                        yval = bar.get_height()
                        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max_counts_tab3,
                                f'{int(yval)}', ha='center', va='bottom', color=self.GALAXY_STAR, fontsize=9)

                    plt.tight_layout(pad=2.0)
                    canvas3 = FigureCanvasTkAgg(fig3, master=tab3_frame)
                    canvas3.draw()
                    canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                    plt.close(fig3)
                else:
                    msg_tab3 = "Không có thuật toán nào giải thành công puzzle nào trong benchmark."
                    ttk.Label(tab3_frame, text=msg_tab3, style="GalaxyTitle.TLabel",
                              justify=tk.CENTER).pack(padx=20, pady=20, expand=True, fill=tk.BOTH)

            ttk.Button(chart_window, text="Đóng", command=chart_window.destroy).pack(pady=10)
            chart_window.wait_window()
# ============================================================
# BENCHMARKING SUITE
# ============================================================
def run_benchmark_suite(puzzles_to_test, algorithms_to_run, time_limit_per_run=60, state_limit_per_run=500000):
    global BENCHMARK_MODE; original_bm_mode = BENCHMARK_MODE; BENCHMARK_MODE = True
    results = []
    total_puzzles = sum(len(p_list) for p_list in puzzles_to_test.values())
    current_puzzle_count = 0
    is_gui_call = hasattr(threading.current_thread(), "_gui_calling_thread")

    if not is_gui_call:
        print("===== BẮT ĐẦU BENCHMARK SUITE =====")
        print(f"Tổng puzzles: {total_puzzles}, Thuật toán: {', '.join(algorithms_to_run)}")
        print(f"Time limit: {time_limit_per_run}s, State limit: {state_limit_per_run}")
        print("-" * 70)

    for difficulty, puzzle_list in puzzles_to_test.items():
        if not is_gui_call:
            print(f"\n--- Độ khó: {difficulty} ---")
        for i, puzzle_str in enumerate(puzzle_list):
            current_puzzle_count += 1
            parsed_grid_initial, parsed_colors, parsed_size, _ = parse_puzzle_extended(puzzle_str)
            num_colors = len(parsed_colors) if parsed_colors else 0

            if parsed_size == 0 or not parsed_colors or parsed_grid_initial is None:
                status_msg = "Skipped (Invalid Puzzle Data)"
                if not is_gui_call: print(f"  Puzzle {i+1} ({status_msg})")
                for algo_name_skip in algorithms_to_run:
                     results.append({"difficulty": difficulty, "puzzle_index": i + 1, "puzzle_size": f"{parsed_size}x{parsed_size}",
                        "num_colors": num_colors, "algorithm": algo_name_skip, "time_taken": -1.0,
                        "states_explored": -1, "solution_found": False, "status": status_msg})
                continue

            if not is_gui_call:
                print(f"\n  Puzzle {i+1}/{len(puzzle_list)} ({parsed_size}x{parsed_size}, {num_colors} màu) (Tổng: {current_puzzle_count}/{total_puzzles})")

            for algo_name in algorithms_to_run:
                if algo_name == "CP" and not ORTOOLS_AVAILABLE:
                    if not is_gui_call: print(f"    Bỏ qua {algo_name} (OR-Tools không có sẵn).")
                    results.append({"difficulty": difficulty, "puzzle_index": i + 1, "puzzle_size": f"{parsed_size}x{parsed_size}",
                        "num_colors": num_colors, "algorithm": algo_name, "time_taken": -1.0,
                        "states_explored": -1, "solution_found": False, "status": "Skipped (OR-Tools unavailable)"})
                    continue

                if not is_gui_call: print(f"    Đang chạy {algo_name}...")
                grid, paths, time_taken, states = None, None, 0.0, 0
                status, found = "Error", False
                start_run_actual = time.time()

                try:
                    if algo_name == "Backtracking": grid, paths, time_taken, states = solve_backtracking(puzzle_str, time_limit_per_run)
                    elif algo_name == "BFS": grid, paths, time_taken, states = solve_bfs(puzzle_str, time_limit_per_run, state_limit_per_run)
                    elif algo_name == "A*": grid, paths, time_taken, states = solve_astar(puzzle_str, h_manhattan_sum, time_limit_per_run, state_limit_per_run) # Default heuristic for benchmark
                    elif algo_name == "CP": grid, paths, time_taken, states = solve_cp(puzzle_str, time_limit_per_run)
                    elif algo_name == "Simulated Annealing":
                        grid, paths, time_taken, states = solve_simulated_annealing(puzzle_str,time_limit=time_limit_per_run)
                    elif algo_name == "AND-OR Search":
                        grid, paths, time_taken, states = solve_and_or_search(puzzle_str, time_limit=time_limit_per_run)                    
                    elif algo_name == "Q-Learning":
                        q_config_for_benchmark = AVAILABLE_QLEARNING_CONFIGS["Default"]
                        episodes_for_benchmark = q_config_for_benchmark.get("episodes", 500)
                        grid, paths, time_taken, states = solve_qlearning(puzzle_str,time_limit=time_limit_per_run,episodes=episodes_for_benchmark,config=q_config_for_benchmark)
                    else: status = "Unknown Algorithm"; states = -1; time_taken = 0.0;

                    if grid is not None and paths is not None:
                        if len(paths) == num_colors:
                            all_paths_valid_format = True
                            for p_color, p_coords in paths.items():
                                if not isinstance(p_coords, list) or len(p_coords) < 2:
                                    all_paths_valid_format = False; break
                                start_node = parsed_colors[p_color]['start']
                                end_node = parsed_colors[p_color]['end']
                                if p_coords[0] != start_node or p_coords[-1] != end_node:
                                    all_paths_valid_format = False; break
                            if all_paths_valid_format and is_grid_full(grid, parsed_size):
                                found = True
                    else:
                        found = False

                    if found: status = "Solved"
                    elif time_taken >= time_limit_per_run * 0.99 : status = "Timed Out"
                    elif algo_name in ["BFS", "A*"] and states >= state_limit_per_run and not found : status = "State Limit Reached"
                    elif grid is None and paths is None: status = "No Solution Data"
                    else: status = "Incomplete/Invalid Solution"

                except Exception as e:
                    time_taken = time.time() - start_run_actual
                    status = f"Crash: {str(e)[:50]}"
                    found = False
                    if not is_gui_call:
                        print(f"      LỖI NGHIÊM TRỌNG khi chạy {algo_name} cho puzzle {difficulty} - {i+1}:")
                        traceback.print_exc()

                results.append({"difficulty": difficulty, "puzzle_index": i + 1, "puzzle_size": f"{parsed_size}x{parsed_size}",
                    "num_colors": num_colors, "algorithm": algo_name, "time_taken": round(time_taken, 4),
                    "states_explored": states, "solution_found": found, "status": status})
                if not is_gui_call:
                    print(f"      Kết quả: Time={round(time_taken, 2):.2f}s, States={states}, Found={found}, Status='{status}'")

    if not is_gui_call:
        print("\n===== BENCHMARK SUITE HOÀN TẤT =====")
    BENCHMARK_MODE = original_bm_mode
    return results


def display_benchmark_results_pretty(results):
    if not results: print("Không có kết quả benchmark."); return
    print("\n\n--- TỔNG KẾT BENCHMARK (CONSOLE) ---")
    headers = ["Difficulty", "Puzz", "Size", "Colors", "Algorithm", "Time(s)", "States", "Found", "Status"]
    col_widths = [len(h) for h in headers]
    for row in results:
        vals = [str(row["difficulty"]), str(row["puzzle_index"]), str(row["puzzle_size"]), str(row["num_colors"]),
                str(row["algorithm"]), f"{row['time_taken']:.2f}" if row['time_taken']!=-1 else "N/A",
                str(row["states_explored"]) if row["states_explored"]!=-1 else "N/A", str(row["solution_found"]), str(row["status"])]
        for i, v in enumerate(vals): col_widths[i] = max(col_widths[i], len(v))
    fmt_str = " | ".join([f"{{:<{w}}}" for w in col_widths])
    print(fmt_str.format(*headers))
    print("-+-".join(["-" * w for w in col_widths]))
    for row in results:
        vals_to_print = [row["difficulty"], row["puzzle_index"], row["puzzle_size"], row["num_colors"],
                         row["algorithm"], f"{row['time_taken']:.2f}" if row['time_taken']!=-1 else "N/A",
                         str(row["states_explored"]) if row["states_explored"]!=-1 else "N/A",
                         row["solution_found"], row["status"]]
        print(fmt_str.format(*[str(v) for v in vals_to_print]))

def save_benchmark_results_csv(results, filename="benchmark_results.csv"):
    if not results: print("Không có kết quả để lưu CSV."); return
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            if not results: return
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader(); writer.writerows(results)
        print(f"\nKết quả benchmark đã lưu vào: {filename}")
    except Exception as e: print(f"Lỗi khi lưu CSV: {e}")

# ============================================================
# CHẠY ỨNG DỤNG HOẶC BENCHMARK
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Flow Free Solver and Benchmark Tool")
    parser.add_argument("--run_benchmark", action="store_true", help="Chạy benchmark thay vì GUI.")
    parser.add_argument("--benchmark_time_limit", type=float, default=20.0, help="Time limit (s) / puzzle.")
    parser.add_argument("--benchmark_state_limit", type=int, default=100000, help="State limit (BFS/A*).")
    default_algos = "A*,BFS,Backtracking,Simulated Annealing,Q-Learning,AND-OR Search,CP"
    if ORTOOLS_AVAILABLE:
        default_algos += ",CP"
    parser.add_argument("--algorithms", type=str, default=default_algos, help="Algorithms (comma-separated). 'ALL_AVAILABLE' for all.")
    parser.add_argument("--puzzles", type=str, default="Tiny (3x3),Easy (5x5)", help="Difficulties (comma-separated). 'ALL' for all.")
    parser.add_argument("--output_csv", type=str, default=f"benchmark_{time.strftime('%Y%m%d_%H%M%S')}.csv", help="Output CSV filename.")
    args = parser.parse_args()

    if args.run_benchmark:
        BENCHMARK_MODE = True
        print("Chế độ Benchmark.")
        if not MATPLOTLIB_AVAILABLE:
            print("Lưu ý: Matplotlib không khả dụng, sẽ không thể vẽ biểu đồ nếu có chức năng đó từ dòng lệnh.")

        available_algos_cli = ["Backtracking", "BFS", "A*", "Simulated Annealing", "Q-Learning", "AND-OR Search"]
        if ORTOOLS_AVAILABLE:
            available_algos_cli.append("CP")

        algos_to_test_str = args.algorithms.upper()
        if algos_to_test_str == "ALL_AVAILABLE":
            algos_to_test = available_algos_cli
        else:
            algos_to_test = [a.strip() for a in args.algorithms.split(',') if a.strip()]

        final_algos = []
        for algo_name_cli in algos_to_test:
            matched_algo = None
            for available_one in available_algos_cli:
                if algo_name_cli.lower() == available_one.lower():
                    matched_algo = available_one
                    break
            if matched_algo:
                final_algos.append(matched_algo)
            else:
                print(f"Cảnh báo: Thuật toán '{algo_name_cli}' không hợp lệ hoặc không có sẵn, sẽ được bỏ qua.")

        if not final_algos: print("Lỗi: Không có thuật toán hợp lệ để chạy. Thoát."); sys.exit(1)

        puzzles_bench = {}
        if args.puzzles.upper() == "ALL": puzzles_bench = PUZZLES
        else:
            for diff_cli in [d.strip() for d in args.puzzles.split(',')]:
                matched_diff_key = None
                for p_key in PUZZLES.keys():
                    if diff_cli.lower() == p_key.lower():
                        matched_diff_key = p_key
                        break
                if matched_diff_key:
                    puzzles_bench[matched_diff_key] = PUZZLES[matched_diff_key]
                else:
                    print(f"Cảnh báo: Độ khó '{diff_cli}' không tìm thấy.")
            if not puzzles_bench: print("Lỗi: Không có puzzles hợp lệ. Thoát."); sys.exit(1)

        print(f"Benchmark trên độ khó: {', '.join(puzzles_bench.keys())}")
        print(f"Sử dụng thuật toán: {', '.join(final_algos)}")

        benchmark_data = run_benchmark_suite(puzzles_bench, final_algos, args.benchmark_time_limit, args.benchmark_state_limit)
        if benchmark_data:
            display_benchmark_results_pretty(benchmark_data)
            save_benchmark_results_csv(benchmark_data, args.output_csv)
        else:
            print("Không có dữ liệu benchmark nào được tạo ra.")
    else:
        BENCHMARK_MODE = False
        if ORTOOLS_AVAILABLE: print("OR-Tools đã import thành công.")
        else: print("Cảnh báo: OR-Tools không có sẵn. Chức năng CP sẽ bị vô hiệu hóa.")
        if MATPLOTLIB_AVAILABLE: print("Matplotlib đã import thành công, chức năng vẽ biểu đồ khả dụng.")

        print("Khởi tạo ứng dụng Flow Free Solver (GUI)...")
        main_root = tk.Tk()
        threading.current_thread()._gui_calling_thread = True
        app = FlowFreeApp(main_root)
        print("Chạy vòng lặp chính Tkinter...")
        try:
            main_root.mainloop()
        except KeyboardInterrupt:
            print("\nĐã dừng bởi người dùng.")
        finally:
            try:
                if 'app' in locals() and hasattr(app, 'root') and app.root is not None:
                    try:
                        if app.root.winfo_exists():
                            print("Ứng dụng đang đóng...")
                        else:
                            print("Ứng dụng đã đóng.")
                    except (tk.TclError, RuntimeError):
                        print("Ứng dụng đã được đóng.")
                else:
                    print("Ứng dụng chưa khởi tạo hoàn chỉnh.")
            except Exception as e:
                print(f"Lỗi khi đóng ứng dụng: {e}")