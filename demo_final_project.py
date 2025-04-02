import pygame
import sys
import random
from collections import deque
import time

# Khởi tạo pygame
pygame.init()

# Cấu hình trò chơi
WIDTH, HEIGHT = 400, 400
ROWS, COLS = 6, 6  # Mở rộng thêm 1 viền
CELL_SIZE = WIDTH // COLS

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 165, 0)]

# Tạo cửa sổ
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Flow Free")

# Lưu trạng thái bảng
board = [[None for _ in range(COLS)] for _ in range(ROWS)]
connections = []
occupied_cells = {}

def bfs_path(start, end, color):
    """ Tìm đường đi ngắn nhất bằng thuật toán BFS tránh các ô đã đi và ô có màu khác """
    queue = deque([(start, [])])
    visited = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path + [(r, c)]
        
        if (r, c) in visited:
            continue
        
        visited.add((r, c))
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < ROWS and 0 <= nc < COLS and (nr, nc) not in visited and
                ((nr, nc) not in occupied_cells or occupied_cells[(nr, nc)] == color)):
                queue.append(((nr, nc), path + [(r, c)]))
    
    return []

def generate_valid_board():
    """ Tạo bàn cờ hợp lệ luôn có đường đi cho tất cả các màu và không ô nào trống """
    global board, pairs, occupied_cells
    
    while True:
        board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        occupied_cells = {}
        pairs = []
        points = [(r, c) for r in range(ROWS) for c in range(COLS)]
        random.shuffle(points)
        valid = True
        
        for color in COLORS:
            found = False
            for _ in range(100):  # Giới hạn số lần thử
                p1, p2 = random.sample(points, 2)
                path = bfs_path(p1, p2, color)
                if path:
                    pairs.append((p1, p2, color))
                    board[p1[0]][p1[1]] = color
                    board[p2[0]][p2[1]] = color
                    occupied_cells[p1] = color
                    occupied_cells[p2] = color
                    for cell in path:
                        occupied_cells[cell] = color
                    points.remove(p1)
                    points.remove(p2)
                    found = True
                    break
            if not found:
                valid = False
                break
        
        if valid and len(occupied_cells) == ROWS * COLS:
            return

def draw_board():
    screen.fill(WHITE)
    for row in range(ROWS):
        for col in range(COLS):
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BLACK, rect, 1)
            if board[row][col]:
                pygame.draw.circle(screen, board[row][col], (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3)
    
    for conn in connections:
        pygame.draw.line(screen, conn[2], 
                         (conn[1] * CELL_SIZE + CELL_SIZE // 2, conn[0] * CELL_SIZE + CELL_SIZE // 2),
                         (conn[3] * CELL_SIZE + CELL_SIZE // 2, conn[2] * CELL_SIZE + CELL_SIZE // 2), 5)
    
    pygame.display.flip()

def animate_paths():
    """ Hiển thị đường đi của tất cả các màu theo từng bước """
    global connections, occupied_cells
    connections = []
    for (r1, c1), (r2, c2), color in pairs:
        path = bfs_path((r1, c1), (r2, c2), color)
        if path:
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i + 1]
                connections.append((r1, c1, r2, c2, color))
                occupied_cells[(r1, c1)] = color
                occupied_cells[(r2, c2)] = color
                draw_board()
                time.sleep(0.2)  # Tạm dừng để tạo hiệu ứng hiển thị từ từ

# Khởi tạo bảng hợp lệ
generate_valid_board()

draw_board()
animate_paths()

# Vòng lặp chính
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
sys.exit()