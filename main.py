import pygame
import random
from collections import deque
import heapq
import time

# Ініціалізація Pygame
pygame.init()

# Налаштування екрану
WIDTH, HEIGHT = 700, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pac-Man")

# Кольори
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (255, 0, 255)
ORANGE = (255, 165, 0)
WHITE = (255, 255, 255)

# Параметри гри
CELL_SIZE = 20
GRID_WIDTH = WIDTH // CELL_SIZE
GRID_HEIGHT = HEIGHT // CELL_SIZE

# Напрями для генерації лабіринту
DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
MAZE_WIDTH = GRID_WIDTH
MAZE_HEIGHT = GRID_HEIGHT

# Генерація лабіринту з використанням алгоритму Прима
def generate_maze(level):
    maze = [['X'] * MAZE_WIDTH for _ in range(MAZE_HEIGHT)]
    for x in range(MAZE_WIDTH):
        maze[0][x] = 'X'
        maze[MAZE_HEIGHT - 1][x] = 'X'
    for y in range(MAZE_HEIGHT):
        maze[y][0] = 'X'
        maze[y][MAZE_WIDTH - 1] = 'X'

    walls_list = []
    start_x = random.randrange(1, MAZE_WIDTH - 1, 2)
    start_y = random.randrange(1, MAZE_HEIGHT - 1, 2)
    maze[start_y][start_x] = '.'

    walls_list.extend([(start_x + dx, start_y + dy, start_x + 2*dx, start_y + 2*dy) for dx, dy in DIRECTIONS])

    while walls_list:
        wx, wy, nx, ny = walls_list.pop(random.randint(0, len(walls_list) - 1))
        if 1 <= nx < MAZE_WIDTH - 1 and 1 <= ny < MAZE_HEIGHT - 1:
            if maze[ny][nx] == 'X':
                maze[wy][wx] = '.'
                maze[ny][nx] = '.'
                for dx, dy in DIRECTIONS:
                    walls_list.append((nx + dx, ny + dy, nx + 2*dx, ny + 2*dy))

    additional_connections = level * 10
    for _ in range(additional_connections):
        x = random.randrange(1, MAZE_WIDTH - 1, 2)
        y = random.randrange(1, MAZE_HEIGHT - 1, 2)
        if maze[y][x] == '.':
            direction = random.choice(DIRECTIONS)
            nx, ny = x + direction[0], y + direction[1]
            if 1 <= nx < MAZE_WIDTH - 1 and 1 <= ny < MAZE_HEIGHT - 1:
                if maze[ny][nx] == 'X':
                    maze[ny][nx] = '.'

    maze[1][1] = '.'
    maze[1][MAZE_WIDTH - 2] = '.'
    maze[MAZE_HEIGHT - 2][1] = '.'
    maze[MAZE_HEIGHT - 2][MAZE_WIDTH - 2] = '.'

    for _ in range(level * 5):
        while True:
            x = random.randint(1, MAZE_WIDTH - 2)
            y = random.randint(1, MAZE_HEIGHT - 2)
            if maze[y][x] == '.':
                maze[y][x] = 'o'
                break

    return maze

class Pacman:
    def __init__(self):
        self.x = GRID_WIDTH // 2
        self.y = GRID_HEIGHT // 2
        self.score = 0
        self.path = []  # Зберігаємо шлях до їжі
        self.counter = 1  # Лічильник для зменшення швидкості

    def draw(self):
        pygame.draw.circle(screen, YELLOW, (self.x * CELL_SIZE + CELL_SIZE // 2, self.y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 2)

    def move(self, maze, ghosts):
        self.counter += 1
        if self.counter % 20 == 0:  # Рух тільки на кожному 20-му кроці
            if not self.path or self.is_danger_near(ghosts):  # Якщо шляху немає або привид поруч, змінюємо поведінку
                self.path = self.find_safest_food_or_escape(maze, ghosts)

            if self.path:
                next_move = self.path.pop(0)
                self.x, self.y = next_move

                if maze[self.y][self.x] == 'o':
                    self.score += 1
                    maze[self.y][self.x] = '.'

    def is_danger_near(self, ghosts):
        for ghost in ghosts:
            distance = abs(self.x - ghost.x) + abs(self.y - ghost.y)
            if distance <= 2:  # Якщо привид на відстані до 2 клітинок
                return True
        return False

    def find_safest_food_or_escape(self, maze, ghosts):
        # Якщо є небезпека, спробуємо втекти
        if self.is_danger_near(ghosts):
            return self.find_escape_path(maze, ghosts)

        # Якщо немає небезпеки, шукаємо безпечну їжу
        return self.find_safest_food(maze, ghosts)

    def find_escape_path(self, maze, ghosts):
        escape_paths = []
        min_risk = float('inf')
        best_path = []

        for dx, dy in DIRECTIONS:
            next_x, next_y = self.x + dx, self.y + dy
            if 0 <= next_x < MAZE_WIDTH and 0 <= next_y < MAZE_HEIGHT and maze[next_y][next_x] in ['.', 'o']:
                path = self.bfs(maze, (self.x, self.y), (next_x, next_y))
                if path:
                    risk = self.calculate_risk(path, ghosts)
                    if risk < min_risk:
                        min_risk = risk
                        best_path = path

        return best_path

    def find_safest_food(self, maze, ghosts):
        food_positions = [(x, y) for y in range(MAZE_HEIGHT) for x in range(MAZE_WIDTH) if maze[y][x] == 'o']
        if not food_positions:
            return []

        min_risk = float('inf')
        best_path = []

        for food in food_positions:
            path = self.bfs(maze, (self.x, self.y), food)
            if path:
                risk = self.calculate_risk(path, ghosts)
                if risk < min_risk:
                    min_risk = risk
                    best_path = path

        return best_path

    def calculate_risk(self, path, ghosts):
        risk = 0
        for step in path:
            for ghost in ghosts:
                distance = abs(step[0] - ghost.x) + abs(step[1] - ghost.y)
                risk += max(0, 10 - distance)  # Чим ближче привид, тим більше ризик
        return risk

    def bfs(self, maze, start, goal):
        queue = deque([start])
        came_from = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dx, dy in DIRECTIONS:
                next_cell = (current[0] + dx, current[1] + dy)
                if 0 <= next_cell[0] < MAZE_WIDTH and 0 <= next_cell[1] < MAZE_HEIGHT and maze[next_cell[1]][next_cell[0]] in ['.', 'o'] and next_cell not in came_from:
                    queue.append(next_cell)
                    came_from[next_cell] = current

        return self.reconstruct_path(came_from, start, goal)

    def reconstruct_path(self, came_from, start, goal):
        path = []
        current = goal
        while current is not None and current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

class Ghost:
    def __init__(self, x, y, role, color, target=None):
        self.x = x
        self.y = y
        self.color = color
        self.role = role
        self.target = target
        self.path = []  # Шлях для руху
        self.counter = 1  # Лічильник для зменшення швидкості
        self.change_time = time.time()  # Час останньої зміни поведінки або цілі

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x * CELL_SIZE, self.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def move(self, maze, pacman):
        self.counter += 1
        current_time = time.time()

        if current_time - self.change_time > 20:  # Кожні 20 секунд змінюємо поведінку або ціль
            self.change_behavior(maze)
            self.change_time = current_time

        if self.counter % 20 == 0:  # Рух тільки на кожному 20-му кроці
            if not self.path or (self.x, self.y) == self.path[-1]:  # Якщо немає шляху або досягли кінця шляху
                if self.role == 'patrol':
                    self.patrol(maze)
                elif self.role == 'chase':
                    self.chase(maze, pacman)
                elif self.role == 'intercept':
                    self.intercept(maze, pacman)
                elif self.role == 'random':
                    self.random_move(maze)

            # Якщо є шлях, зробити крок на одну клітинку
            if self.path:
                next_move = self.path.pop(0)
                self.x, self.y = next_move

    def patrol(self, maze):
        if not self.path:
            self.path = self.dfs(maze, (self.x, self.y), self.target)
        if not self.path:  # Якщо шлях не знайдено, виконуємо випадковий рух
            self.random_move(maze)

    def chase(self, maze, pacman):
        # Використання A* для обчислення шляху до Пакмана
        self.path = self.a_star(maze, (self.x, self.y), (pacman.x, pacman.y))
        if not self.path:  # Якщо шлях не знайдено, виконуємо випадковий рух
            self.random_move(maze)

    def intercept(self, maze, pacman):
        future_x, future_y = self.predict_pacman_position(pacman)
        self.path = self.bfs(maze, (self.x, self.y), (future_x, future_y))
        if not self.path:  # Якщо шлях не знайдено, виконуємо випадковий рух
            self.random_move(maze)

    def predict_pacman_position(self, pacman):
        # Проста евристика для прогнозування руху Пакмана
        direction = random.choice(DIRECTIONS)
        return pacman.x + direction[0] * 3, pacman.y + direction[1] * 3

    def change_behavior(self, maze):
        if self.role == 'patrol':
            food_positions = [(x, y) for y in range(MAZE_HEIGHT) for x in range(MAZE_WIDTH) if maze[y][x] == 'o']
            if food_positions:
                self.target = random.choice(food_positions)
            self.path = []

        elif self.role == 'random':
            self.role = random.choice(['patrol', 'chase', 'intercept'])
            if self.role == 'patrol':
                self.change_behavior(maze)  # Задати нову ціль патрулювання

    def dfs(self, maze, start, goal):
        stack = [(start, [start])]
        visited = set()
        while stack:
            (current, path) = stack.pop()
            if current == goal:
                return path
            if current not in visited:
                visited.add(current)
                for dx, dy in DIRECTIONS:
                    next_cell = (current[0] + dx, current[1] + dy)
                    if 0 <= next_cell[0] < MAZE_WIDTH and 0 <= next_cell[1] < MAZE_HEIGHT and maze[next_cell[1]][next_cell[0]] == '.' and next_cell not in visited:
                        stack.append((next_cell, path + [next_cell]))
        return []

    def bfs(self, maze, start, goal):
        queue = deque([start])
        came_from = {start: None}
        while queue:
            current = queue.popleft()
            if current == goal:
                break
            for dx, dy in DIRECTIONS:
                next_cell = (current[0] + dx, current[1] + dy)
                if 0 <= next_cell[0] < MAZE_WIDTH and 0 <= next_cell[1] < MAZE_HEIGHT and maze[next_cell[1]][next_cell[0]] == '.' and next_cell not in came_from:
                    queue.append(next_cell)
                    came_from[next_cell] = current
        return self.reconstruct_path(came_from, start, goal)

    def a_star(self, maze, start, goal):
        heap = []
        heapq.heappush(heap, (0, start))
        came_from = {start: None}
        cost_so_far = {start: 0}
        while heap:
            _, current = heapq.heappop(heap)
            if current == goal:
                break
            for dx, dy in DIRECTIONS:
                next_cell = (current[0] + dx, current[1] + dy)
                if 0 <= next_cell[0] < MAZE_WIDTH and 0 <= next_cell[1] < MAZE_HEIGHT and maze[next_cell[1]][next_cell[0]] == '.':
                    new_cost = cost_so_far[current] + 1
                    if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                        cost_so_far[next_cell] = new_cost
                        priority = new_cost + self.heuristic(goal, next_cell)
                        heapq.heappush(heap, (priority, next_cell))
                        came_from[next_cell] = current
        return self.reconstruct_path(came_from, start, goal)

    def heuristic(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def reconstruct_path(self, came_from, start, goal):
        if goal not in came_from:
            # Якщо шлях не знайдено, повертаємо порожній список
            return []
        path = []
        current = goal
        while current is not None and current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path[1:]  # Повертаємо шлях без стартової точки

    def random_move(self, maze):
        while True:
            dx, dy = random.choice(DIRECTIONS)
            next_x = self.x + dx
            next_y = self.y + dy
            if 0 <= next_x < MAZE_WIDTH and 0 <= next_y < MAZE_HEIGHT and maze[next_y][next_x] == '.':
                self.path = [(next_x, next_y)]  # Рухаємося на одну клітинку
                break

# Головний цикл гри
def main():
    clock = pygame.time.Clock()
    pacman = Pacman()
    patrol_point = (random.randint(1, MAZE_WIDTH - 2), random.randint(1, MAZE_HEIGHT - 2))
    ghosts = [
        Ghost(x=1, y=1, role='patrol', color=RED, target=patrol_point),
        Ghost(x=MAZE_WIDTH - 2, y=1, color=ORANGE, role='chase'),
        Ghost(x=1, y=MAZE_HEIGHT - 2, color=GREEN, role='intercept'),
        Ghost(x=MAZE_WIDTH - 2, y=MAZE_HEIGHT - 2, color=PURPLE, role='random')
    ]
    level = 1
    maze = generate_maze(level)
    running = True
    game_over = False
    game_won = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    level = min(level + 1, 10)
                    maze = generate_maze(level)  # Оновлення лабіринту
                    for ghost in ghosts:  # Оновлення шляхів для привидів
                        ghost.path = []
                    pacman.path = []  # Оновлення шляху для Пакмана
                elif event.key == pygame.K_MINUS:
                    level = max(level - 1, 1)
                    maze = generate_maze(level)  # Оновлення лабіринту
                    for ghost in ghosts:  # Оновлення шляхів для привидів
                        ghost.path = []
                    pacman.path = []  # Оновлення шляху для Пакмана

        pacman.move(maze, ghosts)  # Передача нового лабіринту у метод руху

        for ghost in ghosts:
            ghost.move(maze, pacman)  # Передача нового лабіринту у метод руху
            if ghost.x == pacman.x and ghost.y == pacman.y:  # Перевірка на зіткнення з привидом
                game_over = True

        screen.fill(BLACK)

        for y in range(MAZE_HEIGHT):
            for x in range(MAZE_WIDTH):
                if maze[y][x] == 'X':
                    pygame.draw.rect(screen, BLUE, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                elif maze[y][x] == 'o':
                    pygame.draw.circle(screen, WHITE, (x * CELL_SIZE + CELL_SIZE // 2, y * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 4)
          
        if not game_over:
            pacman.draw()
            for ghost in ghosts:
                ghost.draw()
        else:
            font = pygame.font.Font(None, 72)
            if game_won:
                game_over_text = font.render("You Won!", True, GREEN)
            else:
                game_over_text = font.render("Game Over", True, RED)
            screen.blit(game_over_text, (WIDTH // 2 - game_over_text.get_width() // 2, HEIGHT // 2 - game_over_text.get_height() // 2))

        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {pacman.score}  Level: {level}  Press +/- to change level", True, WHITE)
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
