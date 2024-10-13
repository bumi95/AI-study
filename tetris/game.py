import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 450
SCREEN_HEIGHT = 900
GRID_SIZE = 30

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Shapes
SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
    [[1, 1, 1], [0, 0, 1]],
    [[1, 1, 1], [1, 0, 0]]
]

# Shape colors
SHAPE_COLORS = [CYAN, YELLOW, PURPLE, GREEN, RED, BLUE, ORANGE]

class Tetris:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tetris")
        self.clock = pygame.time.Clock()
        self.grid = [[BLACK for _ in range(SCREEN_WIDTH // GRID_SIZE)] for _ in range(SCREEN_HEIGHT // GRID_SIZE)]
        self.current_shape = self.get_new_shape()
        self.next_shape = self.get_new_shape()
        self.shape_x = SCREEN_WIDTH // GRID_SIZE // 2 - len(self.current_shape[0]) // 2
        self.shape_y = 0
        self.game_over = False
        
        # Add variables to control the game speed
        self.drop_speed = 50  # Speed at which the shape drops

    def get_new_shape(self):
        shape = random.choice(SHAPES)
        color = SHAPE_COLORS[SHAPES.index(shape)]
        return [[color if cell else BLACK for cell in row] for row in shape]

    def draw_grid(self):
        for y in range(len(self.grid)):
            for x in range(len(self.grid[y])):
                pygame.draw.rect(self.screen, self.grid[y][x], (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)
                pygame.draw.rect(self.screen, WHITE, (x * GRID_SIZE, y * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)

    def draw_shape(self, shape, offset_x, offset_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell != BLACK:
                    pygame.draw.rect(self.screen, cell, ((offset_x + x) * GRID_SIZE, (offset_y + y) * GRID_SIZE, GRID_SIZE, GRID_SIZE), 0)
                    pygame.draw.rect(self.screen, WHITE, ((offset_x + x) * GRID_SIZE, (offset_y + y) * GRID_SIZE, GRID_SIZE, GRID_SIZE), 1)

    def rotate_shape(self):
        self.current_shape = [list(row) for row in zip(*self.current_shape[::-1])]

    def valid_move(self, shape, offset_x, offset_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell != BLACK:
                    new_x = offset_x + x
                    new_y = offset_y + y
                    if new_x < 0 or new_x >= SCREEN_WIDTH // GRID_SIZE or new_y >= SCREEN_HEIGHT // GRID_SIZE or self.grid[new_y][new_x] != BLACK:
                        return False
        return True

    def merge_shape(self):
        for y, row in enumerate(self.current_shape):
            for x, cell in enumerate(row):
                if cell != BLACK:
                    self.grid[self.shape_y + y][self.shape_x + x] = cell

    def clear_lines(self):
        new_grid = [row for row in self.grid if BLACK in row]
        lines_cleared = len(self.grid) - len(new_grid)
        self.grid = [[BLACK for _ in range(SCREEN_WIDTH // GRID_SIZE)] for _ in range(lines_cleared)] + new_grid

    def run(self):
        drop_counter = 0

        while not self.game_over:
            self.screen.fill(BLACK)
            self.draw_grid()
            self.draw_shape(self.current_shape, self.shape_x, self.shape_y)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_over = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT and self.valid_move(self.current_shape, self.shape_x - 1, self.shape_y):
                        self.shape_x -= 1
                    elif event.key == pygame.K_RIGHT and self.valid_move(self.current_shape, self.shape_x + 1, self.shape_y):
                        self.shape_x += 1
                    elif event.key == pygame.K_DOWN and self.valid_move(self.current_shape, self.shape_x, self.shape_y + 1):
                        self.shape_y += 1
                    elif event.key == pygame.K_UP:
                        rotated_shape = [list(row) for row in zip(*self.current_shape[::-1])]
                        if self.valid_move(rotated_shape, self.shape_x, self.shape_y):
                            self.current_shape = rotated_shape
                    elif event.key == pygame.K_SPACE:
                        while self.valid_move(self.current_shape, self.shape_x, self.shape_y + 1):
                            self.shape_y += 1
         
            drop_counter += 1

            if drop_counter >= self.drop_speed:
                if not self.valid_move(self.current_shape, self.shape_x, self.shape_y + 1):
                    self.merge_shape()
                    self.clear_lines()
                    self.current_shape = self.next_shape
                    self.next_shape = self.get_new_shape()
                    self.shape_x = SCREEN_WIDTH // GRID_SIZE // 2 - len(self.current_shape[0]) // 2
                    self.shape_y = 0
                    if not self.valid_move(self.current_shape, self.shape_x, self.shape_y):
                        self.game_over = True
                else:
                    self.shape_y += 1
                drop_counter = 0

        pygame.quit()

if __name__ == "__main__":
    Tetris().run()