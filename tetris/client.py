import pygame
import asyncio
import websockets
import json
import random
import logging
import time
import sys
from typing import List, Dict, Any, Tuple

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class GameConfig:
    """게임 설정 및 상수"""
    GRID_WIDTH = 10
    GRID_HEIGHT = 20
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    
    @classmethod
    def set_screen_size(cls, width: int, height: int):
        cls.SCREEN_WIDTH = width
        cls.SCREEN_HEIGHT = height
    
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    GRAY = (128, 128, 128)
    COLORS = [
        (0, 255, 255), (255, 255, 0), (128, 0, 128),
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 165, 0),
    ]
    
    SHAPES = [
        [[1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[0, 1, 0], [1, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 0, 0], [1, 1, 1]],
        [[0, 0, 1], [1, 1, 1]],
    ]

    # 서버 설정 추가
    SERVER_HOST = "localhost"
    SERVER_PORT = 8765

class TetrisGame:
    """테트리스 게임 클래스"""

    def __init__(self, screen_width=800, screen_height=600):
        """게임 초기화"""
        pygame.init()
        GameConfig.set_screen_size(screen_width, screen_height)
        self.screen = pygame.display.set_mode((GameConfig.SCREEN_WIDTH, GameConfig.SCREEN_HEIGHT))
        pygame.display.set_caption("Multiplayer Tetris")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        self.grid: List[List[int]] = [[0 for _ in range(GameConfig.GRID_WIDTH)] for _ in range(GameConfig.GRID_HEIGHT)]
        self.current_piece: Dict[str, Any] = self.new_piece()
        self.score = 0
        self.game_speed = 1.0
        self.drop_speed = 1.0
        self.drop_interval = 1.0 / self.drop_speed
        
        self.websocket = None
        self.client_id = None
        self.other_players: Dict[str, Dict[str, Any]] = {}
        self.running = True
        self.reconnect_attempts = 0
        
        self.start_time = time.time()
        self.last_speed_increase = self.start_time
        self.last_drop_time = time.time()
        self.game_over = False
        
        self.calculate_sizes()
        

    def calculate_sizes(self) -> None:
        """화면 크기에 따른 게임 요소 크기 계산"""
        main_board_max_width = int(GameConfig.SCREEN_WIDTH * 0.4)
        main_board_max_height = int(GameConfig.SCREEN_HEIGHT * 0.8)
        
        self.main_block_size = min(
            main_board_max_width // GameConfig.GRID_WIDTH,
            main_board_max_height // GameConfig.GRID_HEIGHT
        )
        self.mini_block_size = max(self.main_block_size // 3, 1)
        
        self.main_board_width = GameConfig.GRID_WIDTH * self.main_block_size
        self.main_board_height = GameConfig.GRID_HEIGHT * self.main_block_size
        self.main_board_left = (GameConfig.SCREEN_WIDTH - self.main_board_width) // 2
        self.main_board_top = (GameConfig.SCREEN_HEIGHT - self.main_board_height) // 2

        font_size = int(max(self.main_block_size * 0.8, 12))
        self.font = pygame.font.Font(None, font_size)

    def new_piece(self) -> Dict[str, Any]:
        """새로운 테트리스 조각 생성"""
        shape = random.choice(GameConfig.SHAPES)
        return {
            'shape': shape,
            'color': random.choice(GameConfig.COLORS),
            'x': GameConfig.GRID_WIDTH // 2 - len(shape[0]) // 2,
            'y': 0
        }

    def draw_board(self, grid: List[List[int]], left: int, top: int, block_size: int) -> None:
        """게임 보드 그리기"""
        for y, row in enumerate(grid):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, cell,
                                     (left + x * block_size,
                                      top + y * block_size,
                                      block_size - 1, block_size - 1))

    def draw_piece(self, piece: Dict[str, Any], left: int, top: int, block_size: int) -> None:
        """테트리스 조각 그리기"""
        for y, row in enumerate(piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    pygame.draw.rect(self.screen, piece['color'],
                                     (left + (piece['x'] + x) * block_size,
                                      top + (piece['y'] + y) * block_size,
                                      block_size - 1, block_size - 1))

    def draw_game(self) -> None:
        """전체 게임 화면 그리기"""
        self.screen.fill(GameConfig.BLACK)

        # 메인 플레이어 보드 그리기
        pygame.draw.rect(self.screen, GameConfig.WHITE, (self.main_board_left - 2, self.main_board_top - 2,
                                            self.main_board_width + 4, self.main_board_height + 4), 2)
        self.draw_board(self.grid, self.main_board_left, self.main_board_top, self.main_block_size)
        self.draw_piece(self.current_piece, self.main_board_left, self.main_board_top, self.main_block_size)

        # 게임 정보 표시 (메인 보드 오른쪽)
        info_left = self.main_board_left + self.main_board_width + 20
        info_top = self.main_board_top
        info_spacing = int(GameConfig.SCREEN_HEIGHT * 0.05)
        
        score_text = self.font.render(f"Score: {self.score}", True, GameConfig.WHITE)
        self.screen.blit(score_text, (info_left, info_top))
        
        speed_text = self.font.render(f"Speed: {self.game_speed:.1f}", True, GameConfig.WHITE)
        self.screen.blit(speed_text, (info_left, info_top + info_spacing))
        
        drop_speed_text = self.font.render(f"Drop Speed: {self.drop_speed:.2f}", True, GameConfig.WHITE)
        self.screen.blit(drop_speed_text, (info_left, info_top + 2 * info_spacing))

        elapsed_time = int(time.time() - self.start_time)
        time_text = self.font.render(f"Time: {elapsed_time // 60:02d}:{elapsed_time % 60:02d}", True, GameConfig.WHITE)
        self.screen.blit(time_text, (info_left, info_top + 3 * info_spacing))

        # 다른 플레이어 보드 그리기
        mini_board_width = GameConfig.GRID_WIDTH * self.mini_block_size
        mini_board_height = GameConfig.GRID_HEIGHT * self.mini_block_size
        margin = int(GameConfig.SCREEN_WIDTH * 0.02)
        top_margin = int(GameConfig.SCREEN_HEIGHT * 0.05)  # 상단 여백 추가

        # 다른 플레이어 보드 위치 계산
        other_board_positions = [
            (GameConfig.SCREEN_WIDTH - mini_board_width - margin, top_margin),  # 오른쪽 상단
            (GameConfig.SCREEN_WIDTH - mini_board_width - margin, GameConfig.SCREEN_HEIGHT - mini_board_height - margin),  # 오른쪽 하단
            (margin, top_margin),  # 왼쪽 상단
            (margin, GameConfig.SCREEN_HEIGHT - mini_board_height - margin)  # 왼쪽 하단
        ]

        for i, (player_id, state) in enumerate(list(self.other_players.items())[:4]):
            left, top = other_board_positions[i]
            pygame.draw.rect(self.screen, GameConfig.WHITE, (left - 2, top - 2,
                                                mini_board_width + 4,
                                                mini_board_height + 4), 1)
            self.draw_board(state['board'], left, top, self.mini_block_size)
            player_text = self.font.render(f"P{i+2} Score: {state['score']}", True, GameConfig.WHITE)
            self.screen.blit(player_text, (left, top - 30))

        pygame.display.flip()

    def draw_game_info(self) -> None:
        """게임 정보 (점수, 속도, 시간) 표시"""
        score_text = self.font.render(f"Score: {self.score}", True, GameConfig.WHITE)
        self.screen.blit(score_text, (10, 10))

        speed_text = self.font.render(f"Speed: {self.game_speed:.1f}", True, GameConfig.WHITE)
        self.screen.blit(speed_text, (10, 40))

        elapsed_time = int(time.time() - self.start_time)
        time_text = self.font.render(f"Time: {elapsed_time // 60:02d}:{elapsed_time % 60:02d}", True, GameConfig.WHITE)
        self.screen.blit(time_text, (10, 70))

        drop_speed_text = self.font.render(f"Drop Speed: {self.drop_speed:.2f}", True, GameConfig.WHITE)
        self.screen.blit(drop_speed_text, (10, 100))

    def draw_other_players(self) -> None:
        """다른 플레이어의 게임 보드 그리기"""
        mini_board_width = GameConfig.GRID_WIDTH * self.mini_block_size
        mini_board_height = GameConfig.GRID_HEIGHT * self.mini_block_size
        margin = 20

        other_board_positions = [
            (margin, margin),
            (GameConfig.SCREEN_WIDTH - mini_board_width - margin, margin),
            (margin, GameConfig.SCREEN_HEIGHT - mini_board_height - margin),
            (GameConfig.SCREEN_WIDTH - mini_board_width - margin, GameConfig.SCREEN_HEIGHT - mini_board_height - margin)
        ]

        for i, (player_id, state) in enumerate(list(self.other_players.items())[:4]):
            left, top = other_board_positions[i]
            pygame.draw.rect(self.screen, GameConfig.WHITE, (left - 2, top - 2,
                                                mini_board_width + 4,
                                                mini_board_height + 4), 1)
            self.draw_board(state['board'], left, top, self.mini_block_size)
            player_text = self.font.render(f"P{i+2}: {state['score']}", True, GameConfig.WHITE)
            self.screen.blit(player_text, (left, top - 30))

    async def move(self, dx: int, dy: int) -> bool:
        """테트리스 조각 이동"""
        new_x = self.current_piece['x'] + dx
        new_y = self.current_piece['y'] + dy
        if self.is_valid_position(self.current_piece['shape'], new_x, new_y):
            self.current_piece['x'] = new_x
            self.current_piece['y'] = new_y
            return True
        elif dy > 0:  # 아래로 이동 시 충돌한 경우
            await self.lock_piece()
            return False
        return False

    def rotate(self) -> None:
        """테트리스 조각 회전"""
        rotated_shape = list(zip(*reversed(self.current_piece['shape'])))
        if self.is_valid_position(rotated_shape, self.current_piece['x'], self.current_piece['y']):
            self.current_piece['shape'] = rotated_shape

    async def hard_drop(self) -> None:
        """테트리스 조각 즉시 떨어뜨리기"""
        while await self.move(0, 1):
            pass

    def is_valid_position(self, shape: List[List[int]], x: int, y: int) -> bool:
        """테트리스 조각의 위치가 유효한지 확인"""
        for row in range(len(shape)):
            for col in range(len(shape[row])):
                if shape[row][col]:
                    if (y + row >= GameConfig.GRID_HEIGHT or
                        x + col < 0 or x + col >= GameConfig.GRID_WIDTH or
                        self.grid[y + row][x + col]):
                        return False
        return True

    async def lock_piece(self) -> None:
        """테트리스 조각을 그리드에 고정"""
        for y, row in enumerate(self.current_piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[self.current_piece['y'] + y][self.current_piece['x'] + x] = self.current_piece['color']
        await self.clear_lines()
        self.current_piece = self.new_piece()
        if not self.is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
            self.running = False
            #logging.info("게임 오버")

    async def clear_lines(self) -> None:
        """완성된 라인 제거 및 점수 계산"""
        lines_cleared = 0
        y = GameConfig.GRID_HEIGHT - 1
        while y >= 0:
            if all(self.grid[y]):
                lines_cleared += 1
                del self.grid[y]
                self.grid.insert(0, [0 for _ in range(GameConfig.GRID_WIDTH)])
            else:
                y -= 1

        if lines_cleared > 0:
            self.score += [0, 40, 100, 300, 1200][min(lines_cleared, 4)]
            logging.info(f"{lines_cleared}줄 제거, 현재 점수: {self.score}")
            await self.send_lines_to_others(lines_cleared)

    async def send_lines_to_others(self, lines_cleared):
        if self.websocket is None:
            return  # 웹소켓 연결이 없으면 메서드를 종료합니다.
        message = {
            "type": "send_lines",
            "num_lines": lines_cleared
        }
        await self.websocket.send(json.dumps(message))

    def insert_lines(self, num_lines: int) -> None:
        """하단에 라인 삽입"""
        new_lines = []
        for _ in range(num_lines):
            new_line = [random.choice(GameConfig.COLORS) for _ in range(GameConfig.GRID_WIDTH)]
            gap = random.randint(0, GameConfig.GRID_WIDTH - 1)
            new_line[gap] = 0  # 각 라인에 하나의 빈 칸 생성
            new_lines.append(new_line)
        
        # 기존 그리드의 상단 부분을 잘라내고 새 라인을 하단에 추가
        self.grid = self.grid[num_lines:] + new_lines
        
        # 현재 조각의 위치를 위로 이동
        self.current_piece['y'] -= num_lines
        
        # 현재 조각이 그리드를 벗어났는지 확인
        if self.current_piece['y'] < 0 or not self.is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
            # 조각이 화면 위로 밀려났거나 유효하지 않은 위치에 있다면 게임 오버
            self.running = False
            logging.info("게임 오버: 라인 추가로 인한 오버플로우")
        
        logging.info(f"{num_lines}줄이 하단에 추가됨")

    def set_game_speed(self, speed: float) -> None:
        """게임 속도 설정"""
        self.game_speed = max(0.1, min(speed, 3.0))
        self.drop_speed = self.game_speed
        self.drop_interval = 1.0 / self.drop_speed
        logging.info(f"게임 속도 변경: {self.game_speed:.1f}")

    def update_game_speed(self) -> None:
        """시간에 따른 게임 속도 업데이트"""
        current_time = time.time()
        if current_time - self.last_speed_increase > 60:  # 1분마다 속도 증가
            self.set_game_speed(self.game_speed + 0.1)
            self.last_speed_increase = current_time

    async def game_loop(self) -> None:
        """메인 게임 루프"""
        if not await self.connect_to_server():
            return

        asyncio.create_task(self.receive_updates())

        try:
            while self.running:
                await self.handle_events()
                await self.update_game_state()
                self.draw_game()
                self.clock.tick(60)

                if self.websocket and self.websocket.open:
                    await self.send_state()
                else:
                    await self.reconnect()
                    if not self.running:
                        break                

                await asyncio.sleep(0.01)
        except Exception as e:
            logging.error(f"게임 루프 중 예외 발생: {e}")
        finally:
            self.cleanup()

    async def handle_events(self) -> None:
        """pygame 이벤트 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                logging.info("게임 종료")
                self.running = False
            elif event.type == pygame.KEYDOWN:
                await self.handle_keydown(event.key)

    async def handle_keydown(self, key: int) -> None:
        """키 입력 처리"""
        if key == pygame.K_LEFT:
            await self.move(-1, 0)
        elif key == pygame.K_RIGHT:
            await self.move(1, 0)
        elif key == pygame.K_DOWN:
            await self.move(0, 1)
        elif key == pygame.K_UP:
            self.rotate()
        elif key == pygame.K_SPACE:
            await self.hard_drop()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.set_game_speed(self.game_speed + 0.1)
        elif key == pygame.K_MINUS:
            self.set_game_speed(self.game_speed - 0.1)

    async def update_game_state(self) -> None:
        """게임 상태 업데이트"""
        self.update_game_speed()
        current_time = time.time()
        if current_time - self.last_drop_time > self.drop_interval:
            await self.move(0, 1)
            self.last_drop_time = current_time

    def cleanup(self) -> None:
        """게임 종료 시 정리 작업"""
        for task in asyncio.all_tasks():
            if task.get_name() == 'receive_updates':
                task.cancel()
                break
        if self.websocket:
            asyncio.create_task(self.websocket.close())
        pygame.quit()
        logging.info("게임 루프 종료")

    async def connect_to_server(self) -> bool:
        """서버에 연결"""
        try:
            self.websocket = await websockets.connect(f"ws://{GameConfig.SERVER_HOST}:{GameConfig.SERVER_PORT}")
            response = await self.websocket.recv()
            data = json.loads(response)
            if data["type"] == "connection_success":
                self.client_id = data["player_id"]
                logging.info(f"서버에 연결 성공. 클라이언트 ID: {self.client_id}")
                return True
            else:
                logging.error("서버 연결 실패")
                return False
        except Exception as e:
            logging.error(f"서버 연결 중 오류 발생: {e}")
            return False

    async def reconnect(self) -> None:
        """서버에 재연결 시도"""
        self.reconnect_attempts += 1
        if self.reconnect_attempts > 5:
            logging.error("최대 재연결 시도 횟수 초과")
            self.running = False
            return
        
        logging.info(f"서버에 재연결 시도 중... (시도 {self.reconnect_attempts})")
        if await self.connect_to_server():
            self.reconnect_attempts = 0
        else:
            await asyncio.sleep(5)

    async def send_state(self) -> None:
        """현재 게임 상태를 서버로 전송"""
        state = {
            "type": "update_state",
            "state": {
                "score": self.score,
                "board": self.grid
            }
        }
        await self.websocket.send(json.dumps(state))

    async def receive_updates(self) -> None:
        """서버로부터 업데이트 수신"""
        try:
            async for message in self.websocket:
                data = json.loads(message)
                if data["type"] == "update":
                    self.other_players = {player_id: state for player_id, state in data["game_states"].items() if player_id != self.client_id}
                elif data["type"] == "insert_lines":
                    self.insert_lines(data["num_lines"])
                    logging.info(f"{data['num_lines']}줄 추가됨")
        except websockets.exceptions.ConnectionClosed:
            logging.warning("서버와의 연결이 끊어짐")
        except Exception as e:
            logging.error(f"업데이트 수신 중 오류 발생: {e}")

    def spawn_piece(self):
        """새로운 테트리스 조각을 생성하고 게임 영역 상단에 배치"""
        self.current_piece = self.new_piece()
        self.current_piece['x'] = GameConfig.GRID_WIDTH // 2 - len(self.current_piece['shape'][0]) // 2
        self.current_piece['y'] = 0

        if not self.is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y']):
            self.running = False
            #logging.info("게임 오버: 새 조각을 놓을 공간이 없음")
            return False
        return True

if __name__ == "__main__":
    try:
        logging.info("테트리스 게임 시작")
        game = TetrisGame(1024, 768)  # 원하는 화면 크기 지정
        asyncio.run(game.game_loop())
    except Exception as e:
        logging.error(f"게임 실행 중 예외 발생: {e}", exc_info=True)
        sys.exit(1)