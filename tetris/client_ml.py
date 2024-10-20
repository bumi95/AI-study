import pygame
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from client import TetrisGame, GameConfig
import logging

logging.basicConfig(level=logging.DEBUG)

class TetrisGameML(TetrisGame):
    """강화학습을 위한 테트리스 게임 환경"""

    def __init__(self, screen_width=800, screen_height=600):
        super().__init__(screen_width, screen_height)
        self.websocket = None  # 웹소켓 연결을 None으로 설정
        self.action_space = [
            'left', 'right', 'rotate', 'down', 'drop'
        ]

    async def send_lines_to_others(self, lines_cleared):
        # 학습 환경에서는 이 메서드를 무시합니다.
        pass

    async def clear_lines(self):
        lines_cleared = 0
        y = GameConfig.GRID_HEIGHT - 1
        while y >= 0:
            if all(self.grid[y]):
                lines_cleared += 1
                del self.grid[y]
                self.grid.insert(0, [None] * GameConfig.GRID_WIDTH)
            else:
                y -= 1
        
        if lines_cleared > 0:
            self.score += [0, 40, 100, 300, 1200][min(lines_cleared, 4)]
            # await self.send_lines_to_others(lines_cleared)  # 이 줄을 주석 처리 또는 제거
        
        return lines_cleared

    async def lock_piece(self):
        for y, row in enumerate(self.current_piece['shape']):
            for x, cell in enumerate(row):
                if cell:
                    self.grid[y + self.current_piece['y']][x + self.current_piece['x']] = self.current_piece['color']
        
        await self.clear_lines()
        if not self.spawn_piece():
            self.game_over = True

    def get_state(self) -> np.ndarray:
        """
        현재 게임 상태를 관찰 가능한 형태로 반환
        
        :return: 1차원 numpy 배열로 표현된 게임 상태
        """
        
        # grid를 numpy 배열로 변환하고 형태를 확인
        grid_array = np.zeros((GameConfig.GRID_HEIGHT, GameConfig.GRID_WIDTH), dtype=np.float32)
        
        for y, row in enumerate(self.grid):
            for x, cell in enumerate(row):
                if isinstance(cell, (int, float)):
                    grid_array[y, x] = cell
                elif cell:  # cell이 True나 문자열 등 다른 값일 경우
                    grid_array[y, x] = 1
        
        current_piece = np.zeros((4, GameConfig.GRID_WIDTH), dtype=np.float32)
        
        piece_height = len(self.current_piece['shape'])
        piece_width = len(self.current_piece['shape'][0])
        
        for y in range(piece_height):
            for x in range(piece_width):
                if self.current_piece['shape'][y][x]:
                    grid_x = self.current_piece['x'] + x
                    grid_y = self.current_piece['y'] + y
                    if 0 <= grid_x < GameConfig.GRID_WIDTH and 0 <= grid_y < 4:
                        current_piece[grid_y, grid_x] = 1

        state = np.vstack((grid_array, current_piece))
        return state.flatten()

    def get_reward(self) -> float:
        """
        현재 상태에 대한 보상 계산
        
        :return: 현재 상태에 대한 보상 값
        """
        reward = 0

        # 줄을 지울 때마다 보상
        lines_cleared = sum(1 for row in self.grid if all(row))
        reward += lines_cleared * 10

        # 현재 피스가 바닥에 닿았을 때 보상
        if not self.is_valid_position(self.current_piece['shape'], self.current_piece['x'], self.current_piece['y'] + 1):
            reward += 1

        # 게임 오버 시 큰 패널티
        if self.game_over:
            reward -= 100

        # 추가 보상: 현재 피스가 얼마나 높은 위치에 있는지 (낮을수록 좋음)
        reward -= self.current_piece['y'] * 0.1

        # 추가 보상: 현재 그리드의 높이 (낮을수록 좋음)
        max_height = max((y for y, row in enumerate(self.grid) if any(row)), default=0)
        reward -= max_height * 0.1

        # 추가 보상: 현재 그리드의 구멍 수 (적을수록 좋음)
        holes = sum(1 for x in range(GameConfig.GRID_WIDTH) for y in range(1, GameConfig.GRID_HEIGHT) if self.grid[y][x] is None and self.grid[y-1][x] is not None)
        reward -= holes * 0.5

        return reward

    def reset(self) -> np.ndarray:
        """
        게임 상태 초기화 및 초기 상태 반환
        
        :return: 초기화된 게임 상태
        """
        super().__init__(self.screen.get_width(), self.screen.get_height())
        logging.debug("Game reset")  # 로깅 추가
        return self.get_state()

    async def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        에이전트의 행동을 수행하고 결과 반환
        
        :param action: 수행할 행동
        :return: (새로운 상태, 보상, 게임 종료 여부, 추가 정보)
        """
        logging.debug(f"Performing action: {action}")  # 로깅 추가

        if action == 'left':
            await self.move(-1, 0)
        elif action == 'right':
            await self.move(1, 0)
        elif action == 'rotate':
            self.rotate()
        elif action == 'down':
            await self.move(0, 1)
        elif action == 'drop':
            await self.hard_drop()
        else:
            logging.warning(f"Unknown action: {action}")  # 알 수 없는 행동에 대한 경고

        await self.update_game_state()
        
        new_state = self.get_state()
        reward = self.get_reward()
        done = not self.running
        info = {"score": self.score}

        logging.debug(f"Step result - Reward: {reward}, Done: {done}, Score: {self.score}")  # 로깅 추가

        return new_state, reward, done, info

    async def run_episode(self, agent):
        """
        강화학습 에이전트를 사용하여 한 에피소드 실행
        
        :param agent: 강화학습 에이전트
        :return: 에피소드에서 얻은 총 보상
        """
        state = self.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = await self.step(action)
            agent.learn(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward

        return total_reward
