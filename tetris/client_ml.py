import pygame
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # 상위 폴더를 파이썬 경로에 추가
from game import TetrisGame, GameConfig
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)

class TetrisGameML(TetrisGame):
    """강화학습을 위한 테트리스 게임 환경"""

    def __init__(self, screen_width=800, screen_height=600, multiplayer=False):
        super().__init__(screen_width, screen_height)
        self.multiplayer = multiplayer
        self.websocket = None if not self.multiplayer else self.websocket
        self.action_space = [
            'left', 'right', 'rotate', 'down', 'drop'#, 'hold'
        ]  # 에이전트가 선택할 수 있는 행동들
        self.lines_cleared = 0  # 라인 제거 수를 추적하기 위한 변수 추가
        self.last_state = None
        self.episode_steps = 0

    async def send_lines_to_others(self, lines_cleared):
        if self.multiplayer:
            await super().send_lines_to_others(lines_cleared)

    async def clear_lines(self) -> int:
        """완성된 라인을 제거하고 점수를 계산합니다."""
        lines_cleared = await super().clear_lines()
        self.lines_cleared += lines_cleared
        return lines_cleared

    def get_state(self) -> np.ndarray:
        """
        현재 게임 상태를 관찰 가능한 형태로 반환합니다.
        
        :return: 1차원 numpy 배열로 표현된 게임 상태
        """
        
        # grid를 2D numpy 배열로 변환 (수정된 부분)
        grid_array = np.zeros((GameConfig.GRID_HEIGHT, GameConfig.GRID_WIDTH), dtype=np.float32)
        for y in range(GameConfig.GRID_HEIGHT):
            for x in range(GameConfig.GRID_WIDTH):
                if self.grid[y][x]:  # None이 아닌 경우에만 1로 설정
                    grid_array[y][x] = 1.0
        
        # 현재 조각의 상태를 표현
        current_piece_array = np.zeros_like(grid_array)
        if self.current_piece:
            shape = self.current_piece['shape']
            x = self.current_piece['x']
            y = self.current_piece['y']
            
            for i, row in enumerate(shape):
                for j, cell in enumerate(row):
                    if cell and 0 <= y+i < GameConfig.GRID_HEIGHT and 0 <= x+j < GameConfig.GRID_WIDTH:
                        current_piece_array[y+i][x+j] = 1.0
        
        # 게임 보드와 현재 조각 상태를 합침
        combined_grid = grid_array + current_piece_array
        
        # 다음 조각 정보를 원-핫 인코딩으로 표현
        next_piece_array = np.zeros(len(GameConfig.SHAPES), dtype=np.float32)
        if self.next_piece:
            next_piece_index = GameConfig.SHAPES.index(self.next_piece['shape'])
            next_piece_array[next_piece_index] = 1.0
        
        # 최종 상태 벡터 생성
        state = np.concatenate([combined_grid.flatten(), next_piece_array])
        return state

    def get_reward(self) -> float:
        """
        현재 상태에 대한 보상을 계산합니다.
        
        :return: 현재 상태에 대한 보상 값
        """
        
        reward = 0
        
        if self.lines_cleared > 0:
            reward += (self.lines_cleared ** 2) * 100
            self.lines_cleared = 0
        
        if not self.running:
            reward -= 1000
        
        if self.block_merged:
            height = [GameConfig.GRID_HEIGHT - next((y for y in range(GameConfig.GRID_HEIGHT) if self.grid[y][x]), GameConfig.GRID_HEIGHT) for x in range(GameConfig.GRID_WIDTH)]
            max_height = max(height)
            #holes = sum(1 for x in range(GameConfig.GRID_WIDTH) for y in range(GameConfig.GRID_HEIGHT - 1) if self.grid[y][x] == 0 and self.grid[y+1][x] == 1)
            holes = sum(1 for y in range(GameConfig.GRID_HEIGHT - max_height - 1, GameConfig.GRID_HEIGHT) for x in range(GameConfig.GRID_WIDTH) if self.grid[y][x] == 0)
            #bumpiness = sum(abs(height[i] - height[i+1]) for i in range(len(height) - 1))
            
            #reward -= (max_height * 0.5)
            reward -= (holes * 1.0)
            #reward -= (bumpiness * 0.5)
            self.block_merged = False
        
        return reward
    
    def reset(self) -> np.ndarray:
        """
        게임 상태를 초기화하고 초기 상태를 반환합니다.
        
        :return: 초기화된 게임 상태
        """
        super().__init__(self.screen.get_width(), self.screen.get_height())
        self.lines_cleared = 0
        self.episode_steps = 0
        self.score = 0
        self.last_state = self.get_state()
        #logging.info("Game reset")
        return self.last_state

    async def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        에이전트의 행동을 수행하고 결과를 반환합니다.
        
        :param action: 수행할 행동
        :return: (새로운 상태, 보상, 게임 종료 여부, 추가 정보)
        """
        self.episode_steps += 1
        logging.debug(f"Performing action: {action}")

        # 선택된 행동 수행
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
        #elif action == 'hold':
        #    self.hold()
        else:
            logging.warning(f"Unknown action: {action}")

        await self.update_game_state()
        
        new_state = self.get_state()
        reward = self.get_reward()
        done = not self.running #or self.episode_steps >= 1000  # 최대 스텝 수 제한
        info = {
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "max_height": max(GameConfig.GRID_HEIGHT - next((y for y in range(GameConfig.GRID_HEIGHT) if self.grid[y][x]), GameConfig.GRID_HEIGHT) for x in range(GameConfig.GRID_WIDTH))
        }

        logging.debug(f"Step result - Reward: {reward}, Done: {done}, Score: {self.score}, Lines cleared: {self.lines_cleared}")

        self.last_state = new_state
        return new_state, reward, done, info

    async def run_episode(self, agent):
        """
        강화학습 에이전트를 사용하여 한 에피소드를 실행합니다.
        
        :param agent: 강화학습 에이전트
        :return: 에피소드에서 얻은 총 보상
        """
        state = self.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, info = await self.step(action)
            agent.learn(state, action, reward, new_state, done)
            state = new_state
            total_reward += reward
            steps += 1

            if steps % 100 == 0:
                logging.info(f"Episode progress - Steps: {steps}, Total Reward: {total_reward}, Score: {info['score']}")

        logging.info(f"Episode finished - Total Steps: {steps}, Total Reward: {total_reward}, Final Score: {info['score']}")
        return total_reward, steps, info['score']

    async def connect_to_server(self):
        if self.multiplayer:
            return await super().connect_to_server()
        return True

    async def send_state(self):
        if self.multiplayer:
            await super().send_state()

    async def receive_updates(self):
        if self.multiplayer:
            await super().receive_updates()
