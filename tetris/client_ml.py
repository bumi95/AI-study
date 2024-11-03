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
        
        # grid를 numpy 배열로 변환
        grid_array = np.zeros((GameConfig.GRID_HEIGHT, GameConfig.GRID_WIDTH), dtype=np.float32)
        for y in range(GameConfig.GRID_HEIGHT):
            for x in range(GameConfig.GRID_WIDTH):
                if self.grid[y][x]:
                    grid_array[y][x] = 1.0  # 블록이 있는 경우 1로 설정
        
        # 현재 조각의 상태를 표현
        current_piece = np.zeros((4, GameConfig.GRID_WIDTH), dtype=np.float32)
        if self.current_piece:  # current_piece가 None이 아닌 경우에만 처리
            piece_height, piece_width = len(self.current_piece['shape']), len(self.current_piece['shape'][0])
            
            for y in range(piece_height):
                for x in range(piece_width):
                    if self.current_piece['shape'][y][x]:
                        grid_x, grid_y = self.current_piece['x'] + x, self.current_piece['y'] + y
                        if 0 <= grid_x < GameConfig.GRID_WIDTH and 0 <= grid_y < 4:
                            current_piece[grid_y, grid_x] = 1

        # 다음 조각의 상태를 표현
        next_piece = np.zeros((4, GameConfig.GRID_WIDTH), dtype=np.float32)
        if self.next_piece:
            piece_height, piece_width = len(self.next_piece['shape']), len(self.next_piece['shape'][0])
            for y in range(piece_height):
                for x in range(piece_width):
                    if self.next_piece['shape'][y][x]:
                        next_piece[y, x] = 1

        # 보유 조각의 상태를 표현
        hold_piece = np.zeros((4, GameConfig.GRID_WIDTH), dtype=np.float32)
        if self.hold_piece:
            piece_height, piece_width = len(self.hold_piece['shape']), len(self.hold_piece['shape'][0])
            for y in range(piece_height):
                for x in range(piece_width):
                    if self.hold_piece['shape'][y][x]:
                        hold_piece[y, x] = 1

        # 게임 보드와 현재 조각 상태를 합쳐서 반환
        state = np.vstack((grid_array, current_piece, next_piece, hold_piece))
        return state.flatten()

    def get_reward(self) -> float:
        """
        현재 상태에 대한 보상을 계산합니다.
        
        :return: 현재 상태에 대한 보상 값
        """
        # 기본 보상: 제거한 라인 수와 점수
        reward = self.lines_cleared * 100 + self.score * 0.1
        
        # 게임 보드의 높이 보상
        heights = [GameConfig.GRID_HEIGHT - next((y for y in range(GameConfig.GRID_HEIGHT) if self.grid[y][x]), GameConfig.GRID_HEIGHT) for x in range(GameConfig.GRID_WIDTH)]
        max_height = max(heights)
        height_reward = (GameConfig.GRID_HEIGHT - max_height) * 0.1
        
        # 구멍 페널티
        holes = sum(1 for x in range(GameConfig.GRID_WIDTH) for y in range(GameConfig.GRID_HEIGHT-1)
                    if not self.grid[y][x] and self.grid[y+1][x])
        hole_penalty = holes * 10
        
        # 표면 평탄도 보상
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        smoothness_reward = (GameConfig.GRID_WIDTH - bumpiness) * 0.1
        
        # 생존 시간 보상
        #survival_reward = self.episode_steps * 0.01
        
        # 최종 보상 계산
        reward += smoothness_reward - hole_penalty + height_reward
        
        # 게임 오버 페널티
        if not self.running:
            reward -= 500
        
        return max(reward, 0)  # 보상이 음수가 되지 않도록 함
        '''
        # 기본 보상: 제거한 라인 수와 점수
        reward = 2**self.lines_cleared + self.score
        
        # 게임 보드의 높이 보상
        #heights = [GameConfig.GRID_HEIGHT - next((y for y in range(GameConfig.GRID_HEIGHT) if self.grid[y][x]), GameConfig.GRID_HEIGHT) for x in range(GameConfig.GRID_WIDTH)]
        #max_height = max(heights)
        #height_reward = (GameConfig.GRID_HEIGHT - max_height) * 0.1
        
        # 게임 보드의 높이 패널티
        heights = [GameConfig.GRID_HEIGHT - next((y for y in range(GameConfig.GRID_HEIGHT) if self.grid[y][x]), GameConfig.GRID_HEIGHT) for x in range(GameConfig.GRID_WIDTH)]
        max_height = max(heights)
        min_height = min(heights)
        #height_penalty = max_height 
        #reward -= height_penalty * 0.1
        
        # 구멍 페널티
        holes = sum(1 for x in range(GameConfig.GRID_WIDTH) for y in range(GameConfig.GRID_HEIGHT-1)
                    if not self.grid[y][x] and self.grid[y+1][x])
        hole_penalty = holes * 0.1
        #hole_penalty = holes
        
        # 표면 평탄도 보상
        bumpiness = sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))
        smoothness_reward = (GameConfig.GRID_WIDTH - bumpiness) * 0.1
        
        # 말이 바닥에 가까워질수록 작은 보상
        piece_bottom_y = self.current_piece['y'] + len(self.current_piece['shape']) - 1
        distance_to_bottom = GameConfig.GRID_HEIGHT - piece_bottom_y
        proximity_reward = (GameConfig.GRID_HEIGHT - distance_to_bottom) * 0.05

        height_difference = max_height - min_height
        balance_penalty = height_difference * 0.1  # 높이 차이에 비례한 페널티
        # 빈 공간 계산
        empty_spaces = [sum(1 for y in range(GameConfig.GRID_HEIGHT) if not self.grid[y][x]) for x in range(GameConfig.GRID_WIDTH)]
        max_empty_spaces = max(empty_spaces)
        min_empty_spaces = min(empty_spaces)
        empty_space_difference = max_empty_spaces - min_empty_spaces
        empty_space_penalty = empty_space_difference * 0.1  # 빈 공간 차이에 비례한 페널티
        
        
        # 최종 보상 계산
        #reward += proximity_reward - balance_penalty - empty_space_penalty #hole_penalty
        reward -= hole_penalty#proximity_reward#empty_space_penalty#balance_penalty
        
        # 말이 바닥에 닿으면 보상 추가
        if piece_bottom_y >= GameConfig.GRID_HEIGHT - 1:
            reward += 1
        
        # 게임 오버 페널티
        if not self.running:
            reward -= 50
            #reward -= 500
        
        return reward
        #return max(reward, 0)  # 보상이 음수가 되지 않도록 함
        '''
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
