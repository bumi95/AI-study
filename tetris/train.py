#import numpy as np
import pygame
from agent import A3CAgent
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from client_ml import TetrisGameML
import asyncio
import torch.multiprocessing as mp
#from torch.distributions import Categorical

def worker(num_episodes):
    pygame.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = f"{30},{30}"
    game = TetrisGameML()
    local_agent = A3CAgent(game.get_state().shape[0], len(game.action_space))
    
    #local_agent.model.load_state_dict(global_agent.model.state_dict())
    
    record = 0
    record_reward = 0
    
    for episode in range(num_episodes):
        state = game.reset()
        trajectory = []
        total_reward = 0
        done = False
        total_score = 0
        
        while not done:
            state = local_agent.preprocess_state(state, 800, 600)
            action, log_prob, value = local_agent.get_action(state)
            game_action = game.action_space[action.item()]
            next_state, reward, done, info = asyncio.run(game.step(game_action))
            
            trajectory.append((log_prob, value, reward))
            
            state = next_state
            total_reward += reward
            total_score = info["score"]
            
            game.draw_game()
            pygame.display.flip()
            #pygame.time.wait(20)
        
        local_agent.update(trajectory, total_reward, record_reward)
        #global_agent.update_global_model(local_agent)
        
        print(f"총 보상 : {total_reward}")

        if total_score > 0:
            if total_score > record:
                record = total_score
                local_agent.save_model()
            print(f"에피소드 {episode + 1}, 총 보상: {total_reward}, 스코어: {total_score}, 최고 스코어: {record}")
    pygame.quit()
    
    
def train(num_episodes=10000):
    worker(num_episodes)
    '''
    screen_height = 600
    monitor_width = 1920
    sum = 0
    max_columns = 0
    while sum < monitor_width:
        sum += screen_height
        max_columns += 1

    game = TetrisGameML()
    input_size = game.get_state().shape[0]
    action_size = len(game.action_space)
    global_agent = A3CAgent(input_size, action_size)
    
    workers = []
    for i in range(num_workers):
        p = mp.Process(target=worker, args=(i, global_agent, num_episodes // num_workers, input_size, action_size, max_columns))
        p.start()
        workers.append(p)
        
    for p in workers:
        p.join()
        
    global_agent.save_model()
    '''
    
if __name__ == "__main__":
    #mp.set_start_method("spawn")
    train()
