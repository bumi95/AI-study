import torch
import torch.optim as optim
from model import A3CModel
#import numpy as np
import logging
import os
import copy

logging.basicConfig(level=logging.INFO)

class A3CAgent:
    def __init__(self, input_size, action_size, lr=0.0001):
        self.input_size = input_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델을 GPU로 이동
        self.model = A3CModel(input_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        # last_valid_policy 초기화 추가
        self.last_valid_policy = torch.ones(action_size).to(self.device) / action_size
        
        # 하이퍼파라미터 설정
        self.gamma = 0.99  # 할인율
        self.entropy_coef = 0.01  # 엔트로피 계수
        self.max_grad_norm = 0.5  # gradient clipping
        
        # 학습 관련 변수들
        self.training = True
        self.total_steps = 0
        
    def get_action(self, state):
        if self.training:
            self.model.train()
            
            # 상태 전처리
            #if not isinstance(state, torch.Tensor):
            #    state = torch.from_numpy(state).float()
            
            # 배치 차원 확인 및 추가
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # GPU로 이동
            state = state.to(self.device)
            
            # 정책과 가치 얻기
            policy, value = self.model(state)
            
            # epsilon-greedy 전략 구현
            epsilon = max(0.1, 1.0 - (self.total_steps / 50000))  # 점진적으로 감소
            if torch.rand(1).item() < epsilon:
                # 무작위 탐색
                action = torch.randint(0, self.action_size, (1,)).to(self.device)
                action_dist = torch.distributions.Categorical(policy)
                log_prob = action_dist.log_prob(action)
            else:
                # 정책 기반 선택
                policy = self.validate_policy(policy)
                #if torch.any(torch.isnan(policy)) or torch.any(torch.isinf(policy)):
                #    policy = torch.ones(self.action_size).to(self.device) / self.action_size
                
                # 정책에 온도 파라미터 적용
                temperature = max(0.5, 1.0 - (self.total_steps / 100000))
                policy = (policy / temperature).softmax(dim=-1)
                
                action_dist = torch.distributions.Categorical(policy)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
            
            self.total_steps += 1
            return action, log_prob, value
        else:
            self.model.eval()
            with torch.no_grad():
                policy, value = self.model(state.to(self.device))
                # 평가시에는 최적의 행동 선택
                action = torch.argmax(policy, dim=-1)
                return action, None, value

    
    def update(self, trajectory, global_model, total_reward, best_reward):
        self.model.train()
        
        # 기존 GAE 계산
        log_probs = torch.stack([x[0] for x in trajectory]).to(self.device)
        values = torch.cat([x[1] for x in trajectory]).to(self.device)
        rewards = torch.tensor([x[2] for x in trajectory], dtype=torch.float32).to(self.device)
        
        # 리워드 정규화 추가
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # GAE 계산 개선
        gae = 0
        advantages = []
        next_value = 0
        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_value - v
            gae = delta + self.gamma * 0.95 * gae
            next_value = v
            advantages.insert(0, gae)
        
        advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 손실 계산 개선
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = 0.5 * (values - rewards).pow(2).mean()
        policy_entropy = (-log_probs * torch.exp(log_probs)).mean()
        
        # 손실 가중치 조정
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * policy_entropy
        
        # 그래디언트 계산 및 클리핑
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        # 글로벌 모델 업데이트 조건 완화
        improvement_ratio = total_reward / (best_reward + 1e-8)
        update_probability = min(improvement_ratio, 1.0)  # 확률적 업데이트
        
        if torch.rand(1).item() < update_probability:
            # 부분적 모델 업데이트 (Soft Update)
            tau = 0.1  # 업데이트 비율
            for global_param, local_param in zip(global_model.parameters(), self.model.parameters()):
                global_param.data.copy_(
                    global_param.data * (1.0 - tau) + local_param.data * tau
                )
            
            # 동적 하이퍼파라미터 조정
            if improvement_ratio > 1.2:  # 20% 이상 향상
                self.entropy_coef = max(0.01, self.entropy_coef * 0.95)  # 탐색 감소
                new_lr = min(self.optimizer.param_groups[0]['lr'] * 1.1, 0.001)
                self.optimizer.param_groups[0]['lr'] = new_lr
            elif improvement_ratio < 0.8:  # 성능 저하
                self.entropy_coef = min(0.05, self.entropy_coef * 1.05)  # 탐색 증가
                new_lr = max(self.optimizer.param_groups[0]['lr'] * 0.95, 0.00001)
                self.optimizer.param_groups[0]['lr'] = new_lr
                #logging.info(f"엔트로피 계수 증가: {self.entropy_coef:.4f}")
            
            # 3. 경험 리플레이 우선순위 조정
            #priority = min(improvement_ratio, 2.0)  # 우선순위 상한 설정
            #self.total_steps += 1
            
            # 4. 모델 체크포인트 저장
            #if improvement_ratio > 1.5:  # 50% 이상의 큰 향상
            #    self.save_checkpoint(f"checkpoint_improvement_{self.total_steps}.pth")
            
            #logging.info(f"성능 향상 감지: {improvement_ratio:.2f}배")
        
    def save_model(self):
        model_dir = os.path.join(os.getcwd(), "model")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def load_model(self):
        model_dir = os.path.join(os.getcwd(), "model")
        model_path = os.path.join(model_dir, "model.pth")
        self.model.load_state_dict(torch.load(model_path))
        logging.info(f"Model loaded from {model_path}")

    def preprocess_state(self, state, width, height):
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        
        # 게임 보드와 다음 조각 정보 분리
        board_size = width * height
        board_state = state[:board_size]
        next_piece_state = state[board_size:]
        
        # 보드 상태 정규화
        board_mean = board_state.mean()
        board_std = board_state.std() + 1e-8
        normalized_board = (board_state - board_mean) / board_std
        
        # 다시 결합
        normalized_state = torch.cat([normalized_board, next_piece_state])
        
        if normalized_state.dim() == 1:
            normalized_state = normalized_state.unsqueeze(0)
        
        return normalized_state.to(self.device)

    def validate_policy(self, policy):
        if torch.any(torch.isnan(policy)) or torch.any(torch.isinf(policy)):
            logging.warning("비정상적인 정책 감지, 이전 정책 사용")
            return self.last_valid_policy
        
        self.last_valid_policy = policy.clone().detach()
        return policy
