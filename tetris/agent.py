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
        
        # 하이퍼파라미터 설정
        self.gamma = 0.99  # 할인율
        self.entropy_coef = 0.01  # 엔트로피 계수
        self.max_grad_norm = 0.5  # gradient clipping
        
        # 학습 관련 변수들
        self.training = True
        #self.total_steps = 0
        
    def get_action(self, state):
        if self.training:
            self.model.train()  # 학습 모드로 설정
        else:
            self.model.eval()  # 평가 모드로 설정
            
        # 상태 전처리
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        
        # 배치 차원 확인 및 추가
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # GPU로 이동
        state = state.to(self.device)
        
        # 정책과 가치 얻기 (requires_grad 설정)
        if self.training:
            policy, value = self.model(state)
        else:
            with torch.no_grad():
                policy, value = self.model(state)
        
        # 정책이 올바른지 확인
        if torch.any(torch.isnan(policy)) or torch.any(torch.isinf(policy)):
            logging.warning("비정상적인 정책 분포가 감지되었습니다.")
            policy = torch.ones(self.action_size).to(self.device) / self.action_size
        
        if self.training:
            # 학습 중일 때는 정책에서 샘플링
            action_dist = torch.distributions.Categorical(policy)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return action, log_prob, value
        else:
            # 평가 중일 때는 최적의 행동 선택
            action = torch.argmax(policy, dim=-1)
            return action, None, value

    
    def update(self, trajectory, global_model, total_reward, best_reward):
        # 모델을 학습 모드로 설정
        self.model.train()
        
        # 디바이스로 이동
        log_probs = torch.stack([x[0] for x in trajectory]).to(self.device)
        values = torch.cat([x[1] for x in trajectory]).to(self.device)
        rewards = torch.tensor([x[2] for x in trajectory], dtype=torch.float32).to(self.device)
        
        # GAE 계산
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
        
        # 손실 계산 (requires_grad가 유지되도록)
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = 0.5 * (values - rewards).pow(2).mean()
        
        # 엔트로피 보너스 추가
        policy_entropy = (-log_probs * torch.exp(log_probs)).mean()
        
        # 총 손실 계산
        total_loss = actor_loss + critic_loss - self.entropy_coef * policy_entropy
        
        # 그래디언트 계산 및 클리핑
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # 로컬 모델 업데이트
        self.optimizer.step()
        
        # 글로벌 모델 업데이트 (더 나은 총 보상을 달성했을 때)
        if total_reward > best_reward:
            global_model.load_state_dict(self.model.state_dict())
            
            # 성능 향상 정도에 따른 동적 조정
            improvement_ratio = total_reward / (best_reward + 1e-8)
            
            # 1. 학습률 동적 조정
            if improvement_ratio > 1.5:  # 50% 이상 향상
                new_lr = self.optimizer.param_groups[0]['lr'] * 1.2
                self.optimizer.param_groups[0]['lr'] = min(new_lr, 0.001)  # 최대 학습률 제한
                #logging.info(f"학습률 증가: {new_lr:.6f}")
            elif improvement_ratio < 1.05:  # 5% 미만 향상
                new_lr = self.optimizer.param_groups[0]['lr'] * 0.8
                self.optimizer.param_groups[0]['lr'] = max(new_lr, 0.00001)  # 최소 학습률 제한
                #logging.info(f"학습률 감소: {new_lr:.6f}")
            
            # 2. 엔트로피 계수 조정
            if improvement_ratio > 1.3:  # 30% 이상 향상
                self.entropy_coef *= 0.9  # 탐색 감소
                #logging.info(f"엔트로피 계수 감소: {self.entropy_coef:.4f}")
            elif improvement_ratio < 1.1:  # 10% 미만 향상
                self.entropy_coef = min(self.entropy_coef * 1.1, 0.05)  # 탐색 증가
                #logging.info(f"엔트로피 계수 증가: {self.entropy_coef:.4f}")
            
            # 3. 경험 리플레이 우선순위 조정
            #priority = min(improvement_ratio, 2.0)  # 우선순위 상한 설정
            #self.total_steps += 1
            
            # 4. 모델 체크포인트 저장
            #if improvement_ratio > 1.5:  # 50% 이상의 큰 향상
            #    self.save_checkpoint(f"checkpoint_improvement_{self.total_steps}.pth")
            
            #logging.info(f"성능 향상 감지: {improvement_ratio:.2f}배")
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': policy_entropy.item(),
            'total_loss': total_loss.item(),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item(),
            'improvement_ratio': improvement_ratio if total_reward > best_reward else 1.0,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'entropy_coef': self.entropy_coef
        }
    
    def save_checkpoint(self, filename):
        """중요한 성능 향상이 있을 때 체크포인트 저장"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'entropy_coef': self.entropy_coef
        }
        model_dir = os.path.join(os.getcwd(), "checkpoints")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        path = os.path.join(model_dir, filename)
        torch.save(checkpoint, path)
        logging.info(f"체크포인트 저장됨: {path}")
    
    def update_global_model(self, local_agent):
        # Copy the local agent's gradients to the global model
        for global_param, local_param in zip(self.model.parameters(), local_agent.model.parameters()):
            if local_param.grad is not None:
                if global_param.grad is None:
                    global_param.grad = copy.deepcopy(local_param.grad)
                else:
                    global_param.grad += local_param.grad
        
        # Apply the gradients to update the global model
        self.optimizer.step()
        self.optimizer.zero_grad()
        
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
