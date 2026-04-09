"""
=========================================================
DQN with Stabilization Techniques
=========================================================

[실험 목적]
- CartPole-v1 환경에서 DQN 학습 과정을 시각화
- DQN의 대표적인 안정화 기법 3가지를 코드로 명확히 설명
  1) Experience Replay
  2) Fixed Q-Target (Target Network)
  3) Double DQN

[사용 환경]
- Environment : CartPole-v1
- Algorithm  : DQN (Value-based RL)
- Framework  : PyTorch + Gymnasium
- Logging    : TensorBoard

[적용된 안정화 기법]
1) Experience Replay        
2) Fixed Q-Target Network   
3) Double DQN               

[실행 명령어]
python dqn.py --total-timesteps 50000 --learning-starts 1000
tensorboard --logdir runs/
=========================================================
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# =========================
# 안정화 기법 1) Experience Replay
# =========================
class SimpleReplayBuffer:
    def __init__(self, buffer_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.device = device

    def add(self, obs, next_obs, action, reward, done):
        self.buffer.append((obs, next_obs, action, reward, done))

    def sample(self, batch_size):
        # 버퍼에 있는 수천 개의 기억 중 무작위로 batch_size(128개)를 뽑음
        batch = random.sample(self.buffer, batch_size)
        # 학습하기 좋게 텐서(Tensor)로 변환
        obs, next_obs, actions, rewards, dones = map(np.array, zip(*batch))

        return {
            "observations": torch.tensor(obs, dtype=torch.float32, device=self.device),
            "next_observations": torch.tensor(next_obs, dtype=torch.float32, device=self.device),
            "actions": torch.tensor(actions, dtype=torch.long, device=self.device),
            "rewards": torch.tensor(rewards, dtype=torch.float32, device=self.device),
            "dones": torch.tensor(dones, dtype=torch.float32, device=self.device),
        }

    def __len__(self):
        return len(self.buffer)
    
# Args 클래스
# 학습에 필요한 모든 변수 정의
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = True
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific
    env_id: str = "CartPole-v1"
    total_timesteps: int = 300000 # 전체 학습 시간 50,000 ~ 200,000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10


# =========================
# Environment helper
# =========================
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


# QNetwork 클래스
# 상태 (observation)을 입력받아 각 action에 대한 q-value 출력
# linear -> ReLU -> ... 구조의 간단한 MLP(Multi-Layer Perception)
class QNetwork(nn.Module):
    """
    Q(s, a)를 근사하는 네트워크
    출력 차원 = 행동 개수
    각 출력 값 = Q(s, a)
    """
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# =========================
# Main
# =========================
if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{run_name}") # episode, 학습 loss 기록

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed, 0, args.capture_video, run_name)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    # Networks
    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # =========================
    # 안정화 기법 1) Experience Replay
    # =========================
    rb = SimpleReplayBuffer(args.buffer_size, device)

    start_time = time.time()
    obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):

        # =========================
        # 행동 선택 (정책)
        # π(s) = argmax_a Q(s, a)
        # =========================
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample()]) # 랜덤 행동 
        else:
            q_values = q_network(torch.tensor(obs, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy() # 최적 행동 (활용. exploitation)

        # Step environment
        # 환경 상호작용 및 저장
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)

        # =========================
        # Experience Replay 저장
        # =========================
        real_next_obs = next_obs

        # 행동을 취하고, 결과를 받아 버퍼 rb에 저장
        rb.add(
            obs[0],
            real_next_obs[0],
            actions[0],
            rewards[0],
            terminations[0],
        )

        obs = next_obs

        # =========================
        # Training
        # =========================
        if global_step > args.learning_starts and global_step % args.train_frequency == 0:
            if len(rb) >= args.batch_size:
                data = rb.sample(args.batch_size)

            # ==================================================
            # 안정화 기법 3) Double DQN (TD Target 계산)
            #
            # a* = argmax_a Q_online(s', a)
            # y  = r + γ * Q_target(s', a*)
            # ==================================================
            with torch.no_grad():

                # 행동 선택: online network가 담당
                next_actions = q_network(
                    data["next_observations"]
                ).argmax(dim=1)

                # 가치 평가: target network가 담당
                target_q = target_network(
                    data["next_observations"]
                ).gather(1, next_actions.unsqueeze(1)).squeeze()

                # td target 계산
                td_target = (
                    data["rewards"]
                    + args.gamma * target_q
                    * (1 - data["dones"])
                )

            old_val = q_network(
                data["observations"]
            ).gather(1, data["actions"].unsqueeze(1)).squeeze()

            loss = F.mse_loss(td_target, old_val) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("losses/td_loss", loss.item(), global_step)
        # =========================
        # 안정화 기법 2) Target Network Update
        # =========================
        if global_step % args.target_network_frequency == 0:
            for target_param, q_param in zip(
                target_network.parameters(),
                q_network.parameters(),
            ):
                target_param.data.copy_(
                    args.tau * q_param.data + (1.0 - args.tau) * target_param.data
                )

    envs.close()
    writer.close()
