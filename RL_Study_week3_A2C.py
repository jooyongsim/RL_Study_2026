import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

# =====================
# 공통 설정 (CPU)
# =====================
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
EPISODES = 1000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Utility: Moving Variance
# =====================
def moving_variance(x, window=20):
    out = []
    for i in range(len(x)):
        start = max(0, i - window + 1)
        out.append(np.var(x[start:i+1]))
    return out

# =====================
# Policy Network
# =====================
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# Value Network (Critic)
# =====================
class ValueNet(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# Unit 4: REINFORCE
# =====================
def train_reinforce():
    env = gym.make(ENV_NAME)
    policy = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    optimizer = optim.Adam(policy.parameters(), lr=LR)

    episode_returns = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).to(DEVICE)
            probs = policy(state_t)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)

        # Monte Carlo return
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(DEVICE)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = 0
        for log_prob, Gt in zip(log_probs, returns):
            loss += -log_prob * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_returns.append(sum(rewards))

    env.close()
    return episode_returns

# =====================
# Unit 6: Advantage Actor-Critic (A2C)
# =====================
def train_a2c():
    env = gym.make(ENV_NAME)
    actor = PolicyNet(env.observation_space.shape[0], env.action_space.n).to(DEVICE)
    critic = ValueNet(env.observation_space.shape[0]).to(DEVICE)

    actor_optim = optim.Adam(actor.parameters(), lr=LR)
    critic_optim = optim.Adam(critic.parameters(), lr=LR)

    episode_returns = []
    advantage_log = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32).to(DEVICE)

            probs = actor(state_t)
            dist = Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done, _, _ = env.step(action.item())
            next_state_t = torch.tensor(next_state, dtype=torch.float32).to(DEVICE)

            value = critic(state_t)
            next_value = critic(next_state_t).detach()

            td_error = reward + GAMMA * next_value * (1 - done) - value
            advantage_log.append(td_error.item())

            # Actor update
            actor_loss = -log_prob * td_error.detach()
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            # Critic update
            critic_loss = td_error.pow(2)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

            state = next_state
            total_reward += reward

        episode_returns.append(total_reward)

    env.close()
    return episode_returns, advantage_log

# =====================
# Main
# =====================
if __name__ == "__main__":
    print("Training REINFORCE (Unit 4)...")
    reinforce_returns = train_reinforce()

    print("Training A2C (Unit 6)...")
    a2c_returns, a2c_advantages = train_a2c()

    # =====================
    # Plot 1: Episode Return
    # =====================
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(reinforce_returns, label="REINFORCE")
    plt.plot(a2c_returns, label="A2C")
    plt.xlabel("Episode")
    plt.ylabel("Episode Return")
    plt.legend()
    plt.title("Episode Return")

    # =====================
    # Plot 2: Return Moving Variance
    # =====================
    window = 20
    plt.subplot(1, 2, 2)
    plt.plot(moving_variance(reinforce_returns, window), label="REINFORCE Var")
    plt.plot(moving_variance(a2c_returns, window), label="A2C Var")
    plt.xlabel("Episode")
    plt.ylabel(f"Return Variance (window={window})")
    plt.legend()
    plt.title("Moving Variance of Return")

    plt.tight_layout()
    plt.show()

    # =====================
    # Plot 3: Advantage Variance (A2C)
    # =====================
    plt.figure()
    plt.plot(moving_variance(a2c_advantages, window=100))
    plt.xlabel("Step")
    plt.ylabel("Advantage Variance")
    plt.title("Moving Variance of Advantage (TD Error) - A2C")
    plt.show()
