# task1: CartPole DQN Trainer with MLP and CLI support

import os
import argparse
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import imageio

# ==== CLI ==== 
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, help='path to checkpoint')
parser.add_argument('--episodes', type=int, default=10000)
args = parser.parse_args()

# ==== 超參數 ====
GAMMA = 0.99
LR = 1e-3
MEMORY_SIZE = 10000
BATCH_SIZE = 64
TARGET_UPDATE = 100
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 500

os.makedirs("../history_task1", exist_ok=True)
os.makedirs("../best_task1", exist_ok=True)
os.makedirs("../pic_task1", exist_ok=True)
os.makedirs("../vid_task1", exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ==== MLP ==== 
class MLP(nn.Module):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.FloatTensor(state).to(device),
            torch.LongTensor(action).unsqueeze(1).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

def epsilon_by_step(step):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * step / EPSILON_DECAY)

def evaluate(policy_net, env_name='CartPole-v1', filename='task1_eval.gif'):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    total_reward = 0
    images = []

    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy_net(state_tensor).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        images.append(env.render())

    imageio.mimsave(f"../vid_task1/{filename}", images, fps=30)
    print(f"🎥 Saved {filename} | Total Reward: {total_reward}")
    env.close()

# ==== 訓練函數 ====
def train():
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = MLP(state_dim, action_dim).to(device)
    target_net = MLP(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    if args.resume:
        policy_net.load_state_dict(torch.load(args.resume))
        print(f"📦 Resumed from {args.resume}")

    step_count = 0
    best_reward = float('-inf')
    rewards, losses = [], []

    for episode in range(args.episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            epsilon = epsilon_by_step(step_count)
            step_count += 1
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > BATCH_SIZE:
                states, actions, rewards_b, next_states, dones = memory.sample(BATCH_SIZE)
                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q = rewards_b + GAMMA * next_q_values * (1 - dones)
                loss = nn.MSELoss()(q_values, expected_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())

            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        rewards.append(total_reward)
        print(f"🎮 Episode {episode} - Reward: {total_reward:.2f} - Epsilon: {epsilon:.3f}")

        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), f"../history_task1/task1_ep{episode}.pt")
            evaluate(policy_net, filename=f"task1_eval_ep{episode}.gif")

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy_net.state_dict(), f"../best_task1/task1_best.pt")
            evaluate(policy_net, filename=f"task1_best_ep{episode}.gif")

    torch.save(policy_net.state_dict(), f"../history_task1/task1_final.pt")
    env.close()

    # ==== 畫圖 ====
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Episode Rewards')

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', color='red')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.grid()
    plt.title('Loss Curve')

    plt.tight_layout()
    plt.savefig("../pic_task1/task1_result.png")
    plt.show()

if __name__ == "__main__":
    train()
