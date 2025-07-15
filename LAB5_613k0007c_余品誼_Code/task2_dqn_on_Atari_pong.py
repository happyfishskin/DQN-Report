#task2
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib.pyplot as plt
import imageio
import os
from gym.wrappers import AtariPreprocessing, FrameStack

# ==== 超參數 ====
GAMMA = 0.99
LR = 1e-4
MEMORY_SIZE = 20000
BATCH_SIZE = 16
TARGET_UPDATE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 1_000_000
EPISODES = 3000

# ==== 自動建立資料夾 ====
os.makedirs("../history_task2", exist_ok=True)
os.makedirs("../best_task2", exist_ok=True)
os.makedirs("../pic_task2", exist_ok=True)
os.makedirs("../vid_task2", exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ CUDA available:", torch.cuda.is_available())
print("✅ Using device:", device)

# ==== 建立 Atari 環境 ====
def make_env(env_name="ALE/Pong-v5", render_mode=None):
    base_env = gym.make(env_name, render_mode=render_mode, frameskip=1)
    env = AtariPreprocessing(base_env, grayscale_obs=True, scale_obs=True, frame_skip=4)
    env = FrameStack(env, num_stack=4)
    return env

# ==== Q-Network ====
class CNN(nn.Module):
    def __init__(self, action_dim):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (
            torch.stack(state).to(device),
            torch.LongTensor(action).unsqueeze(1).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.stack(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

# ==== 模擬影片保存 ====
def save_evaluation_gif(policy_net, env_name="ALE/Pong-v5", filename="pong_eval_final.gif"):
    env = make_env(env_name=env_name, render_mode="rgb_array")
    state, _ = env.reset()
    done = False
    images = []
    total_reward = 0

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        images.append(env.render())
        state = next_state
        total_reward += reward

    env.close()
    imageio.mimsave(f"../vid_task2/{filename}", images, fps=30)
    print(f"🎥 模擬影片已儲存為 ../vid_task2/{filename} | 得分: {total_reward}")

# ==== 訓練主函數 ====
def train():
    env = make_env(render_mode=None)
    action_dim = env.action_space.n
    policy_net = CNN(action_dim).to(device)
    target_net = CNN(action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    step_count = 0
    episode_rewards = []
    losses = []
    best_reward = float('-inf')
    eval_counter = 0

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-1. * step_count / EPSILON_DECAY)
            step_count += 1

            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(np.array(state), dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)
                q_values = policy_net(states).gather(1, actions)
                next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q = rewards + GAMMA * next_q_values * (1 - dones)

                loss = nn.MSELoss()(q_values, expected_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)
        print(f"🎮 Episode {episode} - Reward: {episode_reward:.2f} - Epsilon: {epsilon:.3f}")

        if episode % 100 == 0:
            torch.save(policy_net.state_dict(), f"../history_task2/pong_ep{episode}.pt")
            print("💾 快照已儲存")

        # 儲存新最佳模型與 GIF
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), "../best_task2/best_model.pt")
            print(f"🏆 新最佳模型（Reward: {best_reward:.2f}）！進行評估...")

            # 強制加載最新模型
            policy_net.load_state_dict(torch.load("../best_task2/best_model.pt"))
            
            # 儲存 GIF
            save_evaluation_gif(policy_net, filename=f"pong_eval_best_ep{episode}.gif")
        else:
            print(f"Episode {episode} did not surpass best reward.")


    torch.save(policy_net.state_dict(), "../history_task2/LAB5_613k0007c_task2_pong_final.pt")
    env.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(losses, label='Loss', color='red')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid()

    plt.tight_layout()
    plt.savefig("../pic_task2/LAB5_613k0007c_task2_reward_loss.png")
    plt.show()

    save_evaluation_gif(policy_net)  # 最終評估
    return policy_net

if __name__ == "__main__":    
    model = train()
