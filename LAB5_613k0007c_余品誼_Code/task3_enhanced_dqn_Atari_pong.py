#task3
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
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
MULTI_STEP = 3
ALPHA = 0.6                     # PER 中，決定優先級 p 的重要性 (p^alpha)
BETA_START = 0.4                # PER 中，重要性採樣 (Importance Sampling) 的初始 beta 值
BETA_FRAMES = 1_000_000         # Beta 從初始值增長到 1.0 所需的步數

# ==== 自動建立資料夾 ====
os.makedirs("../history_task32", exist_ok=True)
os.makedirs("../best_task32", exist_ok=True)
os.makedirs("../pic_task32", exist_ok=True)
os.makedirs("../vid_task32", exist_ok=True)


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("CUDA available:", torch.cuda.is_available())
print("Using device:", device)

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

# ==== Multi-Step + Prioritized Experience Replay Buffer ====
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# 實現了 PER 和 Multi-Step Learning 的經驗回放緩衝區
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, n_step):
        self.capacity = capacity
        self.buffer = []
        # 用一個 numpy array 來儲存每個經驗的優先級 (priority)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.alpha = alpha
        self.n_step = n_step
        # 暫存最近 n_step 步的經驗，用於計算 n-step return
        self.nstep_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done):
        # 將單步經驗存入 n-step 緩衝區
        transition = Transition(state, action, reward, next_state, done)
        self.nstep_buffer.append(transition)

        # 如果 n-step 緩衝區還沒滿，就直接返回
        if len(self.nstep_buffer) < self.n_step:
            return

        # 計算 n-step return
        reward, next_state, done = self._get_n_step_info()
        # 取出 n-step 緩衝區中最早的 state 和 action
        state, action = self.nstep_buffer[0].state, self.nstep_buffer[0].action

        # 設定新經驗的初始優先級為當前最高優先級，確保它有機會被抽到
        max_prio = self.priorities.max() if self.buffer else 1.0
        # 將 n-step 經驗存入主緩衝區
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        # 更新對應位置的優先級
        self.priorities[self.pos] = max_prio
        # 更新寫入位置
        self.pos = (self.pos + 1) % self.capacity

    def _get_n_step_info(self):
        # 計算 n-step return 的核心邏輯
        # 從 n-step 緩衝區的最後一筆經驗開始回溯
        reward, next_state, done = self.nstep_buffer[-1].reward, self.nstep_buffer[-1].next_state, self.nstep_buffer[-1].done
        # 迴圈反向遍歷 n-step 緩衝區中的前 n-1 筆經驗
        for transition in reversed(list(self.nstep_buffer)[:-1]):
            r, n_s, d = transition.reward, transition.next_state, transition.done
            # 累加折扣後的獎勵
            reward = r + GAMMA * reward * (1 - d)
            # 如果中間的步驟就結束了，那麼最終的 next_state 和 done 就用那一筆的
            next_state, done = (n_s, d) if d else (next_state, done)
        return reward, next_state, done

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return [], [], [], [], [], [], []

        # 根據優先級計算抽樣概率
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).unsqueeze(1).to(device)

        states, actions, rewards, next_states, dones = zip(*samples)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).unsqueeze(1).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device),
            weights, indices
        )

    def update_priorities(self, indices, priorities):
        for i, p in zip(indices, priorities):
            self.priorities[i] = p

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
    imageio.mimsave(f"../vid_task32/{filename}", images, fps=30)
    print(f"🎥 模擬影片已儲存為 ../vid_task32/{filename} | 得分: {total_reward}")

# ==== 訓練主函數 ====
def train():
    env = make_env(render_mode=None)
    action_dim = env.action_space.n
    policy_net = CNN(action_dim).to(device)
    target_net = CNN(action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = PrioritizedReplayBuffer(MEMORY_SIZE, ALPHA, MULTI_STEP)

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
            beta = min(1.0, BETA_START + step_count * (1.0 - BETA_START) / BETA_FRAMES)
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
                states, actions, rewards, next_states, dones, weights, indices = memory.sample(BATCH_SIZE, beta)
                with torch.no_grad():
                    next_actions = policy_net(next_states).argmax(1, keepdim=True)
                    next_q = target_net(next_states).gather(1, next_actions)
                    target_q = rewards + GAMMA**MULTI_STEP * next_q * (1 - dones)

                current_q = policy_net(states).gather(1, actions)
                loss = (current_q - target_q).pow(2) * weights
                prios = loss + 1e-5
                loss = loss.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                memory.update_priorities(indices, prios.data.cpu().numpy())
                losses.append(loss.item())

            if step_count % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(episode_reward)
        print(f"🎮 Episode {episode} - Reward: {episode_reward:.2f} - Epsilon: {epsilon:.3f}")

        if episode % 50 == 0:
            torch.save(policy_net.state_dict(), f"../history_task32/pong_ep{episode}.pt")
            print("💾 快照已儲存")

        # 儲存新最佳模型與 GIF
        if episode_reward > best_reward:
            best_reward = episode_reward
            torch.save(policy_net.state_dict(), "../best_task32/best_model.pt")
            print(f"🏆 新最佳模型（Reward: {best_reward:.2f}）！進行評估...")

            # 強制加載最新模型
            policy_net.load_state_dict(torch.load("../best_task32/best_model.pt"))
            
            # 儲存 GIF
            save_evaluation_gif(policy_net, filename=f"pong_eval_best_ep{episode}.gif")
        else:
            print(f"Episode {episode} did not surpass best reward.")

    torch.save(policy_net.state_dict(), "../history_task32/LAB5_613k0007c_task3_pong_final.pt")
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
    plt.savefig("../pic_task32/LAB5_613k0007c_task3_reward_loss.png")
    plt.show()

    save_evaluation_gif(policy_net)  # 最終評估
    return policy_net

if __name__ == "__main__":
    model = train()
