import os
import argparse
import gym
import torch
import torch.nn as nn
import numpy as np
import imageio

# ==== CLI ====
parser = argparse.ArgumentParser()
parser.add_argument('--resume', type=str, required=True, help='path to checkpoint')  # 強制要給模型路徑
parser.add_argument('--output', type=str, default="task1_infer.gif", help='output gif filename')
args = parser.parse_args()

# ==== 裝置設定 ====
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ==== MLP 定義 ====
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

# ==== 推論函數 ====
def inference():
    env = gym.make('CartPole-v1', render_mode='rgb_array')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 建立網路並載入權重
    policy_net = MLP(state_dim, action_dim).to(device)
    policy_net.load_state_dict(torch.load(args.resume, map_location=device))
    policy_net.eval()
    print(f"📦 Loaded model from {args.resume}")

    # 遊玩一場並錄影
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

    os.makedirs("../vid_task1_infer", exist_ok=True)
    save_path = f"../vid_task1_infer/{args.output}"
    imageio.mimsave(save_path, images, fps=30)
    print(f"🎥 Saved inference result to {save_path} | Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    inference()
