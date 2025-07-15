import gym
import numpy as np
import torch
import torch.nn as nn
import imageio
import os
from gym.wrappers import AtariPreprocessing, FrameStack

# ==== 自動建立資料夾 ====
os.makedirs("../vid_inference", exist_ok=True)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
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

# ==== 模擬影片保存 ====
def save_inference_gif(policy_net, env_name="ALE/Pong-v5", filename="pong_eval_inference_task2.gif"):
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
    imageio.mimsave(f"../vid_inference/{filename}", images, fps=30)
    print(f"🎥 模擬影片已儲存為 ../vid_inference/{filename} | 得分: {total_reward}")

# ==== 載入模型並進行推論 ====
def inference(model_path="../best/best_model.pt"):
    # 載入已訓練的模型
    action_dim = 6  # 修改為對應的行為數量
    policy_net = CNN(action_dim).to(device)
    policy_net.load_state_dict(torch.load(model_path))
    policy_net.eval()  # 設置為推論模式

    # 開始推論
    save_inference_gif(policy_net, filename="pong_inference_task2.gif")

if __name__ == "__main__":
    # 進行推論
    inference(model_path="../best_task2/best_model.pt")  # 記得修改模型的路徑
