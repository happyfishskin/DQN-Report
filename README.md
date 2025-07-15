# 🧠 Deep Q-Network (DQN) Projects

本專案包含三個使用 DQN 演算法的強化學習實作，涵蓋經典控制問題（CartPole）與 Atari 遊戲（Pong），並進一步引入增強技術如 Multi-Step Learning 與 Prioritized Experience Replay。

---

## 📁 專案結構

```
.
├── task1_dqn.py              # Task 1: CartPole 訓練程式
├── task1_inferance.py        # Task 1: 推論並輸出動畫
├── task2_dqn_on_Atari_pong.py# Task 2: 基礎 Atari Pong 訓練
├── task2_inferance.py        # Task 2: Atari Pong 推論
├── task3_enhanced_dqn_Atari_pong.py # Task 3: 增強版 DQN
├── task3_inferance.py        # Task 3: 增強版 Atari Pong 推論
├── requirements.txt          # (建議加入) 所需套件清單
└── README.md                 # 使用說明文件
```

---

## 🧰 環境需求

- Python 3.8+
- PyTorch
- Gym (含 Atari)
- imageio
- matplotlib
- numpy

### 安裝範例：

```bash
pip install -r requirements.txt
```

建議使用 GPU 執行（自動偵測 CUDA 設備），請確保您的環境支援 `torch.cuda`.

---

## 🚀 執行方式

### 🔹 Task 1：CartPole DQN

**訓練**
```bash
python task1_dqn.py --episodes 1000
```

**推論**
```bash
python task1_inferance.py --resume ../best_task1/task1_best.pt --output demo.gif
```

- 使用 MLP 網路結構。
- 動畫輸出至 `../vid_task1_infer/`

---

### 🔹 Task 2：Atari Pong DQN (基本版)

**訓練**
```bash
python task2_dqn_on_Atari_pong.py
```

**推論**
```bash
python task2_inferance.py
```

- 使用 CNN + FrameStack 處理 Atari 畫面。
- 訓練結果存在 `../best_task2/`，動畫輸出至 `../vid_inference/`。

---

### 🔹 Task 3：Atari Pong DQN (增強版)

**訓練**
```bash
python task3_enhanced_dqn_Atari_pong.py
```

- 加入 Multi-Step TD Learning、Prioritized Replay。
- 更佳的收斂速度與穩定性。

**推論**
```bash
python task3_inferance.py
```

- 模型路徑預設為 `../best_task32/best_model.pt`
- 推論動畫儲存於 `../vid_task3_inferance/`

---

## 📊 輸出結果

- `../history_task*/`: 每 50 回儲存一次模型
- `../best_task*/`: 最高 reward 模型快照
- `../vid_task*/`: 評估動畫（.gif）
- `../pic_task*/`: Reward / Loss 曲線圖（.png）

---

## 📌 備註

- 所有訓練均使用 epsilon-greedy 探索策略與 target network 更新。
- 若顯示 `env.reset()` 傳回兩個值，需使用 `state, _ = env.reset()`（符合 Gym 新版 API）。
- 訓練時間依設備與 episode 數量而異，請自行調整 `EPISODES`。
