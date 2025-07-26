import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F

# --- 元のスクリプトからハイパーパラメータを再現 ---
# SB3/rl_zoo3のA2Cのデフォルトに近い設定
hyperparams = {
    "n_steps": 8,
    "gamma": 0.9,
    "gae_lambda": 0.9,
    "ent_coef": 0.0,
    "vf_coef": 0.4,
    "max_grad_norm": 0.5,
    "learning_rate": 7e-4,
    "policy_kwargs": {"log_std_init": -2.0, "ortho_init": False},
    "total_timesteps": 400_000,
    "env_id": "Pendulum-v1",
    "seed": 43,
}

# --- Actor-Critic ネットワークの定義 ---
# SB3のデフォルト(64, 64, Tanh)を模倣
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, log_std_init):
        super(ActorCritic, self).__init__()
        # ActorとCriticで共有するネットワーク部分
        self.shared_net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        # Actorの出力層 (行動の平均値)
        self.actor_mean = nn.Linear(64, action_dim)
        # Criticの出力層 (状態価値)
        self.critic_value = nn.Linear(64, 1)

        # 行動の標準偏差(log_std)を学習可能なパラメータとして定義
        self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)

        # 重みの初期化 (ortho_init=Falseなのでデフォルトのまま)

    def forward(self, x):
        # 共有ネットワークを通して特徴量を抽出
        shared_features = self.shared_net(x)
        # 行動の平均値と状態価値を計算
        action_mean = self.actor_mean(shared_features)
        value = self.critic_value(shared_features)
        
        # 行動の確率分布(正規分布)を作成
        std = torch.exp(self.log_std)
        dist = Normal(action_mean, std)
        
        return dist, value

# --- メインの学習処理 ---
def train():
    # 環境の初期化
    env = gym.make(hyperparams["env_id"])
    torch.manual_seed(hyperparams["seed"])
    np.random.seed(hyperparams["seed"])
    env.reset(seed=hyperparams["seed"])

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # モデルとオプティマイザの初期化
    # SB3のA2CはRMSpropを使用
    model = ActorCritic(obs_dim, action_dim, hyperparams["policy_kwargs"]["log_std_init"])
    optimizer = optim.RMSprop(model.parameters(), lr=hyperparams["learning_rate"], alpha=0.99, eps=1e-5)

    # 学習ループ
    obs, _ = env.reset()
    total_steps = 0
    
    # --- 学習進捗トラッキング用 ---
    episode_rewards = []
    current_episode_reward = 0
    
    print("PyTorchによるA2C学習を開始します...")

    while total_steps < hyperparams["total_timesteps"]:
        # --- 1. ロールアウト(n_steps分のデータ収集) ---
        log_probs, values, rewards, dones, states = [], [], [], [], []

        for _ in range(hyperparams["n_steps"]):
            total_steps += 1
            
            # 現在の状態をテンソルに変換
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            states.append(state_tensor)
            
            # Actor-Criticモデルから行動分布と状態価値を取得
            dist, value = model(state_tensor)
            
            # 行動をサンプリング
            action = dist.sample()
            log_prob = dist.log_prob(action).sum() # .sum()は多次元アクション用

            # 環境を1ステップ進める
            next_obs, reward, terminated, truncated, _ = env.step(action.cpu().numpy().flatten())
            done = terminated or truncated
            
            # 報酬をトラッキング
            current_episode_reward += reward

            # データを保存
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            dones.append(torch.tensor([1.0 - done], dtype=torch.float32))
            
            obs = next_obs
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()

        # --- 2. GAE(Generalized Advantage Estimation)とリターンの計算 ---
        # 最後の状態の価値を計算
        next_state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        _, next_value = model(next_state_tensor)
        
        # valuesとdonesを逆順にして計算しやすくする
        values = values + [next_value]
        gae = 0
        returns = []
        # 後ろのステップから順に計算
        for i in reversed(range(len(rewards))):
            # TD誤差(delta) = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[i] + hyperparams["gamma"] * values[i+1] * dones[i] - values[i]
            # GAE = delta + gamma * lambda * gae_{t+1}
            gae = delta + hyperparams["gamma"] * hyperparams["gae_lambda"] * dones[i] * gae
            # リターン G_t = A_t + V(s_t)
            returns.insert(0, gae + values[i])

        # --- 3. 損失の計算とモデルの更新 ---
        # 収集したデータをテンソルにまとめる
        states = torch.cat(states)
        log_probs = torch.stack(log_probs)
        returns = torch.cat(returns).detach() # 勾配計算に不要
        
        # 現在のポリシーで再度、行動分布と価値を計算
        dist, values_pred = model(states)
        values_pred = values_pred.squeeze()
        
        # Advantage A_t = G_t - V(s_t)
        advantages = returns - values_pred
        
        # Policy loss (Actor)
        policy_loss = -(log_probs * advantages.detach()).mean()
        
        # Value loss (Critic)
        value_loss = F.mse_loss(returns, values_pred)
        
        # Entropy loss (正則化項)
        entropy_loss = dist.entropy().mean()
        
        # Total loss
        loss = (policy_loss 
                + hyperparams["vf_coef"] * value_loss 
                - hyperparams["ent_coef"] * entropy_loss)

        # 勾配計算と更新
        optimizer.zero_grad()
        loss.backward()
        # 勾配クリッピング
        nn.utils.clip_grad_norm_(model.parameters(), hyperparams["max_grad_norm"])
        optimizer.step()

        if total_steps % 5000 == 0:
            if len(episode_rewards) > 0:
                mean_reward = np.mean(episode_rewards[-100:])
                print(f"Steps: {total_steps}/{hyperparams['total_timesteps']}, Loss: {loss.item():.4f}, Mean Reward: {mean_reward:.2f}")
            else:
                print(f"Steps: {total_steps}/{hyperparams['total_timesteps']}, Loss: {loss.item():.4f}")

    print("学習が完了しました。")
    env.close()

if __name__ == "__main__":
    train()
