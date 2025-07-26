import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import torch.nn.functional as F
from typing import Tuple

# --- ハイパーパラメータ ---
hyperparams = {
    "n_steps": 8,
    "gamma": 0.9,
    "gae_lambda": 0.9,
    "ent_coef": 0.0,
    "vf_coef": 0.4,
    "max_grad_norm": 0.5,
    "learning_rate": 7e-4,
    "total_timesteps": 400_000,
    "env_id": "Pendulum-v1",
    "seed": 43,
    "use_sde": True,
    "log_std_init": -2.0,
}

# --- SB3のStateDependentNoiseDistributionを模倣したクラス ---
class StateDependentNoiseDistribution:
    def __init__(self, action_dim: int, latent_sde_dim: int, log_std_init: float = -2.0):
        self.action_dim = action_dim
        self.latent_sde_dim = latent_sde_dim
        # log_stdは学習可能なパラメータ
        self.log_std = nn.Parameter(torch.ones(latent_sde_dim, action_dim) * log_std_init, requires_grad=True)
        self.weights_dist = None
        self.exploration_mat = None
        self.exploration_matrices = None
        self.sample_weights()

    def sample_weights(self) -> None:
        """ノイズ行列のための重みをサンプリング"""
        std = torch.exp(self.log_std)
        self.weights_dist = Normal(torch.zeros_like(std), std)
        self.exploration_mat = self.weights_dist.rsample()

    def get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        """与えられた特徴量からノイズを生成"""
        return torch.mm(latent_sde, self.exploration_mat)

    def get_distribution(self, mean_actions: torch.Tensor, latent_sde: torch.Tensor) -> Normal:
        """行動の平均と特徴量から最終的な正規分布を生成"""
        # SB3の実装: variance = (latent_sde^2) @ (std^2)
        variance = torch.mm(latent_sde**2, torch.exp(self.log_std) ** 2)
        # ゼロ除算を避けるための小さなepsilon
        return Normal(mean_actions, torch.sqrt(variance + 1e-6))

# --- Actor-Critic ネットワーク (SDE対応) ---
class ActorCriticSDE(nn.Module):
    def __init__(self, obs_dim, action_dim, use_sde=False):
        super(ActorCriticSDE, self).__init__()
        self.use_sde = use_sde
        
        self.shared_net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh())
        self.policy_net = nn.Sequential(nn.Linear(64, 64), nn.Tanh())
        self.value_net = nn.Sequential(nn.Linear(64, 64), nn.Tanh())

        self.actor_mean = nn.Linear(64, action_dim)
        self.critic_value = nn.Linear(64, 1)
        
        if self.use_sde:
            # SDE用の特徴量を生成する層
            self.latent_sde_net = nn.Linear(64, 64) # latent_sde_dim = 64

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        (mean_actions, latent_sde, value) を返す
        """
        shared_features = self.shared_net(x)
        latent_pi = self.policy_net(shared_features)
        latent_vf = self.value_net(shared_features)
        
        mean = self.actor_mean(latent_pi)
        value = self.critic_value(latent_vf)
        
        latent_sde = latent_pi
        if self.use_sde:
            # SB3ではpolicy特徴量をそのまま使うか、別の層を通すか選べる
            # ここでは簡単のため、policy特徴量をそのままSDE特徴量として使用
            pass
            
        return mean, latent_sde, value

# --- メインの学習処理 ---
def train():
    env = gym.make(hyperparams["env_id"])
    torch.manual_seed(hyperparams["seed"])
    np.random.seed(hyperparams["seed"])
    env.reset(seed=hyperparams["seed"])

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = ActorCriticSDE(obs_dim, action_dim, use_sde=hyperparams["use_sde"])
    
    # SDEディストリビューションを初期化
    if hyperparams["use_sde"]:
        # latent_sde_dimはActorCriticSDEのlatent_piの出力次元(64)に合わせる
        sde_dist = StateDependentNoiseDistribution(action_dim, 64, hyperparams["log_std_init"])
        # SDEのlog_stdもオプティマイザの対象に含める
        optimizer = optim.RMSprop([
            {'params': model.parameters()},
            {'params': sde_dist.log_std}
        ], lr=hyperparams["learning_rate"], alpha=0.99, eps=1e-5)
    else:
        # SDEを使わない場合のオプティマイザ（今回はSDEのみテスト）
        optimizer = optim.RMSprop(model.parameters(), lr=hyperparams["learning_rate"], alpha=0.99, eps=1e-5)


    obs, _ = env.reset()
    total_steps = 0
    episode_rewards = []
    current_episode_reward = 0
    
    print(f"PyTorchによるA2C学習を開始します (SDE: {hyperparams['use_sde']})...")

    while total_steps < hyperparams["total_timesteps"]:
        # --- 1. ロールアウト(n_steps分のデータ収集) ---
        # log_probsの代わりにactionsを保存
        actions, values, rewards, dones, states, latent_sdes = [], [], [], [], [], []
        
        if hyperparams["use_sde"]:
            sde_dist.sample_weights()

        for _ in range(hyperparams["n_steps"]):
            total_steps += 1
            state_tensor = torch.FloatTensor(obs).unsqueeze(0)
            
            mean, latent_sde, value = model(state_tensor)
            
            dist = sde_dist.get_distribution(mean, latent_sde)
            noise = sde_dist.get_noise(latent_sde)
            action = mean + noise
            
            # 実行したactionを保存
            actions.append(action)
            
            next_obs, reward, terminated, truncated, _ = env.step(action.detach().cpu().numpy().flatten())
            done = terminated or truncated
            
            current_episode_reward += reward
            values.append(value)
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            dones.append(torch.tensor([1.0 - done], dtype=torch.float32))
            states.append(state_tensor)
            latent_sdes.append(latent_sde)
            
            obs = next_obs
            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0
                obs, _ = env.reset()

        next_state_tensor = torch.FloatTensor(obs).unsqueeze(0)
        _, _, next_value = model(next_state_tensor)
        values = values + [next_value]
        gae = 0
        returns = []
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + hyperparams["gamma"] * values[i+1] * dones[i] - values[i]
            gae = delta + hyperparams["gamma"] * hyperparams["gae_lambda"] * dones[i] * gae
            returns.insert(0, gae + values[i])

        # --- 3. 損失の計算とモデルの更新 ---
        states = torch.cat(states)
        actions = torch.cat(actions)
        latent_sdes = torch.cat(latent_sdes)
        returns = torch.cat(returns).detach()
        
        mean_pred, _, values_pred = model(states)
        values_pred = values_pred.squeeze()
        
        # 最新のポリシーでlog_probとentropyを再計算
        dist_pred = sde_dist.get_distribution(mean_pred, latent_sdes)
        log_probs_pred = dist_pred.log_prob(actions).sum(dim=-1)
        entropy_loss = dist_pred.entropy().mean()
        
        advantages = returns - values_pred
        
        policy_loss = -(log_probs_pred * advantages.detach()).mean()
        value_loss = F.mse_loss(returns, values_pred)
        
        loss = (policy_loss 
                + hyperparams["vf_coef"] * value_loss 
                - hyperparams["ent_coef"] * entropy_loss)

        optimizer.zero_grad()
        loss.backward()
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