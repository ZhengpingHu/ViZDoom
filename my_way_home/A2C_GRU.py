import torch.nn as nn
import torch
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import NatureCNN
from vizdoom import DoomGame, ScreenResolution, Button, Mode
from gym import Env, spaces
import numpy as np
import time
from stable_baselines3.common.vec_env import SubprocVecEnv
from datetime import datetime

class VizDoomMyWayHomeEnv(Env):
    def __init__(self, config_file_path):
        super(VizDoomMyWayHomeEnv, self).__init__()

        self.game = DoomGame()
        self.game.load_config(config_file_path)
        self.game.set_screen_resolution(ScreenResolution.RES_640X480)
        self.game.add_available_button(Button.MOVE_FORWARD)
        self.game.add_available_button(Button.TURN_LEFT)
        self.game.add_available_button(Button.TURN_RIGHT)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()
        self.game.set_ticrate(350)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 640, 480), dtype=np.uint8)

    def reset(self):
        self.game.new_episode()
        return np.array(self.game.get_state().screen_buffer).transpose(0, 2, 1)

    def step(self, action):
        action_one_hot = [0, 0, 0]
        action_one_hot[action] = 1
        reward = self.game.make_action(action_one_hot)
        done = self.game.is_episode_finished()
        if not done:
            next_state = np.array(self.game.get_state().screen_buffer).transpose(0,2,1)
        else:
            next_state = np.zeros((3, 640, 480))
        return next_state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        self.game.close()


class CustomGRUPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, net_arch=None, activation_fn=nn.ReLU, *args,
                 **kwargs):
        super(CustomGRUPolicy, self).__init__(observation_space, action_space, lr_schedule, net_arch, activation_fn,
                                              *args, **kwargs)

        # Overriding the default feature extractor (CNN)
        self.features_extractor = NatureCNN(observation_space, features_dim=512)

        # The GRU layer
        self.gru = nn.GRU(input_size=512, hidden_size=256, num_layers=1, batch_first=True)
        self.gru_hidden = None
        self.last_features = None

    def _get_latent(self, obs: torch.Tensor) -> torch.Tensor:
        # Extract features using the CNN
        features = self.features_extractor(obs)

        # If the agent is starting a new episode, reset the hidden state
        if self.gru_hidden is None or self.last_features is None:
            self.gru_hidden = torch.zeros(1, features.size(0), 256).to(features.device)
            self.last_features = torch.zeros_like(features)

        features_seq = torch.stack([self.last_features, features], dim=1)
        self.last_features = features

        # Pass the sequence through the GRU
        gru_out, self.gru_hidden = self.gru(features_seq, self.gru_hidden)

        return gru_out[:, -1, :]

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        if obs.shape[0] == 1:  # Beginning of a new episode
            self.gru_hidden = None
            self.last_features = None
        return super().forward(obs, deterministic)

def make_env(config_file_path):
    def _init():
        return VizDoomMyWayHomeEnv(config_file_path)
    return _init


def main():
    num_envs = 8
    envs = [make_env("../scenarios/my_way_home.cfg") for _ in range(num_envs)]
    env = SubprocVecEnv(envs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = A2C(CustomGRUPolicy, env, device=device, verbose=1)

    episodes = 1000
    max_steps = 100000

    episode_durations = []
    episode_rewards = []

    for episode in range(episodes):
        obs = env.reset()
        total_reward = 0
        start_time = time.time()

        for timestep in range(max_steps):
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if all(done):
                break

        episode_durations.append(time.time() - start_time)
        episode_rewards.append(total_reward)
        print(f"A2C Agent: Episode {episode + 1} - Duration: {episode_durations[-1]:.2f} seconds - Total Reward: {episode_rewards[-1]}")

    print("Durations:", episode_durations)
    print("Rewards:", episode_rewards)

    now = datetime.now()

    formatted_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"A2C_GRU_task1_{formatted_date_time}.zip"
    nn_filename = f"A2C_GRU_task_NN_{formatted_date_time}.pth"
    model.save(model_filename, {"info": "my additional info"})
    torch.save({
        "model_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": model.policy.optimizer.state_dict(),
    }, nn_filename)

if __name__ == '__main__':
    main()