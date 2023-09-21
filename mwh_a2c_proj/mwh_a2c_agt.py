import gym
from gym import spaces
import vizdoom as vzd
import itertools as it
import numpy as np
#import pandas as pd

class VizdoomEnv(gym.Env):
    def __init__(self, config_file_path):
        super(VizdoomEnv, self).__init__()
        self.game = vzd.DoomGame()
        self.game.load_config(config_file_path)
        self.game.init()

        # Define action and observation space
        self.action_space = spaces.Discrete(len([list(a) for a in it.product([0, 1], repeat=self.game.get_available_buttons_size())]))
        # you can modify observation space based on your needs
        sample_obs = self.game.get_state().screen_buffer
        height, width, channels = sample_obs.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, channels), dtype=np.uint8)

    def reset(self):
        self.game.new_episode()
        return self.game.get_state().screen_buffer

    def step(self, action):
        reward = self.game.make_action(action)
        done = self.game.is_episode_finished()
        obs = self.game.get_state().screen_buffer if not done else None
        return obs, reward, done, {}

    def render(self, mode='human'):
        # Implement this if you'd like to see the game while the agent plays
        pass

    def close(self):
        # Clean up resources, if needed.
        pass


from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from a2c import learn

# Initialize the environment
env = VizdoomEnv(config_file_path="../scenarios/my_way_home.cfg")
env = DummyVecEnv([lambda: env])  # Vectorize the environment
print(env)
# Run the A2C algorithm
model = learn(network='mlp', env=env, total_timesteps=100000)
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
