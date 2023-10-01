from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from vizdoom import DoomGame, ScreenResolution, Button, Mode
from gym import Env, spaces
import torch
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

        self.action_space = spaces.Discrete(3)  # MOVE_FORWARD, TURN_LEFT, TURN_RIGHT
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
        pass  # Visualization can be added if needed

    def close(self):
        self.game.close()

#loaded_model = A2C.load("task1")
env = VizDoomMyWayHomeEnv("../scenarios/my_way_home.cfg")
env = DummyVecEnv([lambda: env])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = A2C("CnnPolicy", env, device=device, verbose=1)
#model = A2C("CnnPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

model.save("task1", {"info": "my additional info"})
