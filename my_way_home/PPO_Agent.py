import time
from stable_baselines3 import PPO
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
        pass

    def close(self):
        self.game.close()

env = VizDoomMyWayHomeEnv("../scenarios/my_way_home.cfg")
env = DummyVecEnv([lambda: env])
device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPO("CnnPolicy", env, device=device, verbose=1)

## Snapshot of the training
#model = PPO.load("PPO_task1", env)

## only keep the neural network with a brand new environment.
#checkpoint = torch.load("PPO_task1_NN.pth")
#model.policy.load_state_dict(checkpoint["model_state_dict"])
#model.policy.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

# Training Process
episodes = 1
max_steps = 10000

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

        if done:
            break

    episode_durations.append(time.time() - start_time)
    episode_rewards.append(total_reward)
    print(f"PPO Agent: Episode {episode + 1} - Duration: {episode_durations[-1]:.2f} seconds - Total Reward: {episode_rewards[-1]}")

# Print two lists at the end.
print("Durations:", episode_durations)
print("Rewards:", str(episode_rewards))

model.save("PPO_task1", {"info": "my additional info"})
torch.save({
    "model_state_dict": model.policy.state_dict(),
    "optimizer_state_dict": model.policy.optimizer.state_dict(),
}, "PPO_task1_NN.pth")