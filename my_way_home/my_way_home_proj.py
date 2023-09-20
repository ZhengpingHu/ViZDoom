import vizdoom as vzd
import itertools as it
import random

MAX_STUCK = 100


def initialize_vizdoom(config_file_path):
    game = vzd.DoomGame()
    game.load_config(config_file_path)
    game.init()
    return game


game = initialize_vizdoom("../scenarios/my_way_home.cfg")


class RandomAgent:
    def __init__(self, actions):
        self.actions = actions

    def choose_action(self, _state):
        return random.randint(0, len(self.actions) - 1)


actions = [list(a) for a in it.product([0, 1], repeat=game.get_available_buttons_size())]
agent = RandomAgent(actions)


def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


# Start Training:


episodes = 100
for i in range(episodes):
    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
        action_idx = agent.choose_action(state)
        reward = game.make_action(actions[action_idx])
        stuck_counter = 0
        prev_position = None
        while not game.is_episode_finished():
            current_position = game.get_game_variable(vzd.GameVariable.POSITION_X), game.get_game_variable(
                vzd.GameVariable.POSITION_Y)
            if prev_position is not None and distance(current_position, prev_position) < 1:
                stuck_counter += 1
            else:
                stuck_counter = 0
            if stuck_counter > MAX_STUCK:
                break
            prev_position = current_position
    print(f"Episode {i} finished with total reward: {game.get_total_reward()}")
