import vizdoom as vzd
import keyboard

def main():
    game = vzd.DoomGame()
    game.load_config("../scenarios/my_way_home.cfg")
    game.init()

    actions = [
        [0, 0, 1, 0, 0],  # Forward (W)
        [1, 0, 0, 0, 0],  # Turn Left (A)
        [0, 1, 0, 0, 0],  # Turn Right (D)
        [0, 0, 0, 1, 0],  # Move Left
        [0, 0, 0, 0, 1],  # Move Right
    ]

    episodes = 10
    for _ in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            game.set_render_hud(False)
            game.set_render_crosshair(False)
            game.set_render_weapon(False)
            game.set_render_decals(False)
            game.set_render_particles(False)
            game.set_window_visible(True)
            game.set_mode(vzd.Mode.ASYNC_PLAYER)

            if keyboard.is_pressed('w'):
                game.make_action(actions[0])
            elif keyboard.is_pressed('a'):
                game.make_action(actions[1])
            elif keyboard.is_pressed('d'):
                game.make_action(actions[2])

        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    game.close()

if __name__ == '__main__':
    main()
