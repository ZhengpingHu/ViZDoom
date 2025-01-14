#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played.
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

import os
from random import choice
from time import sleep

import vizdoom as vzd


if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables depth buffer (turned off by default).
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in-game objects labeling (turned off by default).
    game.set_labels_buffer_enabled(True)

    # Enables buffer with a top-down map of the current episode/level (turned off by default).
    game.set_automap_buffer_enabled(True)

    # Enables information about all objects present in the current episode/level (turned off by default).
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout/geometry, turned off by default).
    game.set_sectors_info_enabled(True)
    
    # Set the game period into 1 second.
    game.set_episode_timeout(vzd.DEFAULT_TICRATE)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Like smoke and blood
    game.set_render_messages(False)  # In-game text messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(
        True
    )  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed to use.
    # This can be done by adding buttons one by one:
    # game.clear_available_buttons()
    # game.add_available_button(vzd.Button.MOVE_LEFT)
    # game.add_available_button(vzd.Button.MOVE_RIGHT)
    # game.add_available_button(vzd.Button.ATTACK)
    # Or by setting them all at once:
    game.set_available_buttons(
        [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK]
    )
    # Buttons that will be used can be also checked by:
    print("Available buttons:", [b.name for b in game.get_available_buttons()])

    # Adds game variables that will be included in state.
    # Similarly to buttons, they can be added one by one:
    # game.clear_available_game_variables()
    # game.add_available_game_variable(vzd.GameVariable.AMMO2)
    # Or:
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    print(
        "Available game variables:",
        [v.name for v in game.get_available_game_variables()],
    )

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    game.set_window_visible(True)

    # Turns on the sound. (turned off by default)
    # game.set_sound_enabled(True)
    # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented,
    # the sound is only useful for humans watching the game.

    # Turns on the audio buffer. (turned off by default)
    # If this is switched on, the audio will stop playing on device, even with game.set_sound_enabled(True)
    # Setting game.set_sound_enabled(True) is not required for audio buffer to work.
    # game.set_audio_buffer_enabled(True)
    # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented.

    # Sets the living reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Enables engine output to console, in case of a problem this might provide additional information.
    # game.set_console_enabled(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    # Define some actions. Each list entry corresponds to declared buttons:
    # MOVE_LEFT, MOVE_RIGHT, ATTACK
    # game.get_available_buttons_size() can be used to check the number of available buttons.
    # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
    actions = [[True, False, False], [False, True, False], [False, False, True]]

    # Run this many episodes
    episodes = 10

    all_enemy_positions = []

for i in range(episodes):
    print("Episode #" + str(i + 1))
    
    enemy_positions = []
    
    game.new_episode()
    
    while not game.is_episode_finished():
        state = game.get_state()
        labels = state.labels

        # 使用屏幕宽度来确定敌人是在左边还是右边
        screen_width = game.get_screen_width()

        for label in labels:
            if label.object_name == "Cacodemon":
                enemy_x = label.x
                position = "left" if enemy_x < screen_width / 2 else "right"
                enemy_positions.append((enemy_x, position))
                break

        game.make_action(choice(actions))

    all_enemy_positions.append(enemy_positions)
    print("Episode finished.")
    print("************************")

for episode_idx, positions in enumerate(all_enemy_positions):
    print(f"Episode #{episode_idx + 1} enemy positions:", positions)

game.close()
