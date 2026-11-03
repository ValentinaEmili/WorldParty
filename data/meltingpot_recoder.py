# log data: raw observations, actions, player pose, and full environment state

import numpy as np
from meltingpot import substrate
from action_party.scripts.dataset.create_all_games_dataset import get_world_frame, get_positions, get_orientations

name = "clean_up"
config = substrate.get_config(name)
environment = substrate.build(name, roles=config.default_player_roles)
actions = config.action_set

num_players = len(config.default_player_roles)
player_states = []

# from MeltingPot
try: 
    timestep = environment.reset()
    observation = timestep.observation
    for i in range(num_players):
        position = observation[i]['POSITION']
        orientation = observation[i]['ORIENTATION']
        player_states.append({
            'player_id': i,
            'position': np.asarray(position, dtype=np.int32),
            'orientation': int(orientation)
        })

    world_rgb = observation[0]['WORLD.RGB'] 
finally:
    environment.close()


# from Action Party'
# try:
#     timestep = environment.reset()
#     observation = timestep.observation
#     positions = get_positions(observation, num_players)
#     orientations = get_orientations(observation, num_players)
    
#     for i in range(num_players):
#         player_states.append({
#             'player_id': i,
#             'position': [int(positions[i][0]), int(positions[i][1])],
#             'orientation': np.array(orientations[i], dtype=np.int64)
#         })

#     world_rgb = get_world_frame(observation)
# finally:
#     environment.close() 


