import torch
import pytest
# from data.meltingpot_recorder import 
from meltingpot import substrate  

def pixel_to_grid(x_pixel, y_pixel, tile_size=32):
    return x // tile_size, y // tile_size

def grid_to_pixel_bbox(col, row, tile_size=32):
    # (x_start, y_start, x_end, y_end)
    return col * tile_size, row * tile_size, (col + 1) * tile_size, (row + 1) * tile_size

def test_coordinate_trasforms():
    assert pixel_to_grid(1, 3) == (32, 96, 64, 128)
    assert grid_to_pixel_bbox(45, 100) == (1, 3)


name = "clean_up"
config = substrate.get_config(name)
env = substrate.build(name, roles=config.default_player_roles)
timestep = env.reset()
obs = timestep.observation