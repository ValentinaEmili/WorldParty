"""Coordinate-transform tests, checklist item 5.

Verify the coordinate transform between the 16x16 grid and 512x512 rendered pixels. 
The pipeline relies on three conventions:

1. ``POSITION`` observations are ``(column, row)``.
2. Every stored grid is indexed ``[row, column]``.
3. Native Melting Pot tiles are 8x8 px; the canonical 16x16 render is 512x512,
   i.e. 32x32 px per canonical tile.

It will further be applied to the pipeline to check the coordinates consistency.

Run with the repo's test environment::

    ./venv/bin/python -m pytest tests/test_coordinates.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

SPRITE_PX = 8               # Melting Pot tile size
CANONICAL_SIZE = 16         # grid size is 16 x 16 tiles
CANONICAL_IMAGE_PX = 512    # image size is 512 x 512 pixels
CANONICAL_TILE_PX = CANONICAL_IMAGE_PX // CANONICAL_SIZE  # 32 x 32

def tile_to_pixel_box(row: int, col: int, sprite_px: int = SPRITE_PX):
    """Return ``(y0, y1, x0, x1)`` for a native tile.
    Rows run along the image y axis, columns along the x axis.
    """
    return (
        row * sprite_px,
        (row + 1) * sprite_px,
        col * sprite_px,
        (col + 1) * sprite_px,
    )

def pixel_to_tile(y: int, x: int, sprite_px: int = SPRITE_PX):
    """Return (row, col) format"""
    return y // sprite_px, x // sprite_px

def canonical_tile_to_pixel_box(
    row: int,
    col: int,
    image_size: int = CANONICAL_IMAGE_PX,
    grid_size: int = CANONICAL_SIZE,
):
    """Return ``(y0, y1, x0, x1)`` for a canonical tile."""
    if image_size % grid_size != 0:
        raise ValueError(
            f"image_size {image_size} is not divisible by grid_size {grid_size}"
        )
    tile_px = image_size // grid_size
    return (
        row * tile_px,
        (row + 1) * tile_px,
        col * tile_px,
        (col + 1) * tile_px,
    )

def canonicalize_positions(
    positions,
    native_shape: tuple[int, int],
    size: int = CANONICAL_SIZE,
) -> np.ndarray:
    """Map native ``(column, row)`` positions to canonical ``(column, row)``."""
    height, width = native_shape
    positions = np.asarray(positions, dtype=np.int32).reshape(-1, 2)
    cols = np.clip(
        ((positions[:, 0] + 0.5) * size / width).astype(int), 0, size - 1
    )
    rows = np.clip(
        ((positions[:, 1] + 0.5) * size / height).astype(int), 0, size - 1
    )
    return np.stack([cols, rows], axis=1).astype(np.int32)

def occupancy_from_positions(positions, size: int = CANONICAL_SIZE):
    """Build ``[row, column]`` occupancy: 0=empty, p+1=player p."""
    occupancy = np.zeros((size, size), dtype=np.uint8)
    for player, (col, row) in enumerate(np.asarray(positions).reshape(-1, 2)):
        occupancy[int(row), int(col)] = player + 1
    return occupancy

def resample_nearest(grid: np.ndarray, size: int = CANONICAL_SIZE):
    """Nearest-neighbour resample ``[H, W, ...]`` to ``[size, size, ...]``."""
    grid = np.asarray(grid)
    height, width = grid.shape[:2]
    rows = np.clip(
        ((np.arange(size) + 0.5) * height / size).astype(int), 0, height - 1
    )
    cols = np.clip(
        ((np.arange(size) + 0.5) * width / size).astype(int), 0, width - 1
    )
    return grid[np.ix_(rows, cols)]


# --- Tests -----------------------------------------------------------------
def test_position_is_column_row_and_grid_is_row_column():
    """POSITION=(col,row) must be read as ``grid[row, col]``."""
    # 3 rows x 5 columns: a row/column swap is detectable.
    text = "WWWWW\nW  AW\nWWWWW"
    rows = text.splitlines()
    assert (len(rows), len(rows[0])) == (3, 5)

    column, row = 3, 1
    assert rows[row][column] == "A"

    occupancy = np.zeros((len(rows), len(rows[0])), dtype=np.uint8)
    for player, (col, r) in enumerate([(column, row)]):
        occupancy[r, col] = player + 1

    assert occupancy[1, 3] == 1
    assert int((occupancy > 0).sum()) == 1

    # The swapped access is invalid here, which is exactly why the convention
    # must be explicit.
    with pytest.raises(IndexError):
        _ = rows[column][row]


def test_native_tile_to_pixel_box_is_row_major_and_8px():
    """A native tile box is (y0, y1, x0, x1) with 8 px per tile."""
    assert SPRITE_PX == 8
    assert tile_to_pixel_box(row=0, col=0) == (0, 8, 0, 8)
    assert tile_to_pixel_box(row=2, col=3) == (16, 24, 24, 32)


def test_native_pixel_to_tile_round_trips_every_pixel():
    """Every pixel in a native tile maps back to that tile."""
    for row in range(3):
        for col in range(4):
            y0, y1, x0, x1 = tile_to_pixel_box(row, col)
            for y in (y0, y1 - 1):
                for x in (x0, x1 - 1):
                    assert pixel_to_tile(y, x) == (row, col)


def test_canonical_tile_to_pixel_box_is_32px_for_512_over_16():
    """512 / 16 == 32 px per canonical tile."""
    assert CANONICAL_SIZE == 16
    assert CANONICAL_IMAGE_PX == 512
    assert CANONICAL_TILE_PX == 32
    assert canonical_tile_to_pixel_box(0, 0) == (0, 32, 0, 32)
    assert canonical_tile_to_pixel_box(1, 2) == (32, 64, 64, 96)
    assert canonical_tile_to_pixel_box(15, 15) == (480, 512, 480, 512)


def test_canonical_tiles_cover_the_image_exactly_once():
    """The 16x16 canonical tiles partition the 512x512 image."""
    covered = np.zeros((CANONICAL_IMAGE_PX, CANONICAL_IMAGE_PX), dtype=np.uint8)
    for row in range(CANONICAL_SIZE):
        for col in range(CANONICAL_SIZE):
            y0, y1, x0, x1 = canonical_tile_to_pixel_box(row, col)
            covered[y0:y1, x0:x1] += 1
    assert covered.min() == 1
    assert covered.max() == 1


def test_canonical_tile_to_pixel_box_rejects_non_divisible_size():
    with pytest.raises(ValueError):
        canonical_tile_to_pixel_box(0, 0, image_size=500, grid_size=16)


def test_canonicalize_positions_maps_native_extremes_to_canonical_corners():
    """A native 21x30 grid must map its corners onto the 16x16 corners."""
    native_shape = (21, 30)  # Clean Up: rows x columns
    positions = np.asarray([[0, 0], [29, 20]], dtype=np.int32)  # (col, row)
    canonical = canonicalize_positions(positions, native_shape)

    assert canonical[0].tolist() == [0, 0]
    assert canonical[1].tolist() == [15, 15]
    assert canonical.shape == (2, 2)


def test_canonicalize_positions_preserves_orientation_of_both_axes():
    """Increasing native column/row increases canonical column/row."""
    native_shape = (21, 30)

    positions = np.asarray([[0, 0], [10, 0], [20, 0], [29, 0]], dtype=np.int32)
    canonical = canonicalize_positions(positions, native_shape)
    assert canonical[:, 0].tolist() == sorted(canonical[:, 0].tolist())
    assert canonical[:, 0].tolist() == [0, 5, 10, 15]

    positions = np.asarray([[0, 0], [0, 10], [0, 20]], dtype=np.int32)
    canonical = canonicalize_positions(positions, native_shape)
    assert canonical[:, 1].tolist() == [0, 8, 15]


def test_canonical_occupancy_is_built_from_transformed_positions():
    """Two players at distinct native tiles stay distinct canonically."""
    native_shape = (21, 30)
    positions = np.asarray([[0, 0], [29, 20]], dtype=np.int32)
    canonical = canonicalize_positions(positions, native_shape)
    occupancy = occupancy_from_positions(canonical)

    assert occupancy[0, 0] == 1
    assert occupancy[15, 15] == 2
    assert int((occupancy > 0).sum()) == 2


def test_native_and_canonical_render_sizes_are_documented_and_consistent():
    """Native 21x30 -> 168x240 px; canonical 16x16 -> 512x512 px."""
    native_shape = (21, 30)
    native_height = native_shape[0] * SPRITE_PX
    native_width = native_shape[1] * SPRITE_PX
    assert (native_height, native_width) == (168, 240)

    canonical_px = CANONICAL_SIZE * (CANONICAL_IMAGE_PX // CANONICAL_SIZE)
    assert canonical_px == 512


def test_resample_nearest_keeps_row_column_order():
    """Resampling a row-ramp must keep the ramp along axis 0."""
    grid = np.repeat(np.arange(16, dtype=np.uint8)[:, None], 16, axis=1)
    resampled = resample_nearest(grid, size=8)

    assert resampled.shape == (8, 8)
    assert resampled[:, 0].tolist() == sorted(resampled[:, 0].tolist())
    assert resampled[0, 0] < resampled[-1, 0]
    assert np.all(resampled == resampled[:, :1])
