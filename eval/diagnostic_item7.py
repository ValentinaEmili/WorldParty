"""Diagnostic renderer, checklist item 7.

Diagnostic notebook/script that renders:

    RGB frame | semantic map | players/actions | changed tiles

This script reads a rollout written by ``data/meltingpot_recorder_item4.py``
and writes one four-panel PNG. The semantic map is built directly from the 
``GLOBAL.TEXT`` grid.

Examples
--------

    # one timestep
    ./venv/bin/python eval/diagnostic_item7.py \
        --rollout examples/clean_up/data/clean_up_item4.npz --step 25 --output examples/clean_up/eval/item7.png

    # several timesteps stacked vertically
    ./venv/bin/python eval/diagnostic_item7.py \
        --rollout examples/clean_up/data/clean_up_item4.npz --steps 0 10 20 30 \
        --output examples/clean_up/eval/item7_sheet.png

    # animated version of every timestep
    ./venv/bin/python eval/diagnostic_item7.py \
        --rollout examples/clean_up/data/clean_up_item4.npz --gif examples/clean_up/eval/item7.gif

Player colours
--------------
Panels 3 and 4 use the avatar colours inferred from ``world_rgb``, so they
match the players in panel 1.  ``PLAYER_COLORS`` is only a fallback.

Score line
----------
The footer shows cumulative ``rewards`` through the displayed step. For Coins,
match/mismatch counts are reconstructed from each step's reward pattern:
a match is ``(+1, 0)`` and a mismatch is ``(+1, -2)``. For Clean Up, the score
is the number of apples eaten; ``FIRE_ZAP`` and ``FIRE_CLEAN`` are shown but
give 0 direct points in the default substrate configuration.

Apple overlay
-------------
In Clean Up, ``GLOBAL.TEXT`` uses ``A`` for both avatars and apples.  An ``A``
tile occupied by a player is the avatar; an unoccupied ``A`` tile is an apple.
Apples are marked with a filled red circle in panels 2 and 3, keeping the
underlying tile colour; their coordinates are listed in the footer.  Coins uses
``c`` for coins, so this only applies to Clean Up.

Coordinate convention
---------------------
``POSITION`` is ``(column, row)``; ``GLOBAL.TEXT`` rows are printed top to
bottom, so a cell is ``rows[row][column]``.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PANEL = 320
HEADER = 24
FOOTER_LINE = 16
BG = (18, 18, 18)
TEXT = (235, 235, 235)
DIM_TEXT = (190, 190, 190)

# ``GLOBAL.TEXT`` is a character per tile
# the characters cover the Clean Up and Coins rollouts
CHAR_STYLE = {
    "W": ("wall", (70, 70, 70)),
    "S": ("sand", (222, 205, 135)),
    "G": ("grass", (106, 168, 79)),
    "w": ("water", (58, 120, 190)),
    "D": ("dirt", (120, 72, 40)),
    "B": ("effect", (120, 220, 220)),
    " ": ("floor", (205, 205, 205)),
    "A": ("agent/object", (255, 215, 0)),
}
# anything unknown is grey
UNKNOWN_STYLE = ("other", (130, 130, 130))

# Fallback only: used when ``world_rgb`` is missing or a player is never visible
PLAYER_COLORS = (
    (125, 50, 200),   # purple
    (205, 5, 165),    # pink
    (245, 65, 65),    # red
    (245, 130, 0),    # orange
    (195, 180, 0),    # yellow
    (125, 185, 65),   # green
    (35, 185, 175),   # cyan
)

AVATAR_MIN_PIXELS = 10
AVATAR_MIN_SATURATION = 45

# 0=N, 1=E, 2=S, 3=W.  -1 means unknown.
ARROWS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0), -1: (0, 0)}

CHANGED_COLOR = (255, 240, 0) # yellow

# Apple body colour from the Clean Up sprite palette.
APPLE_COLOR = (212, 80, 57)

# LOADING

def load_rollout(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}

def pick(data: dict, *names: str) -> np.ndarray:
    """Return the first array present under any of ``names``."""
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"none of {names} found; available keys: {sorted(data)}")

def decode_text(value) -> str:
    """Return a ``GLOBAL.TEXT`` value as a Python string."""
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

def split_rows(text: str) -> list[str]:
    """Split ``GLOBAL.TEXT`` into equal-length rows."""
    rows = text.strip("\n").splitlines()
    if not rows:
        raise ValueError("GLOBAL.TEXT is empty")
    width = max(len(row) for row in rows)
    return [row.ljust(width) for row in rows]

def infer_player_colors(data: dict) -> tuple[tuple[int, int, int], ...]:
    """Read each player's avatar colour from the recorded ``world_rgb``.
    Falls back to ``PLAYER_COLORS`` when ``world_rgb`` or pose is unavailable.
    """
    world = data.get("world_rgb")       # (steps, height_px, width_px, RGB)
    texts = data.get("global_text")     
    try:
        positions = pick(data, "positions", "player_positions", "obs_POSITION") # (steps, players, 2)
    except KeyError:
        return PLAYER_COLORS
    if world is None or texts is None or world.ndim != 4:
        return PLAYER_COLORS

    rows = split_rows(decode_text(texts[0]))
    grid_h, grid_w = len(rows), len(rows[0])
    tile_h = world.shape[1] // grid_h
    tile_w = world.shape[2] // grid_w
    if tile_h < 1 or tile_w < 1:
        return PLAYER_COLORS

    colors = []
    num_steps = min(len(texts), len(positions), len(world))
    for player in range(positions.shape[1]):
        votes = Counter()
        for step in range(num_steps):
            col, row = (int(value) for value in positions[step, player])
            if not (0 <= row < grid_h and 0 <= col < grid_w):
                continue
            tile = world[
                step,
                row * tile_h : (row + 1) * tile_h, # row slice
                col * tile_w : (col + 1) * tile_w, # col slice
            ].reshape(-1, 3) # (tile_h * tile_w * 3)
            counts = Counter(map(tuple, tile.tolist())) # counts how many pixels share each exact RGB color
            candidates = [
                color
                for color, count in counts.items()
                if (
                    count >= AVATAR_MIN_PIXELS
                    and max(color) - min(color) >= AVATAR_MIN_SATURATION
                )
            ]
            if candidates:
                saturated = max(candidates, key=lambda c: max(c) - min(c))
                votes[saturated] += 1

        if votes:
            colors.append(votes.most_common(1)[0][0])
        else:
            colors.append(PLAYER_COLORS[player % len(PLAYER_COLORS)])
    return tuple(colors)

# Semantic map

def semantic_rgb(rows: list[str]) -> np.ndarray:
    """Turn the character grid into an ``[H, W, 3]`` colour image."""
    height, width = len(rows), len(rows[0])
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for row, line in enumerate(rows):
        for col, char in enumerate(line):
            rgb[row, col] = CHAR_STYLE.get(char, UNKNOWN_STYLE)[1]
    return rgb

def apple_cells(
    rows: list[str], positions: np.ndarray, game: str
) -> list[tuple[int, int]]:
    """Return Clean Up apple tiles: ``A`` cells not occupied by a player.

    ``GLOBAL.TEXT`` reuses ``A`` for avatars and apples.  A tile occupied by a
    player is the avatar; any other ``A`` tile is an apple.  Coins uses ``c``
    for coins, so this only applies to Clean Up.
    """
    if game != "clean_up":
        return []
    occupied = {
        (int(col), int(row))
        for col, row in np.asarray(positions).reshape(-1, 2)
    }
    return [
        (col, row)
        for row, line in enumerate(rows)
        for col, char in enumerate(line)
        if char == "A" and (col, row) not in occupied
    ]


def dim(rgb: np.ndarray, factor: float = 0.40) -> np.ndarray:
    """Darken a colour image so overlays stand out."""
    return np.clip(rgb.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def changed_rgb(current: list[str], previous: list[str] | None) -> np.ndarray:
    """Dimmed current map with yellow tiles that changed since ``previous``."""
    rgb = dim(semantic_rgb(current), factor=0.30)
    if previous is None:
        return rgb
    rows = min(len(current), len(previous))
    for row in range(rows):
        width = min(len(current[row]), len(previous[row]))
        for col in range(width):
            if current[row][col] != previous[row][col]:
                rgb[row, col] = CHANGED_COLOR
    return rgb

# Painting

def fit(image: Image.Image, size: int = PANEL) -> tuple[Image.Image, tuple]:
    """Scale the image to fit a square panel, preserving aspect ratio.
    It also returns ``(width, height, x_offset, y_offset)`` of the
    scaled image, so tiles can be mapped back to pixels.
    """
    width, height = image.size
    scale = min(size / width, size / height) # every single ``1 x 1`` tile in the image, is now a ``scale x scale`` square of pixels
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    scaled = image.resize((new_width, new_height), Image.NEAREST)
    offset_x = (size - new_width) // 2
    offset_y = (size - new_height) // 2
    panel = Image.new("RGB", (size, size), BG)
    panel.paste(scaled, (offset_x, offset_y)) # ``new_width x new_height`` image onto a blank 320 x 320 background
    return panel, (new_width, new_height, offset_x, offset_y)

def draw_players(
    image: Image.Image,
    box: tuple,
    rows: list[str],
    positions: np.ndarray,
    orientations: np.ndarray,
    player_colors: tuple[tuple[int, int, int], ...],
) -> None:
    """Draw a coloured square, an arrow and ``P<index>`` for each player."""
    draw = ImageDraw.Draw(image)
    width, height, offset_x, offset_y = box
    grid_h, grid_w = len(rows), len(rows[0])
    tile_w, tile_h = width / grid_w, height / grid_h # ``1 x 1`` tile has ``tile_w, tile_h`` pixels

    for player, (col, row) in enumerate(np.asarray(positions).reshape(-1, 2)):
        col, row = int(col), int(row)
        if not (0 <= row < grid_h and 0 <= col < grid_w) or rows[row][col] != 'A':
            continue
        color = player_colors[player % len(player_colors)]
        # player coords (col, row)-------------------------------> SCALED  ->
        # (col * tile_w, row * tile_h)---------------------------> SHIFTED ->
        # (offset_x + (col * tile_w), offset_y + (row * tile_h))
        x0 = offset_x + col * tile_w
        y0 = offset_y + row * tile_h
        x1 = x0 + tile_w
        y1 = y0 + tile_h
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        direction = int(orientations[player])
        dx, dy = ARROWS.get(direction, (0, 0))
        draw.line(
            [cx, cy, cx + dx * tile_w * 0.80, cy + dy * tile_h * 0.80],
            fill=color,
            width=2,
        )
        draw.text((x0 + 1, y0 + 1), f"P{player}", fill=color)

def draw_apples(
    image: Image.Image,
    box: tuple,
    rows: list[str],
    apples: list[tuple[int, int]],
) -> None:
    """Draw only a filled red circle on each apple tile, keeping its colour."""
    if not apples:
        return
    draw = ImageDraw.Draw(image)
    width, height, offset_x, offset_y = box
    grid_h, grid_w = len(rows), len(rows[0])
    tile_w, tile_h = width / grid_w, height / grid_h
    for col, row in apples:
        if not (0 <= row < grid_h and 0 <= col < grid_w):
            continue
        x0 = offset_x + col * tile_w
        y0 = offset_y + row * tile_h
        cx, cy = x0 + tile_w / 2, y0 + tile_h / 2
        radius = max(1, min(tile_w, tile_h) * 0.25)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=APPLE_COLOR,
        )


def label_panel(image: Image.Image, title: str) -> Image.Image:
    """Put a title above one square panel."""
    panel = Image.new("RGB", (PANEL, PANEL + HEADER), BG)
    panel.paste(image, (0, HEADER))
    ImageDraw.Draw(panel).text((6, 7), title, fill=TEXT)
    return panel

def action_summary(actions: np.ndarray, names: list[str]) -> str:
    """``P0=FORWARD  P1=TURN_LEFT  ...`` for one timestep."""
    parts = []
    for player, index in enumerate(np.asarray(actions).reshape(-1)):
        index = int(index)
        name = names[index] if 0 <= index < len(names) else "END"
        parts.append(f"P{player}={name}")
    return "  ".join(parts)

def action_index(names: list[str], label: str) -> int | None:
    """Return the index of ``label`` in ``names``, or None if absent."""
    try:
        return names.index(label)
    except ValueError:
        return None


def format_score(value) -> str:
    """Format a reward total compactly, e.g. ``+3`` or ``-1.5``."""
    number = float(value)
    if number.is_integer():
        return f"{int(number):+d}"
    return f"{number:+.1f}"


def score_summary(
    data: dict, game: str, names: list[str], step: int
) -> list[str] | None:
    """Return the cumulative scoreboard lines, or None if rewards are missing.

    ``rewards[t]`` is the reward received when producing state ``t``, so the
    score at step ``t`` is the sum of rewards ``0..t``. Clean Up uses several
    lines because seven players do not fit in one footer line.
    """
    rewards = data.get("rewards")
    if rewards is None:
        return None
    rewards = np.asarray(rewards) # shape of (num_timesteps, num_players)
    if rewards.ndim != 2 or not 0 <= step < len(rewards):
        return None

    totals = rewards[: step + 1].sum(axis=0)
    num_players = rewards.shape[1]
    parts: list[str] = []

    if game == "coins":
        for player in range(num_players):
            partner = 1 - player
            matches = mismatches = 0
            for t in range(step + 1):
                if rewards[t, player] > 0:
                    if rewards[t, partner] < 0:
                        mismatches += 1
                    else:
                        matches += 1
            parts.append(
                f"P{player}={format_score(totals[player])} "
                f"(match={matches}, mismatch={mismatches})"
            )
        return ["score " + "  ".join(parts)]

    if game == "clean_up":
        apples = (rewards[: step + 1] > 0).sum(axis=0)
        actions = np.asarray(data.get("actions", []))
        zap_index = action_index(names, "FIRE_ZAP")     # 7
        clean_index = action_index(names, "FIRE_CLEAN") # 8
        executed = actions[:step] if actions.ndim == 2 else actions[:0] # action being taken from the displayed frame
        if zap_index is not None and executed.size:
            zaps = (executed == zap_index).sum(axis=0)
        else:
            zaps = np.zeros(num_players, dtype=int)
        if clean_index is not None and executed.size:
            cleans = (executed == clean_index).sum(axis=0)
        else:
            cleans = np.zeros(num_players, dtype=int)
        for player in range(num_players):
            parts.append(
                f"P{player}={format_score(totals[player])} "
                f"(apple={int(apples[player])}, "
                f"zap={int(zaps[player])}, clean={int(cleans[player])})"
            )
        lines = []
        for start in range(0, len(parts), 4):
            prefix = "score " if start == 0 else "      "
            lines.append(prefix + "  ".join(parts[start : start + 4]))
        lines[-1] += "  [zap/clean = 0 pts]"
        return lines

    return [
        "score "
        + "  ".join(
            f"P{player}={format_score(totals[player])}"
            for player in range(num_players)
        )
    ]


def legend(rows: list[str], game: str = "") -> str:
    """Explain only the characters that actually occur in this frame."""
    chars = sorted({char for row in rows for char in row})
    parts = [
        f"{char}={CHAR_STYLE.get(char, UNKNOWN_STYLE)[0]}" for char in chars
    ]
    if game == "clean_up" and "A" in chars:
        parts = [
            "A=avatar(gold)/apple(red)" if part.startswith("A=") else part
            for part in parts
        ]
    return "  ".join(parts)

# RENDERING

def render_step(
    data: dict,
    game: str,
    names: list[str],
    step: int,
    player_colors: tuple[tuple[int, int, int], ...],
) -> Image.Image:
    """Render the four-panel row for one timestep."""
    texts = data["global_text"]
    world = data["world_rgb"]
    actions = data["actions"]
    positions = pick(data, "positions", "player_positions", "obs_POSITION")
    orientations = pick(data, "orientations", "player_orientations", "obs_ORIENTATION")

    step = int(step)
    if not 0 <= step < len(actions):
        raise ValueError(
            f"step {step} out of range: this rollout has {len(actions)} "
            f"timesteps, so valid steps are 0..{len(actions) - 1}"
        )

    rows = split_rows(decode_text(texts[step]))
    previous = split_rows(decode_text(texts[step - 1])) if step > 0 else None
    semantic = semantic_rgb(rows)

    apples = apple_cells(rows, positions[step], game)

    rgb_panel, _ = fit(Image.fromarray(np.asarray(world[step], dtype=np.uint8)))

    semantic_panel, semantic_box = fit(Image.fromarray(semantic))
    draw_apples(semantic_panel, semantic_box, rows, apples)

    players_panel, players_box = fit(Image.fromarray(dim(semantic, 0.45)))
    draw_players(
        players_panel,
        players_box,
        rows,
        positions[step],
        orientations[step],
        player_colors,
    )
    draw_apples(players_panel, players_box, rows, apples)

    changed_panel, changed_box = fit(Image.fromarray(changed_rgb(rows, previous)))
    draw_players(
        changed_panel,
        changed_box,
        rows,
        positions[step],
        orientations[step],
        player_colors,
    )

    changed_title = (
        "4. Changed tiles (t-1 -> t)" if step > 0
        else "4. Changed tiles (first frame)"
    )
    panels = [
        label_panel(rgb_panel, f"1. RGB frame (t={step})"),
        label_panel(semantic_panel, "2. Semantic map (GLOBAL.TEXT)"),
        label_panel(players_panel, "3. Players / actions"),
        label_panel(changed_panel, changed_title),
    ]

    score = score_summary(data, game, names, step)
    footer_lines = [
        f"step {step}  game {game}  actions: {action_summary(actions[step], names)}",
    ]
    if score:
        footer_lines.extend(score)
    if game == "clean_up":
        locations = ", ".join(f"({col},{row})" for col, row in apples[:10])
        if len(apples) > 10:
            locations += f", +{len(apples) - 10} more"
        footer_lines.append(f"apples={len(apples)} @ {locations or 'none'}")
    footer_lines.append(f"legend: {legend(rows, game)}")
    footer_height = FOOTER_LINE * len(footer_lines) + 4

    row_image = Image.new(
        "RGB", (PANEL * len(panels), PANEL + HEADER + footer_height), BG
    )
    for index, panel in enumerate(panels):
        row_image.paste(panel, (index * PANEL, 0))

    draw = ImageDraw.Draw(row_image)
    y = PANEL + HEADER + 2
    for line in footer_lines:
        draw.text((6, y), line, fill=DIM_TEXT)
        y += FOOTER_LINE

    return row_image

def contact_sheet(rows: list[Image.Image]) -> Image.Image:
    """Stack four-panel rows vertically."""
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    sheet = Image.new("RGB", (width, height), BG)
    y = 0
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet

def write_gif(
    data: dict,
    game: str,
    names: list[str],
    path: Path,
    player_colors: tuple[tuple[int, int, int], ...],
) -> None:
    import imageio.v2 as imageio

    # Stream frames instead of keeping them all in memory: a 1000-step rollout
    # would otherwise hold more than 1 GB of RGB arrays.
    with imageio.get_writer(path, duration=0.0000001, loop=0) as writer:
        for step in range(len(data["actions"])):
            writer.append_data(
                np.asarray(render_step(data, game, names, step, player_colors))
            )

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=None,
        help="Render a contact sheet for these steps instead of a single frame.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gif", type=Path, default=None)
    args = parser.parse_args()

    data = load_rollout(args.rollout)
    names = [str(name) for name in data.get("action_names", [])]
    game = str(data["game"].item()) if "game" in data else args.rollout.stem
    player_colors = infer_player_colors(data)

    try:
        if args.steps:
            image = contact_sheet(
                [
                    render_step(data, game, names, step, player_colors)
                    for step in args.steps
                ]
            )
        else:
            image = render_step(data, game, names, args.step, player_colors)
    except ValueError as exc:
        parser.error(str(exc))

    output = args.output or args.rollout.with_suffix(".item7.png")
    image.save(output)
    print(f"Player colours: {player_colors}")
    print(f"Saved {output} ({image.width}x{image.height})")

    if args.gif:
        write_gif(data, game, names, args.gif, player_colors)
        print(f"Saved {args.gif}")


if __name__ == "__main__":
    main()
