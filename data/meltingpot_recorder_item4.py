"""Record raw Melting Pot rollouts: checklist item 4.

Checklist item 4
----------------
    Clone and run the official Melting Pot repository; play Clean Up manually;
    log raw observations, actions, player pose, and full environment state.

What this file does
-------------------
* builds Clean Up (or Coins) through the official Melting Pot API;
* asks the simulator for ``POSITION`` / ``ORIENTATION`` (debug observations);
* asks for ``GLOBAL.TEXT``, the simulator's textual whole-map state;
* logs, for every step:

  - the raw per-player observations (``RGB``, ``READY_TO_SHOOT``, ...),
  - the action indices chosen for each player,
  - player position and orientation,
  - rewards, discounts and step types,
  - ``GLOBAL.TEXT`` and ``WORLD.RGB``.

* writes one compressed ``.npz`` plus a ``.json`` sidecar.

Output
------
Default: ``data/<game>_item4.npz`` and ``data/<game>_item4.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from meltingpot import substrate
from meltingpot.configs.substrates import clean_up as clean_up_config
from meltingpot.configs.substrates import coins as coins_config

SUPPORTED_GAMES = ("clean_up", "coins")

CONFIG_MODULES = {
    "clean_up": clean_up_config,
    "coins": coins_config,
}

def prepare_config(name: str):
    # it registers a LocationObserver in the Lua scene, which is what produces POSITION and ORIENTATION.
    CONFIG_MODULES[name]._ENABLE_DEBUG_OBSERVATIONS = True

    config = substrate.get_config(name)

    # ``unlocked`` allows to modify config in memory
    with config.unlocked():

        individual = list(config.individual_observation_names)
        for observation_name in ("POSITION", "ORIENTATION"):
            if observation_name not in individual:
                individual.append(observation_name)
        config.individual_observation_names = individual

        config.global_observation_names = list(
            config.global_observation_names
        ) + ["GLOBAL.TEXT"]

    return config

def action_names(action_set: Sequence[Mapping[str, int]]) -> list[str]:
    """Turn Melting Pot action dictionaries into readable names."""
    names = []
    for action in action_set:
        move = int(action.get("move", 0))
        turn = int(action.get("turn", 0))
        zap = int(action.get("fireZap", 0))
        clean = int(action.get("fireClean", 0))

        if zap:
            name = "FIRE_ZAP"
        elif clean:
            name = "FIRE_CLEAN"
        elif turn == -1:
            name = "TURN_LEFT"
        elif turn == 1:
            name = "TURN_RIGHT"
        elif move == 1:
            name = "FORWARD"
        elif move == 3:
            name = "BACKWARD"
        elif move == 2:
            name = "STEP_RIGHT"
        elif move == 4:
            name = "STEP_LEFT"
        else:
            name = "NOOP"
        names.append(name)
    return names

def decode_text(value) -> str:
    """Return ``GLOBAL.TEXT`` as a Python string."""
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, np.bytes_):
        value = bytes(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)

def get_positions(observation, num_players: int) -> np.ndarray:
    """Return ``[P, 2]`` raw positions. Melting Pot's convention is ``(column, row)``. Missing values are ``-1``."""
    positions = np.full((num_players, 2), -1, dtype=np.int32)
    for player in range(num_players):
        if "POSITION" in observation[player]:
            column, row = observation[player]["POSITION"]
            positions[player] = (int(column), int(row))
    return positions

def get_orientations(observation, num_players: int) -> np.ndarray:
    """Return ``[P]`` raw orientations (0=N, 1=E, 2=S, 3=W, -1=unknown)."""
    orientations = np.full(num_players, -1, dtype=np.int32)
    for player in range(num_players):
        if "ORIENTATION" not in observation[player]:
            continue
        value = observation[player]["ORIENTATION"]
        if isinstance(value, np.ndarray):
            value = value.item()
        orientations[player] = int(value)
    return orientations

def record_episode(
    environment,
    *,
    num_players: int,
    num_steps: int,
    seed: int,
    policy: str,
    num_actions: int,
) -> list[dict]:
    """Run one episode and return one record per logged timestep."""
    rng = np.random.default_rng(seed)
    records: list[dict] = []

    timestep = environment.reset()

    for step in range(num_steps):
        observation = timestep.observation

        # Raw per-player observations, excluding the two globals which are
        # stored separately below.
        raw_observations = {}
        for key in observation[0]:
            if key in ("WORLD.RGB", "GLOBAL.TEXT"):
                continue
            raw_observations[key] = np.stack(
                [np.asarray(observation[player][key]) for player in range(num_players)]
            )

        record = {
            "step_type": int(timestep.step_type),
            "discount": float(timestep.discount),
            "reward": np.asarray(timestep.reward, dtype=np.float32),
            "actions": np.full(num_players, -1, dtype=np.int32),
            "positions": get_positions(observation, num_players),
            "orientations": get_orientations(observation, num_players),
            "global_text": decode_text(observation[0]["GLOBAL.TEXT"]),
            "world_rgb": np.asarray(observation[0]["WORLD.RGB"], dtype=np.uint8),
            "observations": raw_observations,
        }
        records.append(record)

        if timestep.last() or step == num_steps - 1:
            break

        if policy == "random":
            actions = rng.integers(0, num_actions, size=num_players, dtype=np.int32)
        elif policy == "noop":
            actions = np.zeros(num_players, dtype=np.int32)
        else:
            raise ValueError(f"Unknown policy {policy!r}")

        timestep = environment.step(actions.tolist())
        records[-1]["actions"] = actions

    return records

def save_records(
    path: Path,
    *,
    game: str,
    records: Sequence[Mapping],
    num_players: int,
    actions: Sequence[str],
    seed: int,
    policy: str,
) -> tuple[Path, Path]:
    """Stack the per-step records into a ``.npz`` and a ``.json`` sidecar."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    observation_keys = sorted(records[0]["observations"])

    arrays: dict[str, np.ndarray] = {
        # One row per timestep.
        "actions": np.stack([r["actions"] for r in records]).astype(np.int32),
        "rewards": np.stack([r["reward"] for r in records]).astype(np.float32),
        "step_types": np.asarray([r["step_type"] for r in records], dtype=np.int8),
        "discounts": np.asarray([r["discount"] for r in records], dtype=np.float32),
        # Player pose, raw from the simulator.
        "positions": np.stack([r["positions"] for r in records]).astype(np.int32),
        "orientations": np.stack([r["orientations"] for r in records]).astype(np.int32),
        # Full environment state.
        "global_text": np.asarray([r["global_text"] for r in records]),
        "world_rgb": np.stack([r["world_rgb"] for r in records]).astype(np.uint8),
        # Small metadata arrays for convenience when loading the npz alone.
        "action_names": np.asarray(list(actions)),
        "game": np.asarray(game),
        "num_players": np.asarray(num_players, dtype=np.int64),
        "num_steps": np.asarray(len(records), dtype=np.int64),
    }

    # Every raw per-player observation becomes ``obs_<NAME>``.
    for key in observation_keys:
        arrays[f"obs_{key}"] = np.stack(
            [r["observations"][key] for r in records]
        )

    np.savez_compressed(path, **arrays)

    metadata = {
        "checklist_item": 4,
        "game": game,
        "num_players": num_players,
        "num_steps": len(records),
        "policy_seed": seed,
        "policy": policy,
        "action_names": list(actions),
        "observation_keys": observation_keys,
        "coordinate_convention": {
            "positions": "(column, row), exactly as returned by POSITION",
            "global_text": "one character per tile, rows separated by newlines",
        },
        "sources": {
            "observations": "timestep.observation per player, raw",
            "actions": "action index chosen for that step",
            "pose": "POSITION and ORIENTATION debug observations",
            "full_state": "GLOBAL.TEXT (whole-map text) and WORLD.RGB",
            "rewards": "timestep.reward, one value per player",
        },
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_dtypes": {key: str(value.dtype) for key, value in arrays.items()},
    }
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=SUPPORTED_GAMES, default="clean_up")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Seed for random action sampling only; the environment itself is "
            "not seeded by this value."
        ),
    )
    parser.add_argument("--policy", choices=("random", "noop"), default="random")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to data/<game>_item4.npz",
    )
    args = parser.parse_args()

    config = prepare_config(args.name)
    actions = action_names(config.action_set)
    num_players = len(config.default_player_roles)

    environment = substrate.build_from_config(
        config, roles=config.default_player_roles
    )
    try:
        records = record_episode(
            environment,
            num_players=num_players,
            num_steps=args.num_steps,
            seed=args.seed,
            policy=args.policy,
            num_actions=len(actions),
        )
    finally:
        environment.close()

    output = args.output or Path(__file__).resolve().parent / f"{args.name}_item4.npz"
    npz_path, json_path = save_records(
        output,
        game=args.name,
        records=records,
        num_players=num_players,
        actions=actions,
        seed=args.seed,
        policy=args.policy,
    )

    print(
        f"Recorded {len(records)} steps x {num_players} players "
        f"({args.name}, policy={args.policy})"
    )
    print(f"Saved: {npz_path} ({npz_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Metadata: {json_path}")
    print(f"Actions ({len(actions)}): {', '.join(actions)}")


if __name__ == "__main__":
    main()
