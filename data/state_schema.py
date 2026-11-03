from meltingpot.configs.substrates.clean_up import create_prefabs, ASCII_MAP, CHAR_PREFAB_MAP

prefabs = create_prefabs()

TERRAIN = dict({
    'empty': 0,
    'wall': 1,
    'sand': 2,
    'grass': 3,
    'river': 4
})
OBJECT = dict({
    'none': 0,
    'potential_apple': 1,
    'spawn_point': 2
    })

EFFECT = dict({
    'none': 0,
    'potential_dirt': 1,
    'actual_dirt': 2
})

OCCUPANCY = dict({
    'empty': 0,
    'player': 1
})

def map_ascii_to _grid():
    rows = [list(row) for row in ASCII_MAP.strip().splitlines()]
    grid_h, grid_w= len(rows), len(rows[0])

    semantic_map = dict({
        'terrain': np.zeros((grid_h, grid_w), dtype=np.int32),
        'object': np.zeros((grid_h, grid_w), dtype=np.int32),
        'effect': np.zeros((grid_h, grid_w), dtype=np.int32),
        'occupancy': np.zeros((grid_h, grid_w), dtype=np.int32)
    })

    for i in range(grid_h):
        for j in range(grid_w):
            char = rows[i][j]
            if char not in CHAR_PREFAB_MAP:
                continue

            prefab = CHAR_PREFAB_MAP[char]

            prefab_list = []
            if isinstance(prefab, str):
                prefab_list = [prefab]
            elif isinstance(prefab, dict) and 'list' in prefab:
                prefab_list = prefab['list']

            for p in pqrefab_list:
                if p in list(TERRAIN.keys()):
                    semantic_map['terrain'][i, j] = TERRAIN[p]
                elif p in list(OBJECT.keys()):
                    semantic_map['object'][i, j] = OBJECT[p]
                elif p in list(EFFECT.keys()):
                    semantic_map['effect'][i, j] = EFFECT[p]
                elif p in list(OCCUPANCY.keys()):
                    semantic_map['occupancy'][i, j] = OCCUPANCY[p]
