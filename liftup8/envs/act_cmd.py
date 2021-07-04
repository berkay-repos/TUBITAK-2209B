import numpy as np


def action2command(vel, action):
    u = np.array([
        [vel, 0, 0],      # Maintain speed
        [vel + 4, 0, 0],  # Target acc
        [vel - 6, 0, 0],  # Target decc
        [vel, 0, 1],      # Right turn maintain x3
        [vel, 0, -1],     # Left turn maintain x3
        [vel, 2, 0],      # climb
        [vel, -2, 0]      # descend
    ])
    return u[action]
