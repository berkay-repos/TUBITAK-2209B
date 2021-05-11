import numpy as np


class Config:
    EPISODES = 10000  # number of episodes
    vel_mps = 50  # velocity of aircrafts
    action_time = 0.6  # action delta T
    action_size = 4  # number of discrete actions

    max_bank = 70  # maximum bank angle in deg
    sleep = 0  # sleep input of train
    n_act = 500  # action steps
    roll_rate = 70  # roll-rate in deg
    vel_max_cmd = 70  # upper velocity limit
    vel_min_cmd = 30  # lower velocity limit
    pitch_rate = 6  # in deg

    bank_hold = False  # bank holding red (on/off)
    R = 200  # radius of circle
    bank = np.arccos(1 /
                     np.sqrt((vel_mps**2 / 9.81 / R)**2 +
                             1))
    red_health = 0
    blue_health = 0

    # input dim
    window_width = 800  # pixels
    window_height = 800  # pixels
    window_z = 800  # pixels
    d_min = 25
    d_max = 150
