from gym.envs.registration import register

register(
    id='dubinsAC-v0',
    entry_point='gym_dubins_airplane.envs:DubinsEnv',
)
