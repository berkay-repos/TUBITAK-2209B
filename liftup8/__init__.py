from gym.envs.registration import register

register(
    id='dubinsAC-v0',
    entry_point='liftup8.envs:DubinsEnv',
)
