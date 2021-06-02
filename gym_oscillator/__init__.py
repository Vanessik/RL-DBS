from gym.envs.registration import register

register(
    id='oscillator-v1',
    entry_point='gym_oscillator.envs:oscillatorEnv',
)