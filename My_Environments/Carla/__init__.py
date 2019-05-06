#!/usr/bin/env python3.5
from gym.envs.registration import register
register(
    id='CarlaEnv-v0',
    entry_point='Carla.carla_env:CarlaEnv',
)

