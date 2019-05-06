#!/usr/bin/env python3.5
from gym.envs.registration import register
register(
    id='CarlaEnv-0.9-v0',
    entry_point='Carla_new.env3:CarlaEnv',
)
register(
    id='CarlaEnv-0_9_corl-v0',
    entry_point='Carla_new.env_corl:CarlaEnv',
)
