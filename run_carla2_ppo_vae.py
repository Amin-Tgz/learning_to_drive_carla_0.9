from ppo_with_vae import make_carla_env
from stable_baselines.ppo2.ppo2 import *
import sys
from ppo_with_vae import *
from carla.client import make_carla_client
from stable_baselines.common.policies import MlpPolicy
from vae.controller import VAEController
from stable_baselines import logger
import os
import numpy as np

env, host, port = make_carla_env()
vae = VAEController()
PATH_MODEL_VAE = "vae.json"
vae.load(PATH_MODEL_VAE)
logger.configure(folder='/tmp/ppo_carla2/')
PATH_MODEL_PPO2 = "carla_ppo2_with_vae"


def train():
    model = PPOWithVAE(policy=MlpPolicy, env=env, n_steps=300, nminibatches=4, verbose=1,
                       tensorboard_log='/tmp/ppo_carla2/', full_tensorboard_log=False)
    model.learn(50000, vae=vae, tb_log_name='PPO2')
    return model


if os.path.exists(PATH_MODEL_PPO2 + ".pkl"):
    print("Task: test")
    env.env_method('client_init', *[client])
    env.env_method('set_vae', *[vae])
    model = PPOWithVAE.load(PATH_MODEL_PPO2, env)
    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs = env.step(actions)[0]
        env.render()
else:
    with make_carla_client(host, port) as client:
        env.env_method('client_init', *[client])
        env.env_method('set_vae', *[vae])
        model = train()
        model.save(PATH_MODEL_PPO2)
        vae.save(PATH_MODEL_VAE)
