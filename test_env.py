import sys

sys.path.append('/home/frcvision1/Final/My_Environments/Carla-0.9.4')
sys.path.append('/home/frcvision1/Final/learning-to-drive-in-a-day-carla-0.9')
from stable_baselines.common.vec_env import DummyVecEnv
from vae.controller import VAEController
from stable_baselines import logger
import os
from ppo_with_vae import PPOWithVAE
from stable_baselines.ppo2.ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
import numpy as np


vae = VAEController()
PATH_MODEL_VAE = "vae.json"
logger.configure(folder='/tmp/ppo_carla2/')
PATH_MODEL_PPO2 = "carla_ppo2_with_vae_500_2mil"

def make_carla_env():
    """Import the package for carla Env, this packge calls the __init__ that registers the environment.Did this just to
    be consistent with gym"""
    sys.path.append('/home/frcvision1/Final/My_Environments/Carla_new')
    from env3 import CarlaEnv
    env = CarlaEnv()
    env = DummyVecEnv([lambda: env])
    return env

env = make_carla_env()

def train():
    model = PPOWithVAE(policy=MlpPolicy, env=env, n_steps=500, nminibatches=4, verbose=1,
                       tensorboard_log='/tmp/ppo_carla2/', full_tensorboard_log=False)
    model.learn(2000000, vae=vae, tb_log_name='PPO2')
    return model


if os.path.exists(PATH_MODEL_PPO2 + ".pkl"):
    print("Task: test")
    vae.load(PATH_MODEL_VAE)
    env.env_method('set_vae', *[vae])
    model = PPOWithVAE.load(PATH_MODEL_PPO2, env)
    obs = np.zeros((env.num_envs,) + env.observation_space.shape)
    obs[:] = env.reset()
    while True:
        actions = model.step(obs)[0]
        obs = env.step(actions)[0]


else:

    env.env_method('set_vae', *[vae])
    model = train()
    model.save(PATH_MODEL_PPO2)
    vae.save(PATH_MODEL_VAE)
