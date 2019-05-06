#!/usr/bin/env python3.5
# Copyright (c) 2018 Roma Sokolkov
# MIT License

import os
import gym
import numpy as np
import sys
from stable_baselines.ddpg.policies import LnMlpPolicy
from stable_baselines.ddpg.noise import OrnsteinUhlenbeckActionNoise

from ddpg_with_vae import DDPGWithVAE as DDPG
from vae.controller import VAEController
from stable_baselines import logger

'''Import the package for carla Env, this packge calls the __init__ that registers the environment. Did this
                just to bt cinssitent with gym'''
sys.path.append('/home/frcvision1/Final/My_Environments/')
import Carla_new


env = gym.make('CarlaEnv-0.9-v0')

print('Made env')
PATH_MODEL_VAE = "vae.json"
# Final filename will be PATH_MODEL_DDPG + ".pkl"
PATH_MODEL_DDPG = "ddpg_carla_segmented"

logger.configure(folder='/tmp/ddpg_carla/')

# Initialize VAE model and add it to gym environment.
# VAE does image post processing to latent vector and
# buffers raw image for future optimization.
vae = VAEController()
env.unwrapped.set_vae(vae)

# Run in test mode of trained models exist.
if os.path.exists(PATH_MODEL_DDPG + ".pkl") and \
        os.path.exists(PATH_MODEL_VAE):

    print("Task: test")
    ddpg = DDPG.load(PATH_MODEL_DDPG, env)
    vae.load(PATH_MODEL_VAE)

    obs = env.reset()
    while True:
        action, _states = ddpg.predict(obs)
        # print(action)
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()
        env.render()
# Run in training mode.
else:
    for i in range(0, 50):
        print("Task: train")
    # the noise objects for DDPG
    n_actions = env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions),
        theta=float(0.6) * np.ones(n_actions),
        sigma=float(0.2) * np.ones(n_actions)
    )

    ddpg = DDPG(LnMlpPolicy,
                    env,
                    verbose=1,
                    batch_size=32,
                    clip_norm=5e-3,
                    gamma=0.9,
                    param_noise=None,
                    action_noise=action_noise,
                    memory_limit=10000,
                    nb_train_steps=3000,
                    tensorboard_log='/tmp/ddpg_carla/',
                    full_tensorboard_log=False
                    )
    # vae.load(PATH_MODEL_VAE)
    ddpg.learn(total_timesteps=120000, vae=vae, skip_episodes=1, tb_log_name='DDPG')
    # Finally save model files.
    ddpg.save(PATH_MODEL_DDPG)
    vae.save(PATH_MODEL_VAE)
