import glob
import os
import sys
import carla
import logging
import random
import numpy as np
from gym import Env
from gym.spaces import Box
import weakref

import sys
sys.path.append('/home/frcvision1/Final/My_Environments/Carla-0.9.4')
sys.path.append('/home/frcvision1/Final/learning-to-drive-in-a-day-carla-0.9')
from stable_baselines.common.vec_env import DummyVecEnv

class CarlaEnv(Env):
    def __init__(self, ep_len=400, z_size=512):
        self.z_size = z_size
        self.ep_len = ep_len
        self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
        self.observation_space = Box(low=np.finfo(np.float32).min,
                                     high=np.finfo(np.float32).max,
                                     shape=(1, self.z_size), dtype=np.float32)


    def step(self, action=[0.04]):
        observation = np.random.random((1, 512))
        reward = np.random.random()
        done = False
        info = {}
        print(observation, reward, done, info)
        return observation, reward, done, info

    def reset(self):
        return np.random.random((1, 512))

    def set_vae(self, vae):
        self.vae = vae