import sys
import gym
import time
import cv2
from carla.client import make_carla_client

'''Import the package for carla Env, this packge calls the __init__ that registers the environment. Did this
                just to bt cinssitent with gym'''
sys.path.append('/home/frcvision1/Final/My_Environments/')
import Carla


host = 'localhost'
port = 2000

env = gym.make(id='CarlaEnv-v0')

with make_carla_client(host, port) as client:
    env.client_init(client)
    env.reset()
    time.sleep(10)
    cv2.namedWindow('im', cv2.WINDOW_NORMAL)
    while True:
        env.reset()
        time.sleep(5)
        while True:
            control = env.action_space.sample()
            control = [0.5, 0, 0]
            ob_image, reward, done, info = env.step(control)
            print(reward, done)
            if done:
                break
            cv2.imshow('im', ob_image)
            cv2.waitKey(1)
    #env.step(env.action_space.sample())