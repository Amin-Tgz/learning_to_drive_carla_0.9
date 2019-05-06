import sys
sys.path.append('/home/frcvision1/Final/My_Environments/Carla2/alta/environment')
sys.path.append('/home/frcvision1/Final/learning-to-drive-in-a-day')
from carla_synchronous.env2 import CarlaEnv
from vae.controller import VAEController

vae = VAEController()

env = CarlaEnv()
env.set_vae(vae)

env.reset()
for i in range(0,5000):
    env.step()