import numpy as np

from gym import Env
from gym.spaces import Box

from carla.client import make_carla_client
from carla.planner.planner import Planner
from carla.settings import CarlaSettings
from carla.sensor import Camera
from carla import image_converter
from carla.carla_server_pb2 import Control
import cv2
import time


class CarlaEnv(Env):
    def __init__(self, city_name="Town01", ep_length=400, z_size=512):
        self.z_size = z_size
        self.HEIGHT = 160
        self.WIDTH = 80
        self.NUM_CHANNEL = 3

        # self.action_space = Box(low=np.array([-1.0]), high=np.array([-1.0]), dtype=np.float32)  # Throttle, steer, break
        # self.observation_space = Box(low=0, high=255, shape=(self.HEIGHT, self.WIDTH, self.NUM_CHANNEL), dtype=np.int32)

        self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
        self.observation_space = Box(low=np.finfo(np.float32).min,
                                     high=np.finfo(np.float32).max,
                                     shape=(1, self.z_size), dtype=np.float32)
        self.ep_length = ep_length
        self.current_step = 0
        self.dim = self.observation_space.shape  # The shape of the input on which we are learning

        # Carla Settings
        self.MAX_ALLOWED_OFFROAD = 0.3

        self.carla_settings = CarlaSettings()
        self.carla_settings.set(SynchronousMode=True,
                                SendNonPlayerAgentsInfo=True,
                                NumberOfVehicles=20,
                                NumberOfPedestrians=40,
                                QualityLevel='Low',
                                )

    def client_init(self, client):

        self.client = client

        self.carla_settings = CarlaSettings()
        self.carla_settings.set(SynchronousMode=True,
                                SendNonPlayerAgentsInfo=True,
                                NumberOfVehicles=20,
                                NumberOfPedestrians=40,
                                QualityLevel='Low',
                                )

        camera0 = Camera('CameraRGB')
        camera0.set_image_size(self.HEIGHT, self.WIDTH)
        camera0.set_position(2.0, 0.0, 1.4)
        camera0.set_rotation(0.0, 0.0, 0.0)
        camera1 = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
        camera1.set_image_size(self.HEIGHT, self.WIDTH)
        camera1.set_position(2.0, 0, 1.4)

        self.settings = CarlaSettings()
        self.settings.add_sensor(camera0)
        self.settings.add_sensor(camera1)

        scene = self.client.load_settings(self.settings)
        start_spots = scene.player_start_spots
        self.target = start_spots[26]
        self.player_start = 26
        cv2.namedWindow('im', cv2.WINDOW_NORMAL)

    def reset(self):
        self.current_step = 0
        while True:
            '''Sometimes the client looses connection to the server. So we retry connection until connected'''
            try:
                self.client.start_episode(self.player_start)
                observation, done = self._get_observation()
                break
            except:
                print('could not connect')
                self.client.disconnect()
                time.sleep(5)
                self.client.connect()
                time.sleep(10)

        return observation

    def step(self, action):
        self.current_step += 1
        self.action = action
        control = Control()
        control.throttle = 0.4  # action[0]
        control.steer = action[0]
        control.brake = 0  # action[2]
        try:
            self.client.send_control(control)
            observation, done = self._get_observation()

        except:
            print('Lost Connection')
            self.reset()
            self.client.send_control(control)
            observation, done = self._get_observation()
        reward = self._get_reward(done=done)
        info = {}
        #print("reward: ", reward)
        #print(observation, reward, done, info)

        observation = np.random.random((1, 512))
        reward = np.random.random()
        done = False
        info = {}

        return observation, reward, done, info

    def _get_observation(self):
        measurements, snesor_data = self.client.read_data()
        carla_im = snesor_data.get('CameraSegmentation', None)
        '''For RGB image'''
        #self.observation_image = image_converter.to_rgb_array(carla_im)

        '''For Semantic Segmentation Image'''
        #print(type(carla_im))
        self.observation_image = np.uint8(image_converter.labels_to_cityscapes_palette(carla_im))
        cv2.imshow('im', self.observation_image)
        cv2.waitKey(1)
        stats = measurements.player_measurements

        pos = measurements.player_measurements.transform
        movement = self.get_movemet(pos)
        # print('Movemene ', movement)

        self.percentage_intersection_other_lane = abs(stats.intersection_otherlane)
        self.percentage_offroad = stats.intersection_offroad

        collision = stats.collision_vehicles or stats.collision_pedestrians or stats.collision_other

        '''Sometimes, the collision sensor doesn't register, so I check for distnce moved between each step, after 100
        timesteps because at lower time steps the distance moved is pretty low'''
        car_stuck = self.is_car_stuck(movement)

        done = self._is_game_over(collision, car_stuck)
        #print(type(done))

        observation = self.vae_observation(observation_image=self.observation_image)
        return observation, done

    def vae_observation(self, observation_image):
        """Appending for the vae buffer here. vae.optimize() uses this buffer to train """
        self.vae.buffer_append(observation_image)
        ob = self.vae.encode(observation_image)
        # print(ob)
        return ob

    def get_movemet(self, pos):
        if self.current_step == 0:
            self.old_pos = pos
            return 1e-7
        else:
            dist_moved = np.sqrt(
                (self.old_pos.location.x - pos.location.x) ** 2 + (self.old_pos.location.y - pos.location.y) ** 2)
            self.old_pos = pos

            return dist_moved

    def is_car_stuck(self, dist_moved):
        if dist_moved < 0.01 and self.current_step > 100:
            return True
        else:
            return False

    def _is_game_over(self, collision, car_stuck):
        ''' Game is over after a collision '''
        if collision or car_stuck:
            return True
        else:
            return False

    def _get_reward(self, done):
        ''' Here we try to follow the road divider, so ideally the lane intersection should be 50% '''
        if done:
            return 0.0
        if self.percentage_offroad > self.MAX_ALLOWED_OFFROAD:
            return 0.0
        return 1 - abs(self.action)

    def set_vae(self, vae):
        self.vae = vae

    def render(self, mode='human'):
        cv2.imshow('im', self.observation_image)
        cv2.waitKey(1)