import glob
import os
import sys
import carla
import logging
import random
import numpy as np
from gym import Env
from gym.spaces import Box
import cv2
from carla import ColorConverter as cc
import weakref

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
    import queue
except ImportError:
    import Queue as queue


def draw_image(surface, image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    surface.blit(image_surface, (0, 0))


def get_cv_image(image):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    # array = array[:, :, ::-1]
    return array


def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False


class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.num_collisions = 0
        self._history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.num_collisions += 1
        # print('Collision with %r, id = %d' % (actor_type, event.other_actor.id))
        # impulse = event.normal_impulse
        # intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        # self._history.append((event.frame_number, intensity))
        # if len(self._history) > 4000:
        #    self._history.pop(0)

    def destroy(self):
        self.sensor.destroy()


class LaneInvasionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.num_laneintersections = 0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        # TODO : Handle case of lane invasion for dashed vs solid lane markings
        self.num_laneintersections += 1
        # text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        # self._hud.notification('Crossed line %s' % ' and '.join(text))

    def destroy(self):
        self.sensor.destroy()


class CarlaEnv(Env):
    def __init__(self, ep_len=400, z_size=512):
        self.z_size = z_size
        self.ep_len = ep_len
        self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
        self.observation_space = Box(low=np.finfo(np.float32).min,
                                     high=np.finfo(np.float32).max,
                                     shape=(1, self.z_size), dtype=np.float32)

        self.current_step = 0
        self.MAX_ALLOWD_OFFROAD = 0.3
        self.VAE_WIDTH = 160
        self.VAE_HEIGHT = 80
        # DO CARLA STUFF
        self.actor_list = []
        self.image_queue = None
        self.frame = None

        cv2.namedWindow('im', cv2.WINDOW_NORMAL)

        pygame.init()

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()

        while True:
            try:
                client = carla.Client('localhost', 2000)
                client.set_timeout(2.0)
                self.world = client.get_world()
                break
            except:
                print("could not connect. Trying again")

    def _get_actors(self):
        m = self.world.get_map()
        start_pose = random.choice(m.get_spawn_points())

        blue_print_library = self.world.get_blueprint_library()

        vehicle = self.world.spawn_actor(random.choice(blue_print_library.filter('vehicle*')),
                                         start_pose)
        vehicle.set_simulate_physics(True)

        camera = self.world.spawn_actor(blue_print_library.find('sensor.camera.semantic_segmentation'),
                                        carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=15)),
                                        attach_to=vehicle)
        self.actor_list.append(vehicle)
        self.actor_list.append(camera)
        collision_sensor = CollisionSensor(vehicle)
        lane_invasion_sensor = LaneInvasionSensor(vehicle)
        self.actor_list.append(collision_sensor)
        self.actor_list.append(lane_invasion_sensor)

    def _attach_image_queue_to_camera(self):
        self.image_queue = queue.Queue()
        camera = self.actor_list[1]
        camera.listen(self.image_queue.put)

    def _destroy_actors(self):
        for a in self.actor_list:
            a.destroy()
        del self.image_queue
        self.actor_list = []

    def _reset(self):
        if len(self.actor_list) > 0:
            self._destroy_actors()
        self._get_actors()
        self._attach_image_queue_to_camera()
        self.frame = None
        ob =  np.random.random((1, 512))
        print(ob)
        return ob

    def reset(self):
        self._reset()

    def step(self, action):
        #print(action)
        self.action = action


        # control = carla.VehicleControl()
        # control.throttle = 0.5
        # control.steer = float(action[0])
        # control.brake = 0
        #
        # self.actor_list[0].apply_control(
        #     control)  # actor_list = [vehicle, camera, collision_sensor, lane_invasion_sensor]

        # observation = self._get_observation()
        # done = self._is_game_over(collision=self.actor_list[2].num_collisions,
        #                           lane_intersection=self.actor_list[3].num_laneintersections)
        #
        # reward = self._get_reward(done=done)
        # print(reward, done, observation)
        # info = {}
        #

        observation = np.random.random((1, 512))
        reward = np.random.random()
        done = False
        info = {}
        print(observation, reward, done, info)
        return observation, reward, done, info

    def render(self, carla_image, mode='human'):
        cv2.imshow('im', self.observation_image)
        cv2.waitKey(1)
        draw_image(self.display, carla_image)
        text_surface = self.font.render('% 5d FPS' % self.clock.get_fps(), True, (255, 255, 255))
        self.display.blit(text_surface, (8, 10))
        pygame.display.flip()

    def _get_observation(self):

        self.world.tick()
        ts = self.world.wait_for_tick()
        self.clock.tick()
        # if self.frame is not None:
        #     if ts.frame_count != self.frame + 1:
        #         logging.warning('frame skip!')

        self.frame = ts.frame_count
        while True:
            caarla_image = self.image_queue.get()

            caarla_image.convert(cc.CityScapesPalette)
            self.observation_image = get_cv_image(caarla_image)
            self.observation_image = cv2.resize(self.observation_image, (self.VAE_WIDTH, self.VAE_HEIGHT))
            self.render(carla_image=caarla_image)
            encoded_observation = self.vae_observation(self.observation_image)[0]


            if caarla_image.frame_number == ts.frame_count:
                break
            # logging.warning(
            #     'wrong image time-stampstamp: frame=%d, image.frame=%d',
            #     ts.frame_count,
            #     caarla_image.frame_number)


        return encoded_observation


    def vae_observation(self, observation_image):
        self.vae.buffer_append(observation_image)
        ob = self.vae.encode(observation_image)
        return ob

    def set_vae(self, vae):
        self.vae = vae

    def _get_reward(self, done):
        if done:
            return 0.0
        if self.actor_list[3].num_laneintersections > 0:
            return 0.0
        return 1.0 - abs(self.action)

    def _is_game_over(self, collision, lane_intersection):
        if collision:
            return True
        else:
            return False
