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


class CarlaEnv(Env):
    def __init__(self, ep_len=400, z_size=512):
        self.z_size = z_size
        self.ep_len = ep_len
        self.action_space = Box(low=np.array([-0.5]), high=np.array([0.5]), dtype=np.float32)
        self.observation_space = Box(low=np.finfo(np.float32).min, high=np.finfo(np.float32).max,
                                     shape=(1, self.z_size), dtype=np.float32)

        self.current_step = 0
        self.MAX_ALLOWD_OFFROAD = 0.3

        # DO CARLA STUFF
        self.actor_list = []
        pygame.init()
        while True:
            try:
                client = carla.Client('localhost', 2000)
                client.set_timeout(2.0)
                self.world = client.get_world()
                break
            except:
                print("could not connect. Trying again")

        m = self.world.get_map()
        self.start_pos = random.choice(m.get_spawn_points())

        print('enabling synchronous mode.')
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        self.reset()
        self.clock = pygame.time.Clock()
        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = self.get_font()

        self.make_actor()

        self.actor = self.actor_list[0]
        self.camera = self.actor_list[1]
        self.image_queue = queue.Queue()
        self.camera.listen(self.image_queue.put)

    def step(self):
        control = carla.VehicleControl()
        self.clock.tick()
        self.world.tick()
        ts = self.world.wait_for_tick()
        frame = None
        if frame is not None:
            if ts.frame_count != frame + 1:
                logging.warning('frame skip!')

        frame = ts.frame_count

        while True:
            image = self.image_queue.get()
            if image.frame_number == ts.frame_count:
                break
            logging.warning(
                'wrong image time-stampstamp: frame=%d, image.frame=%d',
                ts.frame_count,
                image.frame_number)

        # waypoint = random.choice(waypoint.next(2))

        control.throttle = 0.5

        actor.apply_control(control)
        self.draw_image(self.display, image)

        text_surface = self.font.render('% 5d FPS' % self.clock.get_fps(), True, (255, 255, 255))
        self.display.blit(text_surface, (8, 10))

        pygame.display.flip()

    def make_actor(self):
        '''Appends a car and an attached camera to the actor_list'''
        m = self.world.get_map()

        blueprint_library = self.world.get_blueprint_library()

        vehicle = self.world.spawn_actor(
            random.choice(blueprint_library.filter('vehicle.*')),
            self.start_pos)
        self.actor_list.append(vehicle)

        vehicle.set_simulate_physics(True)

        camera = self.world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=2.0, y=0, z=1.4), carla.Rotation(pitch=-0)),
            attach_to=vehicle)
        self.actor_list.append(camera)

    def destroy_actors(self):
        '''Destroys all the actors when done.'''
        for actor in self.actor_list:
            actor.destroy()

    def reset(self):
        self.destroy_actors()
        self.make_actor()

    def draw_image(self, surface, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        surface.blit(image_surface, (0, 0))

    def get_font(self):
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)
