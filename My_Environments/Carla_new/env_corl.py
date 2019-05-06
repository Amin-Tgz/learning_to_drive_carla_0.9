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
import time
from copy import deepcopy as dc
from carla.libcarla import Transform
from carla.libcarla import Location
from carla.libcarla import Rotation

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

episode_measurements = {
    "episode_id": None,
    "num_steps": None,
    "location": None,
    "speed": None,
    "distance_to_goal": None,
    "num_collisions": 0,
    "num_laneintersections": 0,
    "static_steps": 0,
    "offlane_steps": 0
    # intersection_offroad
    # intersection_otherlane
    # next_command
}


def paths_straight_Town01():
    # paths contain list of list
    # paths = [path_1, .. , path_n]
    # path_i = [start_transform, target_transform]

    paths = [
        [
            Transform(Location(x=12.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
            Transform(Location(x=76.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
        ],
        [
            Transform(Location(x=92.11, y=316, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0)),
            Transform(Location(x=92.11, y=213, z=1.32062), Rotation(pitch=0, yaw=-90.0003, roll=0))
        ],
        [
            Transform(Location(x=324.0, y=129.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0)),
            Transform(Location(x=108.5, y=199.5, z=1.32062), Rotation(pitch=0, yaw=180, roll=0))
        ],
        [
            Transform(Location(x=102.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0)),
            Transform(Location(x=320.5, y=199.3, z=1.32062), Rotation(pitch=0, yaw=0, roll=0))
        ],
        [
            Transform(Location(x=140.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0)),
            Transform(Location(x=320.00, y=2.02002, z=1.32062), Rotation(pitch=0, yaw=-9.15527e-05, roll=0))
        ]
    ]

    return paths


def get_fixed_short_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return paths_straight_Town01()[0]


def get_fixed_long_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return paths_straight_Town01()[1]


def get_random_straight_path_Town01():
    " Returns a list of [start_transform, target_transform]"
    return random.choice(paths_straight_Town01())


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


def get_speed_from_velocity(velocity):
    speed = np.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
    return speed


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
        self.image_queue = queue.Queue()
        cv2.namedWindow('im', cv2.WINDOW_NORMAL)

        pygame.init()
        self.episode_measurements = episode_measurements
        self.prev_measurement = None

        self.display = pygame.display.set_mode(
            (800, 600),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.font = get_font()
        self.clock = pygame.time.Clock()
        self.total_reward = 0

        while True:
            try:
                client = carla.Client('localhost', 2000)
                client.set_timeout(2.0)
                self.world = client.get_world()
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                self.world.apply_settings(settings)
                break
            except:
                print("could not connect. Trying again")
        self.source_transform, self.destination_transform = get_fixed_short_straight_path_Town01()
        self._get_actors()

    def step(self, action):
        self.action = action
        if action[0] < -0.5:
            action[0] = -0.5
        elif action[0] > 0.5:
            action[0] = 0.5

        control = carla.VehicleControl()
        control.throttle = 0.5
        control.steer = float(action[0])
        print(control.steer)
        control.brake = 0

        self.episode_measurements['control_steer'] = control.steer
        self.episode_measurements['control_throttle'] = control.throttle
        self.episode_measurements['control_brake'] = control.brake
        self.episode_measurements['control_reverse'] = control.reverse
        self.episode_measurements['control_hand_brake'] = control.hand_brake

        self.episode_measurements['num_collisions'] = self.actor_list[2].num_collisions
        self.episode_measurements['num_laneintersections'] = self.actor_list[3].num_laneintersections
        self.location = self.actor_list[0].get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['speed'] = get_speed_from_velocity(velocity=self.actor_list[0].get_velocity())

        self.actor_list[0].apply_control(
            control)  # actor_list = [vehicle, camera, collision_sensor, lane_invasion_sensor]

        observation = self._get_observation()

        reward = self.compute_reward_corl(self.prev_measurement, self.episode_measurements)
        self.total_reward += reward
        self.episode_measurements['reward'] = reward
        self.episode_measurements['total_reward'] = self.total_reward

        done = self._is_game_over(collision=self.actor_list[2].num_collisions,
                                  distance=self.location.distance(self.destination_transform.location))

        self.episode_measurements['done'] = done
        self.prev_measurement = dc(self.episode_measurements)
        info = {}
        if done:
            self.reset()
        return observation, reward, done, info

    def set_vae(self, vae):
        self.vae = vae

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
            self.observation_image = dc(get_cv_image(caarla_image))

            self.observation_image = cv2.resize(self.observation_image, (self.VAE_WIDTH, self.VAE_HEIGHT))
            self.render(carla_image=caarla_image)
            encoded_observation = self.vae_observation(self.observation_image)

            if caarla_image.frame_number == ts.frame_count:
                break
                # logging.warning(
                #     'wrong image time-stampstamp: frame=%d, image.frame=%d',
                #     ts.frame_count,
                #     caarla_image.frame_number)

        return encoded_observation

    def _get_actors(self):
        m = self.world.get_map()
        self.start_pose = m.get_spawn_points()[5]

        blue_print_library = self.world.get_blueprint_library()

        vehicle = self.world.spawn_actor(blue_print_library.filter('vehicle*')[5],
                                         self.source_transform)
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
        time.sleep(2)

    def _attach_image_queue_to_camera(self):

        camera = self.actor_list[1]
        camera.listen(self.image_queue.put)

    def _destroy_actors(self):
        for _ in range(0, len(self.actor_list)):
            a = self.actor_list.pop()
            a.destroy()

    def reset(self):
        self._destroy_actors()
        self._get_actors()
        self._attach_image_queue_to_camera()
        self.frame = None
        self.source_transform, self.destination_transform = get_fixed_short_straight_path_Town01()

        self.episode_measurements['num_collisions'] = self.actor_list[2].num_collisions
        self.episode_measurements['num_laneintersections'] = self.actor_list[3].num_laneintersections
        self.location = self.actor_list[0].get_location()
        self.episode_measurements['distance_to_goal'] = self.location.distance(self.destination_transform.location)
        self.episode_measurements['speed'] = get_speed_from_velocity(velocity=self.actor_list[0].get_velocity())

        self.prev_measurement = dc(self.episode_measurements)

        return np.random.random((1, 512))
        # print("CALLED RESET")
        self._reset()  # THIS CAUSES TROUBLE WITH PPO2

    def vae_observation(self, observation_image):
        self.vae.buffer_append(observation_image)
        ob = self.vae.encode(observation_image)
        return ob

    def set_vae(self, vae):
        self.vae = vae

    def _get_reward(self, done):
        if done:
            return 0  # -10.0
        if self.actor_list[3].num_laneintersections > 0:
            return 0.0
        return 1.0 - 3 * abs(self.action)

    def _is_game_over(self, collision, distance):
        if collision:
            print(distance)
            return True
        elif distance <= 4.0:
            print(distance)
            return True
        else:
            return False

    def compute_reward_corl(self, prev, current):
        cur_dist = current["distance_to_goal"]
        prev_dist = prev["distance_to_goal"]

        # Distance travelled toward the goal in m
        #distance_reward = np.clip(prev_dist - cur_dist, -10.0, 10.0)
        distance_reward = 1/(cur_dist)**0.5
        self.episode_measurements["distance_reward"] = distance_reward

        # Change in speed (km/h)
        speed_reward = 0.05 * (current["speed"] - prev["speed"])
        self.episode_measurements["speed_reward"] = speed_reward

        # Collision damage
        collision_reward = -.00002 * (current["num_collisions"] - prev["num_collisions"])
        self.episode_measurements["collision_reward"] = collision_reward

        # New sidewalk intersection
        lane_intersection_reward = -2 * (current["num_laneintersections"] - prev["num_laneintersections"])
        self.episode_measurements["lane_intersection_reward"] = lane_intersection_reward

        reward = distance_reward + speed_reward + collision_reward + lane_intersection_reward

        # Update state variables
        if np.absolute(lane_intersection_reward) > 0:
            self.episode_measurements["offlane_steps"] += 1
        if current["speed"] == 0:
            self.episode_measurements["static_steps"] += 1
        return reward
