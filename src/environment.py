import sys, os; sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.config import *

import carla
import cv2
import numpy as np

import random
import time

class SimEnv:
    actor_list = []
    front_camera = None

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(TIMEOUT)

        self.world = self.client.load_world(MAP)
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        self.specator = self.world.get_spectator()

        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

        self.traffic_lights = self.world.get_actors().filter('traffic.traffic_light')

    def load_vehicle(self):
        self.vehicle = self.world.spawn_actor(
            self.blueprint_library.filter('model3')[0],
            random.choice(self.spawn_points)
            )
        self.actor_list.append(self.vehicle)

    def set_spectator(self):
        self.specator.set_transform(self.vehicle.get_transform())

    def load_sensor(self):
        rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        rgb_cam.set_attribute('image_size_x', f'{IMG_WIDTH}')
        rgb_cam.set_attribute('image_size_y', f'{IMG_HEIGHT}')
        rgb_cam.set_attribute('fov', f'{FOV}')

        location = carla.Location(CAM_X, CAM_Y, CAM_Z)
        rotation = carla.Rotation(CAM_PITCH, CAM_YAW, CAM_ROLL)
        transform = carla.Transform(location, rotation)
        
        self.sensor = self.world.spawn_actor(rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)

    def process_img(self, data):
        i = np.array(data.raw_data)
        i2 = i.reshape((IMG_HEIGHT, IMG_WIDTH, 4))
        i3 = i2[:, :, :3]
        self.front_camera = i3

    def read_sensor(self):
        self.sensor.listen(lambda data: self.process_img(data))
        while self.front_camera is None:
            time.sleep(0.01)

        if SHOW_PREVIEW:
            cv2.imshow('', self.front_camera)
            while True:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        return self.front_camera

    def destroy(self):
        print('Destroying actors...')
        for actor in self.actor_list:
            actor.destroy()
        print('Done.')

def main():
    s = SimEnv()
    s.load_vehicle()
    s.set_spectator()
    s.load_sensor()
    s.read_sensor()
    s.destroy()

if __name__ == "__main__":
    main()