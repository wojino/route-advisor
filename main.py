from src.environment import SimEnv
from src.bev import front_to_bev
import cv2


def main():
    env = SimEnv()
    env.load_vehicle()
    env.set_spectator()
    env.load_sensor()

    front = env.read_sensor()
    cv2.imwrite('out/front.png', front)

    bev = front_to_bev(front)
    cv2.imwrite('out/bev.png', bev)

    env.destroy()

if __name__ == '__main__':
    main()
