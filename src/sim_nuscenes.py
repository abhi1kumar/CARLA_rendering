import json
import os
import sys
import queue
import pygame
import numpy as np
from pyquaternion import Quaternion
import weakref
from time import sleep
from PIL import Image
from time import time
from glob import glob

sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.5-py3.5-linux-x86_64.egg'))
sys.path.append(os.path.join(os.environ['CARLAPATH'], 'dist/carla-0.9.8-py3.5-linux-x86_64.egg'))
sys.path.append(os.environ['CARLAPATH'])
import carla
from pygame.locals import K_0, K_9, K_ESCAPE, K_SPACE, K_d, K_a, K_s, K_w

from .tools import (add_npcs, init_env, ClientSideBoundingBoxes,
                    weather_ps, name2weather, save_data, CAMORDER, bbox_to_2d_lim,
                    determine_filter)
from .annotator import auto_annotate


def render(cameras, display, current_ix, headless, filter_occluded):
    img_data = []
    depth_data = []
    for cam in cameras:
        image = cam['queue'].get()
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        img_data.append(array.copy())
    if filter_occluded:
        for cam in cameras:
            image = cam['depth_q'].get()
            image.convert(carla.ColorConverter.Depth)
            array = np.array(image.raw_data).reshape((image.height,image.width,4))[:,:,0] * 1000 / 255
            depth_data.append(array.copy())
    if not headless:
        surface = pygame.surfarray.make_surface(img_data[current_ix].swapaxes(0, 1))
        display.blit(surface, (0, 0))
    return img_data, depth_data


def camera_blueprint(world, width, height, VIEW_FOV, motion_blur_strength, depth_camera=False):
    """
    Returns camera blueprint.
    """
    bp_name = 'sensor.camera.depth' if depth_camera else 'sensor.camera.rgb'
    camera_bp = world.get_blueprint_library().find(bp_name)
    camera_bp.set_attribute('image_size_x', str(width))
    camera_bp.set_attribute('image_size_y', str(height))
    camera_bp.set_attribute('fov', str(VIEW_FOV))

    if motion_blur_strength is not None and not depth_camera:
        print('setting blur', motion_blur_strength)
        camera_bp.set_attribute('motion_blur_intensity', str(motion_blur_strength))
        camera_bp.set_attribute('motion_blur_max_distortion', str(motion_blur_strength))

    return camera_bp


def get_cameras(calib, world, width, height, car, motion_blur_strength, cam_adjust, filter_occluded):
    cameras = []
    for camname in CAMORDER:
        info = calib[camname]
        camera_transform = carla.Transform(carla.Location(x=info['trans'][0]+cam_adjust[camname]['x'], y=info['trans'][1], z=info['trans'][2] + cam_adjust[camname]['height']),
                                           carla.Rotation(yaw=cam_adjust[camname]['yaw'], pitch=0.0+cam_adjust[camname]['pitch'], roll=0.0))

        camera = world.spawn_actor(camera_blueprint(world, width, height, info['fov']+cam_adjust[camname]['fov'], motion_blur_strength), camera_transform, attach_to=car)
        if filter_occluded:
            depth_camera = world.spawn_actor(camera_blueprint(world, width, height, info['fov']+cam_adjust[camname]['fov'], motion_blur_strength, depth_camera=True), camera_transform, attach_to=car)

        calibration = np.identity(3)
        calibration[0, 2] = width / 2.0
        calibration[1, 2] = height / 2.0
        calibration[0, 0] = calibration[1, 1] = width / (2.0 * np.tan((info['fov']+cam_adjust[camname]['fov']) * np.pi / 360.0))
        camera.calibration = calibration

        if not filter_occluded:
            cameras.append({'cam': camera, 'queue': queue.Queue()})
        else:
            cameras.append({'cam': camera, 'queue': queue.Queue(), 'depth_cam': depth_camera, 'depth_q': queue.Queue()})

        break # because we only render the front facing camera for viewpoint robustness paper

    for v in cameras:
        v['cam'].listen(v['queue'].put)
        if filter_occluded:
            v['depth_cam'].listen(v['depth_q'].put)

    return cameras


def scrape_single(world, car_bp, ego_start, clock, display, nnpcs,
                  pos_agents, pos_inits, weather_name, pref, width, height,
                  calib, current_ix, start_ix, headless, car_color_range,
                  og_color, nnpc_position_std, nnpc_yaw_std, motion_blur_strength,
                  cam_adjust, filter_occluded):
    world.set_weather(weather_name)
    location = np.random.choice(pos_inits)
    car_bp.set_attribute('color', f'{np.random.randint(256)},{np.random.randint(256)},{np.random.randint(256)}')
    car = world.spawn_actor(car_bp, location)
    car.set_autopilot(True)
    npcs = add_npcs(world, 15, pos_agents, location, pos_inits)

    # SETUP CAMERAS
    cameras = get_cameras(calib, world, width, height, car, motion_blur_strength, cam_adjust, filter_occluded)

    def destroy():
        for cam in cameras:
            cam['cam'].destroy()
            cam['depth_cam'].destroy()
        car.destroy()
        for npc in npcs:
            npc.destroy()
    
    try:
        data = []
        for prestep in range(pref):
            # event handler
            if not headless:
                for event in pygame.event.get():
                    if event.type == pygame.KEYUP and event.key >= K_0 and event.key <= K_9:
                        current_ix = (event.key - K_0) % len(cameras)
                        print(list(calib.keys())[current_ix])
            world.tick()
            if not headless:
                clock.tick_busy_loop(20)

            img_data, depth_data = render(cameras, display, current_ix, headless, filter_occluded)

            # bounding boxes
            bboxes = ClientSideBoundingBoxes.get_global_bbox(npcs)

            # filter out occluded bounding boxes (please note it is not perfect)
            if filter_occluded:
                filtered_out, removed_out, dont_filter = auto_annotate(npcs, cameras[0]['cam'], depth_data[0])
                bboxes = [bboxes[i] for i in range(len(bboxes)) if dont_filter[i]]

            bounding_boxes = ClientSideBoundingBoxes.get_camera_boxes(bboxes, cameras[current_ix]['cam'])

            if not headless:
                ClientSideBoundingBoxes.draw_bounding_boxes(display, bounding_boxes, width, height, BB_COLOR=(248, 64, 24))
                pygame.display.flip()
                pygame.event.pump()

            if prestep >= start_ix:
                car_bboxes = np.array([
                    np.dot(np.linalg.inv(
                        ClientSideBoundingBoxes.get_matrix(car.get_transform())
                        ), bbox)
                    for bbox in bboxes
                ]).tolist()
                data.append({
                    'imgs': img_data,
                    'car_bboxes': car_bboxes,
                })

    finally:
        destroy()
    
    return current_ix, data


def scrape(host='127.0.0.1', port=2000, width=512, height=512, timeout=30.0, ntrials=2500, pref=150,
           calibf='./nusccalib.json', outf=None, start_ix=50, ncarcalib='./nuscncars.json',
           rnd_seed=42, headless=False, skipframes=10,

           map_name='Town03',  # set the map (01-05)

           fix_nnpcs=1,  # set to an integer to fix the number of npcs per scene (0 or greater int)
           uniform_nnpcs=False,  # set true to uniformly sample from the number of npcs

           p_assets=1.0,  # fraction of assets to use (float between 0 and 1)

           car_color_range=0.5,  # each color will be Unif(0.5-car_color_range, 0.5+car_color_range) (float between 0 and 0.5)
           og_color=False,  # set true to leave the color to the default

           weather_max=None,  # weather parameters will be Unif(0, weather_max*100) if set (float between 0 and 1)

           nnpc_position_std=0.0,  # (meters) npc initial position uniform noise half-width (float non-negative)

           nnpc_yaw_std=0.0,  # (degrees) npc initial heading uniform noise half-width (float non-negative)

           motion_blur_strength=None,  # determines amount of motion blur (float between 0 and 1)

           cam_yaw_adjust=0.0,  # (degrees) uniform half-width
           cam_fov_adjust=0.0,  # (degrees) uniform half-width
           cam_pitch_adjust=0.0,
           cam_height_adjust=0.0,
           cam_x_adjust=0.0,

           filter_occluded=True,
           ):
    filter_occluded=True
    np.random.seed(rnd_seed)
    pos_inits, pos_agents, world, calib, car_bp, pos_weathers, scene_names, ncarinfo, client = init_env(host, port, width, height, timeout, calibf,
                                                                                                        ncarcalib, rnd_seed, map_name, p_assets)

    if headless:
        display, clock = None, None
    else:
        pygame.init()
        display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        clock = pygame.time.Clock()

    weather_prob = [weather_ps[i] for i in range(len(pos_weathers))]
    weather_prob = np.array(weather_prob) / sum(weather_prob)
    current_ix = 0  # current viewable camera

    trial_ixes = list(range(ntrials))
    # check folders that have already been made
    if outf is not None:
        trial_fs = glob(os.path.join(outf, '*'))
        trial_ixes = [ix for ix in trial_ixes if os.path.join(outf, str(ix)) not in trial_fs]
        print('SOME FILES MIGHT ALREADY EXIST!!!!!!!!!! ntrials:', ntrials, 'num left:', len(trial_ixes))

    for trial in trial_ixes:
        t0 = time()
        init_ix = np.random.randint(len(pos_inits))

        # nnpcs
        nnpc = np.random.choice(ncarinfo[:, 0], p=ncarinfo[:, 1] / ncarinfo[:, 1].sum())

        if weather_max is not None:
            print('weather random', weather_max)
            chosen_weather = carla.WeatherParameters(cloudiness=np.random.uniform(0.0, 100.0*weather_max),
                                                     precipitation=np.random.uniform(0.0, 100.0*weather_max),
                                                     precipitation_deposits=np.random.uniform(0.0, 100.0*weather_max),
                                                     wind_intensity=np.random.uniform(0.0, 100.0*weather_max),
                                                     wetness=np.random.uniform(0.0, 100.0*weather_max),
                                                     fog_density=np.random.uniform(0.0, 100.0*weather_max),
                                                     sun_altitude_angle=np.random.uniform(-90.0, 0.0) if np.random.uniform(0.0, 1.0) < 0.1 else np.random.uniform(0.0, 90.0*(1.0 - weather_max))
                                                     )
        else:
            print('weather categorical')
            weather_ix = int(np.random.choice(list(range(len(pos_weathers))), p=weather_prob))
            chosen_weather = name2weather[pos_weathers[weather_ix]]

        calib_ix = np.random.randint(len(calib))
        cam_adjust = {cam: {'yaw': cam_yaw_adjust,
                            'fov': cam_fov_adjust,
                            'pitch': cam_pitch_adjust,
                            'height': cam_height_adjust,
                            'x': cam_x_adjust} for cam in CAMORDER}
        print('cam_adjust:', cam_adjust)

        current_ix, data = scrape_single(world, car_bp, pos_inits[init_ix], clock, display, nnpc,
                                        pos_agents, pos_inits, chosen_weather, pref, width, height,
                                        calib[calib_ix], current_ix, start_ix, headless, car_color_range,
                                        og_color, nnpc_position_std, nnpc_yaw_std, motion_blur_strength,
                                        cam_adjust, filter_occluded)

        if outf is not None:
            save_data(data, outf, trial, scene_names[calib_ix], skipframes, cam_adjust)
        t1 = time()
        print('finished episode number', trial, 'in time', t1 - t0)
    if not headless:
        pygame.quit()
