import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from continuous_data_collection import Pusher
from continuous_data_collection import save_video
import torch
import pickle
import pygame
import random
from copy import deepcopy
import argparse
import pymunk
import pymunk.constraints
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
from pymunk.autogeometry import march_soft, march_hard
from IPython import embed
import cv2  



square_pts = [(40, 40), (40, -40), (-40, -40), (-40, 40)]
trapezoid_pts = [(20, 20), (40, -40), (-40, -40), (-20, 20)]
T_pts = [(40, 40), (40, 20), (10, 20), (10, -40), (-10, -40), (-10, 20), (-40, 20), (-40, 40)]

L_img = [
                "        ",
                "  xx    ",
                "  xx    ",
                "  xx    ",
                "  xx    ",
                "  xx    ",
                "  xxxxx ",
                "  xxxxx ",
                "        "
            ]

T_img = [
                "         ",
                "  xxxxxx ",
                "  xxxxxx ",
                "    xx   ",
                "    xx   ",
                "    xx   ",
                "    xx   ",
                "    xx   ",
                "         "
            ]

E_img = [
                "        ",
                " xxxxxx ",
                " xxxxxx ",
                " xx     ",
                " xx     ",
                " xxxxxx ",
                " xxxxxx ",
                " xx     ",
                " xx     ",
                " xxxxxx ",
                " xxxxxx ",
                "        "
            ]

X_img = [
                "           ",
                " xxx   xxx ",
                "  xxx xxx  ",
                "   xxxxx   ",
                "    xxx    ",
                "   xxxxx   ",
                "  xxx xxx  ",
                " xxx   xxx ",
                "           "
                            ]

OPTIONS = {
    'screen_size': (600, 600),
    'damping': 0.01,
    'dt': 1.0/60.0,
    'boundary_pts': [(10, 10), (590, 10), (590, 590), (10, 590)],
    'block_pts': T_pts,
    'block_img': E_img,
    'block_img_scale': 10,
    'block_img_flag': True, 
    'block_mass': 10,
    'block_start_pos': (300, 300),
    'pusher_start_pos': (300, 220),
    'target_start_pos': (200, 400),
    'target_start_angle': np.pi/6,
    'pusher_mass': 10,
    'pusher_radius': 10,
    'elasticity': 0.0,
    'friction': 1.0,
    'block_color': (254, 33, 139, 255),
    'pusher_color': (33, 176, 254, 255.),
    'target_color': (254, 215, 0, 0.),
    'controller_stiffness': 10000,
    'controller_damping': 1000,
    'march_fn': march_soft
}


def create_video_from_actions(input_filename, output_filename,init_state):
    recorded_actions = input_filename

    pusher = Pusher(OPTIONS)
    pusher.push_body.position = Vec2d(init_state[0], init_state[1])
    pusher.block_body.position = Vec2d(init_state[2], init_state[3])
    pusher.block_body.angle = init_state[4]
    pusher.key_body.position = Vec2d(float(init_state[0]), float(init_state[1]))
    pusher.step([0.0, 0.0])
    pusher.render()


    rollout_states = []
    video_frames = []

    for action in recorded_actions:

        state_list = [
            # Pusher state
            pusher.push_body.position.x,
            pusher.push_body.position.y,
            # Block state
            pusher.block_body.position.x,
            pusher.block_body.position.y,
            pusher.block_body.angle,
        ]
        rollout_states.append(deepcopy(state_list))

        pusher.render()
        frame = pygame.surfarray.array3d(pusher.screen)
        video_frames.append(frame.copy())


        pusher.step(action)

    save_video(video_frames, 'rollout_actions.mp4')
    
    return video_frames



k = 17  # take any random video out of [0,50)

states = torch.load("states.pth")
actions = torch.load("rel_actions.pth")

print(states.shape,actions.shape)
with open('seq_lengths.pkl', 'rb') as f:
    seq_len = pickle.load(f)
print(seq_len)
state = states[k][:seq_len[k],:]
action = actions[k][:seq_len[k],:]

video_frames_action_rollout = create_video_from_actions(action,"",state[0])



video_obses_path = "obses/"+f"states{k+1}"+".mp4"

def combine_videos_side_by_side(video1_path, video2_path, output_path):
    """Combine three videos side by side into a single video"""
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    # cap3 = cv2.VideoCapture(video3_path)
    
    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error: Could not open one or more videos")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        
        if not (ret1 and ret2):
            break
            
        combined_frame = np.hstack((frame1, frame2))
        
        out.write(combined_frame)
    
    cap1.release()
    cap2.release()
    out.release()

combine_videos_side_by_side("rollout_actions.mp4", video_obses_path,"comparison_video.mp4")