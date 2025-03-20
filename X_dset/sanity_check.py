import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from continuous_data_collection import create_video_from_actions, create_video_from_states
import torch
import pickle



k = 9  # take any random video out of [0,50)

states = torch.load("states.pth")
actions = torch.load("actions.pth")


with open('seq_len.pkl', 'rb') as f:
    seq_len = pickle.load(f)

state = states[k][:seq_len[k],:]
action = actions[k][:seq_len[k],:]

video_frames_action_rollout = create_video_from_actions(action,"",state[0])
video_frames_states = create_video_from_states(state,"")


video_obses_path = "X_dset/obses/states"+str(k+1)+".mp4"

import cv2
import numpy as np
def combine_videos_side_by_side(video1_path, video2_path, video3_path, output_path):
    """Combine three videos side by side into a single video"""
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    cap3 = cv2.VideoCapture(video3_path)
    
    if not (cap1.isOpened() and cap2.isOpened() and cap3.isOpened()):
        print("Error: Could not open one or more videos")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 3, height))
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        ret3, frame3 = cap3.read()
        
        if not (ret1 and ret2 and ret3):
            break
            
        combined_frame = np.hstack((frame1, frame2, frame3))
        
        out.write(combined_frame)
    
    cap1.release()
    cap2.release()
    cap3.release()
    out.release()

combine_videos_side_by_side("rollout_actions.mp4", "rollout_states.mp4", video_obses_path,"comparison_video.mp4")




