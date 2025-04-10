"""This example showcase an arrow pointing or aiming towards the cursor.
"""

__docformat__ = "reStructuredText"

import sys
import os
import pygame

from copy import deepcopy
import random
import pymunk
import pymunk.constraints
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
from pymunk.autogeometry import march_soft, march_hard
from IPython import embed
import sys
from pymunk import Vec2d
import numpy as np
import cv2
import torch  
from copy import deepcopy
import shapely.geometry as sg
from shapely.ops import unary_union, polygonize
import argparse

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

parser = argparse.ArgumentParser(description='Pushing simulation with configurable block body shape')
parser.add_argument('--shape', type=str, default='X',
                    choices=['L', 'T', 'E', 'X'],
                    help='Block body shape to use (L, T, E, X)')
parser.add_argument('--target', type=str, default='Y',
                    choices=['Y','N'],
                    help='Y target body is shown, N target body is not shown')
parser.add_argument('--skip_zero_action', type=str, default='Y',
                    choices=['Y','N'],
                    help="Y don't record the zero ([0.,0.]) actions , N record the zero actions")
args = parser.parse_args()

# Map argument to corresponding image
shape_mapping = {
    'L': L_img,
    'T': T_img,
    'E': E_img,
    'X': X_img
}


OPTIONS = {
    'screen_size': (600, 600),
    'damping': 0.01,
    'dt': 1.0/60.0,
    'boundary_pts': [(10, 10), (590, 10), (590, 590), (10, 590)],
    'block_pts': T_pts,
    'block_img': shape_mapping[args.shape],
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
    'target_color': (254, 215, 0, 0.) if args.target == 'Y' else (255, 255, 255, 0.),
    'controller_stiffness': 10000,
    'controller_damping': 1000,
    'march_fn': march_soft
}

def pymunk_to_shapely(body, shapes):
    # Store converted polygons and segments separately
    polygons = []
    segment_lines = []
    
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts.append(verts[0])  # Ensure the polygon is closed
            polygons.append(sg.Polygon(verts))
        elif isinstance(shape, pymunk.shapes.Segment):
            a = body.local_to_world(shape.a)
            b = body.local_to_world(shape.b)
            segment_lines.append(sg.LineString([a, b]))
        else:
            raise RuntimeError(f"Unsupported shape type {type(shape)}")
    poly_from_segments = list(polygonize(segment_lines))
    all_polys = polygons + poly_from_segments
    
    if not all_polys:
        raise RuntimeError("No valid polygon could be created from the shapes")

    if len(all_polys) == 1:
        return all_polys[0]

    return unary_union(all_polys)

class Pusher:
    def __init__(self, options = OPTIONS):
        self.options = options
        pygame.init()
        os.environ['SDL_VIDEODRIVER'] = 'dummy' 
        self.screen = pygame.display.set_mode(self.options['screen_size'])
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = self.draw_options.DRAW_SHAPES | self.draw_options.DRAW_COLLISION_POINTS
        self.draw_options.draw_shape = self._shapely_draw_shape
    
        self.space = pymunk.Space()
        self.space.damping = self.options['damping']

        self.IMG_FLAG = self.options['block_img_flag']

        self._setup_env()
        
        if self.IMG_FLAG:
            self._setup_block_img()
            self._setup_target_img()
        else:
            # self._setup_block()
            self._setup_block_lines()
            self._setup_target()
        
        self._setup_pusher()

    def _setup_env(self):
        pts = self.options['boundary_pts']
        for i in range(len(pts)):
            seg = pymunk.Segment(self.space.static_body, pts[i], pts[(i+1)%len(pts)], 2)
            seg.elasticity = 0.999
            self.space.add(seg)

    def _setup_block(self):
        pts = self.options['block_pts']
        mass = self.options['block_mass']
        init_pos = Vec2d(*self.options['block_start_pos'])
        moment = pymunk.moment_for_poly(mass, pts)
        self.block_body = pymunk.Body(mass, moment)
        self.block_body.position = init_pos
        self.block_shape = pymunk.Poly(self.block_body, pts)
        self.block_shape.elasticity = self.options['elasticity']
        self.block_shape.friction = self.options['friction']
        self.block_shape.color = self.options['block_color']
        self.space.add(self.block_body, self.block_shape)

    def _setup_block_lines(self):
        pts = self.options['block_pts']
        mass = self.options['block_mass']
        init_pos = Vec2d(*self.options['block_start_pos'])
        moment = pymunk.moment_for_poly(mass, pts)
        self.block_body = pymunk.Body(mass, moment)
        self.block_body.position = init_pos
        self.block_shape = pymunk.Poly(self.block_body, pts)

        self.space.add(self.block_body)
        for i in range(len(pts)):
            seg = pymunk.Segment(self.block_body, pts[i], pts[(i+1)%len(pts)], 1)
            seg.elasticity = self.options['elasticity']
            seg.friction = self.options['friction']
            seg.color = self.options['block_color']
            self.space.add(seg)

    def _norm_pt(self, P, len_x, len_y):
        pt = P.x - (len_x - 1.0)/2.0, P.y - (len_y - 1.0)/2.0
        return Vec2d(*pt)

    def _setup_block_img(self):
        img = self.options['block_img']
        len_x = len(img)
        len_y = len(img[0])
        scale = self.options['block_img_scale']
        def sample_func(point):
            x = int(point[0])
            y = int(point[1])
            return 1 if img[x][y] == "x" else 0
        pl_set = self.options['march_fn'](pymunk.BB(0,0,len_x-1,len_y-1), len_x, len_y, .5, sample_func)
        edge_set = []
        for poly_line in pl_set:
            for i in range(len(poly_line) - 1):
                a = self._norm_pt(poly_line[i], len_x, len_y)
                edge_set.append(scale*a)

        mass = self.options['block_mass']
        init_pos = Vec2d(*self.options['block_start_pos'])
        moment = pymunk.moment_for_poly(mass, edge_set)
        self.block_body = pymunk.Body(mass, moment)
        self.block_body.position = init_pos

        self.space.add(self.block_body)
        for poly_line in pl_set:
            for i in range(len(poly_line) - 1):
                a = self._norm_pt(poly_line[i], len_x, len_y)
                b = self._norm_pt(poly_line[i + 1], len_x, len_y)
                seg = pymunk.Segment(self.block_body, scale*a, scale*b, 1)
                seg.elasticity = self.options['elasticity']
                seg.friction = self.options['friction']
                seg.color = self.options['block_color']
                self.space.add(seg)


    def _setup_pusher(self):
        init_pos = Vec2d(*self.options['pusher_start_pos'])
        mass = self.options['pusher_mass']
        radius = self.options['pusher_radius']
        moment = pymunk.moment_for_circle(mass, 0, radius, (0, 0))

        self.key_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.key_body.position = init_pos
        self.key_shape = pymunk.Circle(self.key_body, 0.01, (0, 0))
        self.key_shape.filter = pymunk.ShapeFilter(categories=0, mask=0)
        self.space.add(self.key_body, self.key_shape)

        self.push_body = pymunk.Body(mass, moment)
        self.push_body.position = init_pos
        self.push_shape = pymunk.Circle(self.push_body, radius, (0, 0))
        self.push_shape.elasticity = self.options['elasticity']
        self.push_shape.friction = self.options['friction']
        self.push_shape.color = self.options['pusher_color']
        self.space.add(self.push_body, self.push_shape)

        c = pymunk.constraints.DampedSpring(self.key_body, self.push_body, 
            anchor_a = (0,0), anchor_b = (0,0), rest_length=0.0, 
            stiffness=self.options['controller_stiffness'], 
            damping=self.options['controller_damping'])
        self.space.add(c)

    def _setup_target(self):
        pts = self.options['block_pts']
        pos = Vec2d(*self.options['target_start_pos'])
        moment = pymunk.moment_for_circle(0.1, 0, 0.1, (0, 0))
        self.target_body = pymunk.Body(0.1, moment)
        self.target_body.position = pos
        self.target_body.angle = self.options['target_start_angle']

        self.space.add(self.target_body)
        for i in range(len(pts)):
            seg = pymunk.Segment(self.target_body, pts[i], pts[(i+1)%len(pts)], 1)
            seg.filter = pymunk.ShapeFilter(categories=0, mask=0)
            seg.color = self.options['target_color']
            self.space.add(seg)

    def _setup_target_img(self):
        img = self.options['block_img']
        mass = self.options['block_mass']
        pos = Vec2d(*self.options['target_start_pos'])
        moment = pymunk.moment_for_circle(0.1, 0, 0.1, (0, 0))
        self.target_body = pymunk.Body(mass, moment)
        self.target_body.position = pos
        self.target_body.angle = self.options['target_start_angle']
        self.space.add(self.target_body)

        len_x = len(img)
        len_y = len(img[0])
        scale = self.options['block_img_scale']
        def sample_func(point):
            x = int(point[0])
            y = int(point[1])
            return 1 if img[x][y] == "x" else 0
        pl_set = self.options['march_fn'](pymunk.BB(0,0,len_x-1,len_y-1), len_x, len_y, .5, sample_func)

        for poly_line in pl_set:
            for i in range(len(poly_line) - 1):
                a = poly_line[i]
                b = poly_line[i + 1]
                seg = pymunk.Segment(self.target_body, scale*a, scale*b, 1)
                seg.filter = pymunk.ShapeFilter(categories=0, mask=0)
                seg.color = self.options['target_color']
                self.space.add(seg)

    def _shapely_draw_shape(self, shape,body,color):
        try:
            poly = pymunk_to_shapely(body, shape)
            if isinstance(poly, sg.Polygon):
                # Convert to pygame drawing coordinates
                points = [(int(x), int(y))
                        for x,y in poly.exterior.coords]
                pygame.draw.polygon(self.screen, color, points, 0)
        except RuntimeError as e:
            print(f"Could not draw shape: {e}")

    def step(self, action):
        dx = action[0]
        dy = action[1]
        curx = self.key_body.position.x
        cury = self.key_body.position.y
        self.key_body.position = Vec2d(curx + dx, cury + dy)
        self.space.step(self.options['dt'])
        
    def render(self):
        self.screen.fill(pygame.Color("white"))
        #self._shapely_draw_shape(self.block_body.shapes,self.block_body,self.options['block_color'])
        #self._shapely_draw_shape(self.target_body.shapes,self.target_body,self.options['target_color'])
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(20)

def main():
    pusher = Pusher()
    delta_action = 4
    running = True
    while running:
        all_keys = pygame.key.get_pressed()
        action = [0., 0.]
        if all_keys[pygame.K_DOWN]:
            action[1] += delta_action
        if all_keys[pygame.K_UP]:
            action[1] -= delta_action
        if all_keys[pygame.K_LEFT]:
            action[0] -= delta_action
        if all_keys[pygame.K_RIGHT]:
            action[0] += delta_action

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False


        pusher.step(action)
        pusher.render()


def record_simulation(states_file, actions_file, states_txt, actions_txt, video_file_name):
    pusher = Pusher()
    delta_action = 4
    running = True

    recorded_states = []
    recorded_actions = []
    video_frames = []

    pusher_pos_x = float(random.randint(100, 500))
    pusher_pos_y = float(random.randint(100, 500))

    pusher_angle = float(random.random()* 2 * np.pi - np.pi)


    target_pos_x = float(random.randint(100, 500))
    target_pos_y = float(random.randint(100, 500))

    target_angle = float(random.random()* 2 * np.pi - np.pi)

    x = random.randint(1,4)

    if x==1:
        push_body_x = pusher_pos_x - 80
        push_body_y = pusher_pos_y
    if x==2:
        push_body_x = pusher_pos_x + 80
        push_body_y = pusher_pos_y
    if x==3:
        push_body_x = pusher_pos_x
        push_body_y = pusher_pos_y - 80
    if x==4:
        push_body_x = pusher_pos_x
        push_body_y = pusher_pos_y + 80
    
    pusher.block_body.position = Vec2d(pusher_pos_x,pusher_pos_y)
    pusher.block_body.angle = pusher_angle 

    pusher.push_body.position = Vec2d(push_body_x,push_body_y)
    pusher.key_body.position = Vec2d(push_body_x,push_body_y)

    pusher.target_body.position=Vec2d(target_pos_x,target_pos_y)
    pusher.target_body.angle=target_angle

    pusher.step([0.0, 0.0])
    pusher.render()

    while running:
       
        all_keys = pygame.key.get_pressed()
        action = [0.0, 0.0]
        if all_keys[pygame.K_DOWN]:
            action[1] += delta_action
        if all_keys[pygame.K_UP]:
            action[1] -= delta_action
        if all_keys[pygame.K_LEFT]:
            action[0] -= delta_action
        if all_keys[pygame.K_RIGHT]:
            action[0] += delta_action

        if (args.skip_zero_action == 'Y' and (action[0] != 0.0 or action[1] != 0.0)) or args.skip_zero_action == 'N':

            recorded_actions.append(deepcopy(action))
            
            state_list = [
                # Pusher state
                pusher.push_body.position.x,
                pusher.push_body.position.y,
                # Block state
                pusher.block_body.position.x,
                pusher.block_body.position.y,
                pusher.block_body.angle,

                pusher.target_body.position.x,
                pusher.target_body.position.y,
                pusher.target_body.angle,

            ]
            recorded_states.append(deepcopy(state_list))

        
            pusher.render()
            frame = pygame.surfarray.array3d(pusher.screen)
            video_frames.append(frame.copy())

            pusher.step(action)
        


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False


    torch.save(recorded_states, states_file)
    torch.save(recorded_actions, actions_file)

    with open(states_txt, 'w') as f:
        for state in recorded_states:
            formatted_state = [f"{x:.4f}" for x in state]
            f.write(str(formatted_state) + '\n')
    
    with open(actions_txt, 'w') as f:
        for action in recorded_actions:
            formatted_action = [f"{x:.4f}" for x in action]
            f.write(str(formatted_action) + '\n')

 
    save_video(video_frames, video_file_name)

def create_video_from_actions(input_filename, output_filename, init_state, states_from_actions_txt):

    # pygame.quit()
    # os.environ['SDL_VIDEODRIVER'] = 'dummy'
    # pygame.init()
    
    recorded_actions = torch.load(input_filename)

    pusher = Pusher()
    pusher.push_body.position = Vec2d(init_state[0], init_state[1])
    pusher.block_body.position = Vec2d(init_state[2], init_state[3])
    pusher.block_body.angle = init_state[4]
    pusher.key_body.position = Vec2d(float(init_state[0]), float(init_state[1]))

    pusher.target_body.position=Vec2d(float(init_state[5]), float(init_state[6]))
    pusher.target_body.angle=init_state[7]

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

            pusher.target_body.position.x,
            pusher.target_body.position.y,
            pusher.target_body.angle,
        ]
        rollout_states.append(deepcopy(state_list))

  
        pusher.render()
        frame = pygame.surfarray.array3d(pusher.screen)
        video_frames.append(frame.copy())


        pusher.step(action)
    
    with open(states_from_actions_txt, 'w') as f:
        for state in rollout_states:
            formatted_state = [f"{x:.4f}" for x in state]
            f.write(str(formatted_state) + '\n')


    save_video(video_frames, output_filename)


def save_video(frames, filename, fps=30):
    if not frames:
        print("No frames to save.")
        return

    # Prepare the first frame and determine frame dimensions
    frame0 = np.transpose(frames[0], (1, 0, 2))
    height, width = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Initialize VideoWriter and check if it is opened successfully
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise IOError("Could not open the video writer.")

    # Write each frame to the video file
    frame_count = 0
    for frame in frames:
        frame_cv = np.transpose(frame, (1, 0, 2))
        video_writer.write(frame_cv)
        frame_count += 1
    video_writer.release()
    print(f"{frame_count} frames written to {filename}.")

    # Optional: Verify the saved video frame count
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print("Could not open the saved video for verification.")
    else:
        saved_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"Verification: Video contains {saved_frames} frames.")




def combine_videos_side_by_side(video1_path, video2_path, output_path):
    """Combine two videos side by side into a single video"""

    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not (cap1.isOpened() and cap2.isOpened()):
        print("Error: Could not open one or more videos")
        return

    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap1.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))  # Adjust width for two videos

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not (ret1 and ret2):
            break

        combined_frame = np.hstack((frame1, frame2))  # Combine two frames side by side

        out.write(combined_frame)

    cap1.release()
    cap2.release()
    out.release()

def compare_state_files(states_from_actions_txt, recorded_states_txt):
    """Compare two state files line by line and count differences"""
    with open(states_from_actions_txt, 'r') as f1:
        lines1 = f1.readlines()
    
    with open(recorded_states_txt, 'r') as f2:
        lines2 = f2.readlines()

    different_lines = sum(1 for l1, l2 in zip(lines1, lines2) if l1 != l2)
    
    print(f"Number of lines that differ: {different_lines}")

if __name__ == "__main__":
    
    # PTH files
    states_pth = f"./states.pth"
    actions_pth = f"./actions.pth"
    
    # TXT files
    recorded_states_txt = f"./states.txt"
    recorded_actions_txt = f"./actions.txt"
    states_from_actions_txt = f"./states_from_actions.txt"

    
    # MP4 files
    recorded_video_path = f"./recorded_simulation.mp4"
    actions_video_path = f"./actions.mp4"
    combined_video_path = f"./combined_video.mp4"

    # Step 1: Record simulation
    record_simulation(states_pth, actions_pth, recorded_states_txt, recorded_actions_txt, recorded_video_path)

    # Step 2: Load states and create videos
    recorded_states = torch.load(states_pth)

    # Sanity Check

    create_video_from_actions(actions_pth, actions_video_path, recorded_states[0], states_from_actions_txt)
    
    combine_videos_side_by_side(recorded_video_path, actions_video_path, combined_video_path)

    compare_state_files(recorded_states_txt, states_from_actions_txt)
    
    sys.exit(0)