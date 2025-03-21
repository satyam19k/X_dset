"""This example showcase an arrow pointing or aiming towards the cursor.
"""

__docformat__ = "reStructuredText"

import sys
import os
import pygame

from copy import deepcopy

import pymunk
import pymunk.constraints
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
from pymunk.autogeometry import march_soft, march_hard
from IPython import embed
import cv2  
import torch  
from copy import deepcopy

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
    'block_img': X_img,
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

def get_next_file_number(directory, prefix):
    """Get the next available file number in the sequence"""
    # Create directory if it doesn't exist
    
    # List all files in directory that match our prefix
    existing_files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith('.pth')]
    
    if not existing_files:
        return 1
    numbers = []
    for filename in existing_files:
        try:
            # Extract number between prefix and .pth
            num = int(filename[len(prefix):-4])
            numbers.append(num)
        except ValueError:
            continue 
    if not numbers:
        return 1
    return max(numbers) + 1

class Pusher:
    def __init__(self, options = OPTIONS):
        self.options = options
        pygame.init()
        self.screen = pygame.display.set_mode(self.options['screen_size'])
        self.clock = pygame.time.Clock()
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = self.draw_options.DRAW_SHAPES | self.draw_options.DRAW_COLLISION_POINTS
    
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


    def step(self, action):
        dx = action[0]
        dy = action[1]
        curx = self.key_body.position.x
        cury = self.key_body.position.y
        self.key_body.position = Vec2d(curx + dx, cury + dy)
        self.space.step(self.options['dt'])
        
    def render(self):
        self.screen.fill(pygame.Color("white"))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(50)

def main():
    pusher = Pusher()
    delta_action = 2
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



def record_simulation(states_file, actions_file, video_file):
    pusher = Pusher()
    delta_action = 2
    running = True

    recorded_states = []
    recorded_actions = []
    video_frames = []

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

        recorded_actions.append(deepcopy(action))

        state_list = [
        # Block state
            pusher.block_body.position.x,
            pusher.block_body.position.y,
            pusher.block_body.angle,
            # Pusher state
            pusher.push_body.position.x,
            pusher.push_body.position.y,
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

    torch.save(torch.tensor(recorded_states), states_file)
    torch.save(torch.tensor(recorded_actions), actions_file)

    with open('states.txt', 'w') as f:
        for state in recorded_states:
            f.write(str(state) + '\n')
    with open('actions.txt', 'w') as f:
        for action in recorded_actions:
            f.write(str(action) + '\n')

    save_video(video_frames, video_file)

def create_video_from_actions(input_filename, output_filename,init_state):
    recorded_actions = input_filename

    pusher = Pusher()

    pusher.block_body.position = Vec2d(init_state[0], init_state[1])
    pusher.block_body.angle = init_state[2]

    

    pusher.push_body.position = Vec2d(init_state[3], init_state[4])

    rollout_states = []
    video_frames = []

    for action in recorded_actions:

        state_list = [
        # Block state
            pusher.block_body.position.x,
            pusher.block_body.position.y,
            pusher.block_body.angle,

            pusher.push_body.position.x,
            pusher.push_body.position.y,
        ]
        rollout_states.append(deepcopy(state_list))

        pusher.render()
        frame = pygame.surfarray.array3d(pusher.screen)
        video_frames.append(frame.copy())


        pusher.step(action)

    save_video(video_frames, 'rollout_actions.mp4')
    
    return video_frames




def save_video(frames, filename, fps=30):
    if not frames:
        return

    frame0 = np.transpose(frames[0], (1, 0, 2))
    height, width = frame0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for frame in frames:

        frame_cv = np.transpose(frame, (1, 0, 2))
        video_writer.write(frame_cv)
    video_writer.release()


# def create_video_from_states(input_filename,output_filename):
#     recorded_states = input_filename
#     video_frames = []

#     pusher = Pusher()
#     rollout_states = []
#     for state in recorded_states:
#         # Create a new Pusher instance for this state

#         pusher.block_body.position = Vec2d(state[0], state[1])
#         pusher.block_body.angle = state[2]
#         # pusher.block_body.velocity = Vec2d(state[3], state[4])
#         # pusher.block_body.angular_velocity = state[5]
        
#         # Pusher state (indices 6-11)
#         pusher.push_body.position = Vec2d(state[3], state[4])
#         # pusher.push_body.angle = state[8]
#         # pusher.push_body.velocity = Vec2d(state[9], state[10])
#         # pusher.push_body.angular_velocity = state[11]
        
#         # Target state (indices 12-17)
#         # pusher.target_body.position = Vec2d(state[12], state[13])
#         # pusher.target_body.angle = state[14]
#         # pusher.target_body.velocity = Vec2d(state[15], state[16])
#         # pusher.target_body.angular_velocity = state[17]
        

        
#         new_state_list = [
#         # Block state
#             pusher.block_body.position.x,
#             pusher.block_body.position.y,
#             pusher.block_body.angle,
#             # pusher.block_body.velocity.x,
#             # pusher.block_body.velocity.y,
#             # pusher.block_body.angular_velocity,
#             # Pusher state
#             pusher.push_body.position.x,
#             pusher.push_body.position.y,
#             # pusher.push_body.angle,
#             # pusher.push_body.velocity.x,
#             # pusher.push_body.velocity.y,
#             # pusher.push_body.angular_velocity,
#             # Target state
#             # pusher.target_body.position.x,
#             # pusher.target_body.position.y,
#             # pusher.target_body.angle,
#             # pusher.target_body.velocity.x,
#             # pusher.target_body.velocity.y,
#             # pusher.target_body.angular_velocity
#         ]
#         pusher.space.step(pusher.options['dt'])
#         pusher.render()

#         pygame.event.pump()
#         frame = pygame.surfarray.array3d(pusher.screen)
#         video_frames.append(frame.copy())
#         rollout_states.append(new_state_list)

#     save_video(video_frames, "rollout_states.mp4")

#     return video_frames



if __name__ == "__main__":
    os.makedirs("./states", exist_ok=True)
    os.makedirs("./actions", exist_ok=True)
    os.makedirs("./states_video", exist_ok=True)


    while True:
        next_number = get_next_file_number("./states", "states")
        states_file = f"./states/states{next_number}.pth"
        actions_file = f"./actions/actions{next_number}.pth"
        record_simulation(states_file, actions_file,f"./states_video/states{next_number}.mp4")
    sys.exit(0)
