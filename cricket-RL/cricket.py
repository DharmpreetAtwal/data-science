# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: % Dharm Atwal
"""

import pyautogui
import time
import numpy as np
from pyscreeze import Box
import cv2
from PIL import Image
import gymnasium as gym
from typing import Optional

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

from keras import Sequential
from keras.layers import Dense, Normalization, Flatten, Input, Conv2D, Concatenate
from keras.models import Model
from keras.optimizers import Adam

# %%

def capture_ss(capture_area):
    capture_ss = pyautogui.screenshot(
        region=(int(capture_area.left), 
                int(capture_area.top), 
                int(capture_area.width), 
                int(capture_area.height)))
    
    
    # capture_ss.show()
    
    
    return resize(capture_ss)

def resize(capture_ss):
    # Preprocessing, resize to lower resolution
    # Maintain aspect ratio, 5x less data
    ratio = capture_ss.height / capture_ss.width
    new_width = int(capture_ss.width / 5)
    new_height = int(new_width * ratio)
    
    # Convert PIL Image to NP array
    arr_image = np.array(capture_ss)
    bgr = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    
    # Convert from RGB to BGR, then back to RGB
    resized_bgr = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
    resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    
    
    # resized_pil = Image.fromarray(resized_rgb)
    # resized_pil.show()
    
    
    return resized_rgb
    

    
# %%
# {"mouse": array([x, y]), "screen": (screen-shot)}
class CricketMasterEnv(gym.Env):
    def __init__(self, capture_area: Box):
        super(CricketMasterEnv, self).__init__()
        
        self.step_count = 0
        self._capture_area = Box(capture_area.left,
                                capture_area.top, 
                                capture_area.width, 
                                capture_area.height)

        center_x, center_y = self.get_center()
        self.pixel = pyautogui.screenshot().getpixel((center_x, center_y))

        pyautogui.moveTo(center_x, center_y)
        obs = self._get_obs()
        
        self._mouse = obs['mouse']
        self._screen = obs["screen"]
        
        self.action_space = gym.spaces.Discrete(4)
        self._discrete_to_dir = {
            0: np.array([1, 0]),   # +x
            1: np.array([0, 1]),   # +y
            2: np.array([-1, 0]),  # -x
            3: np.array([0, -1]),  # -y
        }

        self.observation_space = gym.spaces.Dict(
            {
                "mouse": gym.spaces.Box(low=np.array([0.0, 0.0]), 
                                        high=np.array([1.0, 1.0]), 
                                        dtype=np.float64),
                "screen": gym.spaces.Box(low=0, high=255, 
                                     shape=(len(self._screen),
                                            len(self._screen[0]),
                                            3), 
                                     dtype=np.uint8)
            }    
        )   

    def _get_obs(self):
        screen = capture_ss(self._capture_area)
        pos = pyautogui.position()
        mouse = np.array([(pos.x - self._capture_area.left) / self._capture_area.width, 
                               int(pos.y - self._capture_area.top) / self._capture_area.height],
                               dtype=np.float64)
        
        return {"mouse": mouse, "screen": screen}
    
    def get_center(self):
        center_x = int(self._capture_area.left + (self._capture_area.width / 2))
        center_y = int(self._capture_area.top + (self._capture_area.height / 2))
        
        return center_x, center_y

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        
        center_x, center_y = self.get_center()
        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()
        
        obs = self._get_obs()
        return obs, {}
        
    def step(self, action):
        incrm = self._discrete_to_dir[action]
        pyautogui.moveRel(incrm[0], incrm[1])

        self.step_count += 1
        obs = self._get_obs()
        reward = self.step_count
        
        center_x, center_y = self.get_center()
        done = pyautogui.screenshot().getpixel((center_x, center_y)) == self.pixel

        return obs, reward, done, {}
    
# %%


def build_model(env):
    # Process screen input
    screen_input = Input(shape=env.observation_space['screen'].shape)
    screen_features = Conv2D(32, (8, 8), strides=4, activation='relu')(screen_input)
    screen_features = Conv2D(64, (4, 4), strides=2, activation='relu')(screen_features)
    screen_features = Conv2D(64, (3, 3), strides=1, activation='relu')(screen_features)
    screen_features = Flatten()(screen_features)
    
    # Process mouse position input
    mouse_input = Input(shape=env.observation_space['mouse'].shape)
    mouse_features = Dense(16, activation='relu')(mouse_input)
    
    # Combine features
    combined = Concatenate()([screen_features, mouse_features])
    combined = Dense(512, activation='relu')(combined)
    output = Dense(env.action_space.n, activation='linear')(combined)
    
    model = Model(inputs=[screen_input, mouse_input], outputs=output)
    return model

def build_agent(model, env):
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=env.action_space.n,
        nb_steps_warmup=1000,
        target_model_update=1e-2
    )
    dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
    return dqn

def main():
    time.sleep(2)
    
    # Use pre-determined menu identifier to determine play/capture area
    # Play represents area where mouse can move
    # Capture represents area where the screenshot fed into the model is taken
    locate_area = pyautogui.locateOnScreen('locate_menu.png', confidence=0.90)
    
    capture_area = Box(locate_area.left,
                       locate_area.top, 
                       locate_area.width, 
                       int(locate_area.height * 1.7))
    play_area = Box(locate_area.left, 
                    locate_area.top, 
                    locate_area.width, 
                    int(locate_area.height * 2.3))
    
    # Your existing environment setup code here
    env = CricketMasterEnv(capture_area)
    
    model = build_model(env)
    dqn = build_agent(model, env)
    
    # Train the agent
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
    
    # Save the trained weights
    dqn.save_weights('dqn_cricket_weights.h5f', overwrite=True)
    
    # Test the agent
    dqn.test(env, nb_episodes=5, visualize=True)

main()
# %%

