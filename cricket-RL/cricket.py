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
from keras import Sequential
from keras.layers import Dense, Normalization, Flatten

# %%

time.sleep(2)

# Use pre-determined menu identifier to determine play/capture area
# Play represents area where mouse can move
# Capture represents area where the screenshot fed into the model is taken
locate_area = pyautogui.locateOnScreen('locate_menu.png', confidence=0.90)
capture_area = Box(locate_area.left, locate_area.top, locate_area.width, locate_area.height * 1.7)
play_area = Box(locate_area.left, locate_area.top, locate_area.width, locate_area.height * 2.3)

center_x = capture_area.left + (capture_area.width / 2)
center_y = capture_area.top + (capture_area.height / 2)
done = pyautogui.screenshot().getpixel((center_x, center_y))

# %%

def capture_ss(capture_area):
    capture_ss = pyautogui.screenshot(
        region=(int(capture_area.left), 
                int(capture_area.top), 
                int(capture_area.width), 
                int(capture_area.height)))
    
    
    capture_ss.show()
    
    
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
        self._capture_area = capture_area
        
        center_x = capture_area.left + (capture_area.width / 2)
        center_y = capture_area.top + (capture_area.height / 2)
        self.pixel = pyautogui.screenshot().getpixel((center_x, center_y))

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
                "mouseX": gym.spaces.Box(low=0, high=1.0, shape=(), dtype=np.float16),
                "mouseY": gym.spaces.Box(low=0, high=1.0, shape=(), dtype=np.float16), 
                "screen": gym.spaces.Box(low=0, high=255, 
                                     shape=(capture_area.width,
                                            capture_area.height,
                                            3), dtype=np.uint8)
            }    
        )

    def _get_obs(self):
        screen = capture_ss(self._capture_area)
        pos = pyautogui.position()
        mouse = np.array([(pos.x - self._capture_area.left) / self._capture_area.width, 
                               (pos.y - self._capture_area.top) / self._capture_area.height],
                               dtype=np.float16)
        
        return {"mouse": mouse, "screen": screen}

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        
        obs = self._get_obs()
        self._mouse = obs['mouse']
        pyautogui.click()
        self._screen = obs['screen']
        
    def step(self, action):
        self.step_count += 1
        obs = self._get_obs()
        
        reward = self.step_count
        
        center_x = capture_area.left + (capture_area.width / 2)
        center_y = capture_area.top + (capture_area.height / 2)
        done = pyautogui.screenshot().getpixel((center_x, center_y)) == self.pixel

        return obs, reward, done, {}
    
# %%
time.sleep(2)

# Use pre-determined menu identifier to determine play/capture area
# Play represents area where mouse can move
# Capture represents area where the screenshot fed into the model is taken
locate_area = pyautogui.locateOnScreen('locate_menu.png', confidence=0.90)
capture_area = Box(locate_area.left, locate_area.top, locate_area.width, locate_area.height * 1.7)
play_area = Box(locate_area.left, locate_area.top, locate_area.width, locate_area.height * 2.3)

env = CricketMasterEnv(capture_area)

# %%
model = Sequential()
model.add(Normalization())
model.add(Flatten(input_shape=))