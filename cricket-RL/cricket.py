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
# %%

time.sleep(2)

# Use pre-determined menu identifier to determine play/capture area
# Play represents area where mouse can move
# Capture represents area where the screenshot fed into the model is taken
locate_area = pyautogui.locateOnScreen('locate_menu.png', confidence=0.95)
capture_area = Box(locate_area.left, locate_area.top, locate_area.width, locate_area.height * 1.7)
play_area = Box(locate_area.left, locate_area.top, locate_area.width, locate_area.height * 2.3)

play_ss = pyautogui.screenshot(
    region=(int(play_area.left), 
            int(play_area.top), 
            int(play_area.width), 
            int(play_area.height)))
play_ss.show()

capture_ss = pyautogui.screenshot(
    region=(int(capture_area.left), 
            int(capture_area.top), 
            int(capture_area.width), 
            int(capture_area.height)))
capture_ss.show()

# %%
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
resized_pil = Image.fromarray(resized_rgb)
resized_pil.show()