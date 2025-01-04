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

time.sleep(1)
play_area = pyautogui.locateOnScreen('play_area.png', confidence=0.9)
capture_area = Box(play_area.left, play_area.top, play_area.width, play_area.height * 0.71)

image = pyautogui.screenshot(
    region=(int(play_area.left), 
            int(play_area.top), 
            int(play_area.width), 
            int(play_area.height)))
image.show()

image2 = pyautogui.screenshot(
    region=(int(capture_area.left), 
            int(capture_area.top), 
            int(capture_area.width), 
            int(capture_area.height)))
image2.show()


# %%
# Preprocessing, resize 
ratio = image2.height / image2.width
new_width = int(image2.width / 5)
new_height = int(new_width * ratio)

# Convert PIL Image to NP array, Convert to BGR from RGB
arr_image = np.array(image2)
bgr = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)

resized_bgr = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST_EXACT)
resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
resized_pil = Image.fromarray(resized_rgb).convert('L')
resized_pil.show()