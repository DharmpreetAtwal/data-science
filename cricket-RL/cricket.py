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
ratio = image.height / image.width
new_width = 100
new_height = int(new_width * ratio)

arr = np.array(image)
bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
resized_bgr = cv2.resize(bgr, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

resized_rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)

resized_pil = Image.fromarray(resized_rgb)
resized_pil.show()