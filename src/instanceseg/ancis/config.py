# -*- coding: utf-8 -*-

INPUT_SIZE = (256, 256)
USE_RANDOM_ROT = True
USE_RANDOM_FLIP = True
USE_DOWNSIZE = False # approximation of x20 scan from x40 acquisitions
USE_RANDOM_CROP = False
USE_RANDOM_GRAY = False
USE_COLOR_JITTER = False

# --- additionnal params for color jitter
DELTA_BRIGHTNESS = 0.25
DELTA_CONTRAST = 0.25
DELTA_SATURATION = 0.25
DELTA_HUE = 0.5
