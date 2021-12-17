import os

import cv2
import ast
import pandas
from pathlib import Path as Path
import torch
from torch import nn

from fastcore.meta import use_kwargs_dict

from fastai.callback.fp16 import to_fp16
from fastai.callback.progress import ProgressCallback
from fastai.callback.schedule import lr_find, fit_one_cycle

from fastai.data.block import MultiCategoryBlock, DataBlock
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import RandomSplitter, ColReader

from fastai.metrics import accuracy_multi, BaseLoss

from fastai.vision.augment import aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.learner import cnn_learner

from torchvision.models import resnet34

input_dir = Path("/Users/bobhope/Documents/youtube-dl/New frames/")
output_dir = input_dir / "formatted/"


# Defines structure of input object
class Input:
    def __init__(self, coords, ispressed=False):
        self.coords = coords
        self.ispressed = ispressed


class Labeler:

    def __init__(self, filename, buttons):
        self.filename = filename
        # This would be a list but idk how to set that here globally instead of passing list
        self.buttons = buttons


# define each controls cords to check in frame and if pressed
controls = {'a': Input((12, 11)),
            's': Input((12, 58)),
            'd': Input((12, 105)),
            'f': Input((12, 152)),
            'z': Input((60, 10)),
            'x': Input((60, 58)),
            'c': Input((60, 103)),
            'i': Input((60, 153)),
            'up': Input((8, 339)),
            'down': Input((57, 339)),
            'left': Input((57, 286)),
            'right': Input((57, 392))
            }


# Checks which buttons in control frame is active
def check_if_active(controls_list, frame):
    # Check which controls are active
    for ctrl, val in controls.items():
        # Button active if pixel value is over 120. I.E closer to pure white which is active
        if frame[val.coords] > 120:
            val.ispressed = True
        else:
            val.ispressed = False
    return controls


save_list = \
    {
        'filename': [],
        'buttons': [],
    }

#   Create output directory if doesn't exist
frame_out = output_dir / "frames/"
frame_out.mkdir(parents=True, exist_ok=True)

# Loop through folder of frames
for n in range(1, 23):
    path = str((input_dir / f"video{n}.png").resolve())
    # Cuts game+control region from 1280x720 image
    screen = cv2.imread(path, 0)[130:720, 220:1265]
    img_control = cv2.imread(path, 0)[20:117, 455:887]
    # Checks which keys were pressed in frame
    active_keys = []
    controls = check_if_active(controls, img_control)
    for key, value in controls.items():
        # Adds found keys to active list
        if value.ispressed:
            active_keys.append(key)
        # Debug
        print("Key {0} is pressed: {1}".format(key, value.ispressed))

    # Checks if active keys were detected
    if not len(active_keys) == 0:
        # appends working image name and active keys found in image to list
        save_list['filename'].append(f"video{n}")
        save_list['buttons'].append(','.join(active_keys))
        # Grabs edges from game screen
        screen = cv2.resize(screen, (418, 236))
        screen = cv2.GaussianBlur(screen, (5, 5), 0)
        screen = cv2.Canny(screen, 103, 227)
        # Write game screen
        cv2.imwrite(str((frame_out / f"video{n}.png").resolve()), screen)

    # Display game screen
    # cv2.imshow("image", screen)
    # print(f"video{n}.png")
    # cv2.waitKey(0)

"""Controls write-out section"""
# save filenames and their respective buttons in csv file
name_of_file = (output_dir / "ctrl/control_list.ctrl").resolve()
name_of_file.parent.mkdir(parents=True, exist_ok=True)
# Write active controls in frame to json file
df = pandas.DataFrame(save_list)
with open(name_of_file, 'w+') as outfile:
    df.to_csv(outfile, index=None)

"""CSV read-in section"""

# Read-in csv to panda frame then convert to dictionary with keys set as names from "filename" column
# list_of_dicts = pandas.read_csv(name_of_file).set_index('filename').T.to_dict("list")

"""Model building section"""

df.head()
model = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                  get_x=ColReader(0, pref=f'{str((output_dir / "frames").resolve())}{os.sep}', suff='.png'),
                  splitter=RandomSplitter(),
                  get_y=ColReader(1, label_delim=','))

dls = model.dataloaders(df, bs=6)
#dls.show_batch()
# dls.show_batch(max_n=9, figsize=(12,9))
