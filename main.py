import cv2
import ast
import pandas

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

directory = "C:\\Users\\bobhope\\Documents\\youtube-dl\\New frames\\"


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

# Loop through folder of frames
for n in range(1, 23):
    path = directory + f"video{n}.png"
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

    # appends working image name and active keys found in image to list
    save_list['filename'].append(f"video{n}")
    save_list['buttons'].append(','.join(active_keys))

    # Grabs edges from game screen
    screen = cv2.resize(screen, (418, 236))
    screen = cv2.GaussianBlur(screen, (5, 5), 0)
    screen = cv2.Canny(screen, 103, 227)

    # Write game screen
    cv2.imwrite(directory + f"formatted\\frames\\video{n}.png", screen)

    # Display game screen
    # cv2.imshow("image", screen)
    # print(f"video{n}.png")
    # cv2.waitKey(0)

"""Controls write-out section"""

# save filenames and their respective buttons in csv file
name_of_file = directory + "formatted\\ctrl\\control_list.ctrl"
# Write active controls in frame to json file
df = pandas.DataFrame(save_list)
with open(name_of_file, 'w') as outfile:
    df.to_csv(outfile, index=None)

"""CSV read-in section"""

"""
# Read csv into Panda dataframe
df = pandas.read_csv(name_of_file)
# Convert dataframe to dictionary with keys set as names from "filename" column
list_of_dicts = df.set_index("filename").T.to_dict()

# Retrieves lists stuck as string

# Stolen from https://www.kite.com/python/answers/how-to-loop-through-all-nested-dictionary-values-using-a-for-loop-in-python
def get_all_values(nested_dictionary):
    for keys, valued in nested_dictionary.items():
        if type(valued) is dict:
            get_all_values(valued)
        else:
            if valued != "[]":
                nested_dictionary[keys] = ast.literal_eval(valued)
get_all_values(list_of_dicts)
"""

"""Model building section"""

df.head()
model = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                  get_x=ColReader(0, pref=directory + f"formatted\\frames\\", suff='.png'),
                  splitter=RandomSplitter(),
                  get_y=ColReader(1, label_delim=','))

dls = model.dataloaders(df, bs=5)
print(dls)
dls.show_batch()
# dls.show_batch(max_n=9, figsize=(12,9))
