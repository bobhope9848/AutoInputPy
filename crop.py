#https://stackoverflow.com/questions/47785918/python-pil-crop-all-images-in-a-folder
from PIL import Image
import os.path, sys

path = "C:\\Users\\bobhope\\Documents\\youtube-dl\\Frames\\"
dirs = os.listdir(path)

def crop():
    for item in dirs:
        fullpath = os.path.join(path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            f, e = os.path.splitext(fullpath)
            imCrop = im.crop((0, 506, 480, 679)) #corrected
            imCrop.save(f + 'Cropped.bmp', "BMP", quality=100)

crop()