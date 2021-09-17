import cv2
import json
import jsonpickle

dir = "C:\\Users\\bobhope\\Documents\\youtube-dl\\New frames\\"


# Defines structure of input object
class Input:
    def __init__(self, coords, ispressed = False):
        self.coords = coords
        self.ispressed = ispressed


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
    for ctrl, value in controls.items():
        # Button active if pixel value is over 120. I.E closer to pure white which is active
        if frame[value.coords] > 120:
            value.ispressed = True
        else:
            value.ispressed = False
    return controls


# Loop through folder of frames
for n in range(580, 39179):
    path = dir + f"video{n}.png"
    # Cuts game+control region from 1280x720 image
    screen = cv2.imread(path, 0)[130:720, 220:1265]
    img_control = cv2.imread(path, 0)[20:117, 455:887]

    # Checks which keys were pressed in frame
    controls = check_if_active(controls, img_control)
    for key, value in controls.items():
        print("Key {0} is pressed: {1}".format(key, value.ispressed))

    # Write active controls in frame to json file
    #with open(dir + f"\\Control\\video{n}.json", 'w') as outfile:
    #    json.dump(jsonpickle.encode(controls), outfile)

    # Grabs edges from game screen
    screen = cv2.resize(screen, (418,236))
    screen = cv2.GaussianBlur(screen, (5, 5), 0)
    screen = cv2.Canny(screen, 103, 227)

    # Display game screen
    cv2.imshow("image", screen)
    print(f"video{n}.png")
    cv2.waitKey(0)
