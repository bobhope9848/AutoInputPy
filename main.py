import cv2

path = "C:\\Users\\bobhope\\Documents\\youtube-dl\\New frames\\"


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
        if frame[value.coords] > 120:
            value.ispressed = True
        else:
            value.ispressed = False
    return controls


# Loop through folder of frames
for n in range(1, 41):
    # Cuts game+control region from 1280x720 image
    screen = cv2.imread(path + f"video{n}.png", 0)[130:720, 220:1265]
    img_control = cv2.imread(path + f"video{n}.png", 0)[20:117, 455:887]

    # Checks which keys were pressed in frame
    controls = check_if_active(controls, img_control)
    print(controls['z'].ispressed)

    # Grabs edges from game screen
    screen = cv2.GaussianBlur(screen, (5, 5), 0)
    screen = cv2.Canny(screen, 103, 227)

    # Display game screen
    cv2.imshow("image", screen)
    print(f"video{n}.png")
    cv2.waitKey(0)
