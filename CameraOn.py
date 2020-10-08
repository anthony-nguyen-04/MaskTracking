###
# Note: Before you run this, you have to do install opencv-python.
# Whenever you first get python, it should come with pip. Do "pip install opencv-python" in the cmd terminal.
###

import cv2

# Turns on Camera feed from port 0 [default] and assigns it to "video" variable
video = cv2.VideoCapture(0)

while True:

    # Continuously takes in the video input and assigns it to "frame"
    _, frame = video.read()

    # Displays the camera feed under the "video" tab
    cv2.imshow("video", frame)

    # Displays the frame at-the-moment before going to the next one, like an actual camera feed
    cv2.waitKey(1)