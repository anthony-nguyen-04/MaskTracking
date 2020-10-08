###
# Using Haar Cascades for facial detection.
# Realistically, we're just gonna use DLib for it, but this is a good starting point
###

import cv2

# The cascade database that we use to determine a face
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video = cv2.VideoCapture(0)

while True:

    # Setting up video input
    _, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Some parameters for facial detection
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(45, 45),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # For each face detected, draw a rectangle around it and put "Face" above rectangle
    for (x, y, w, h) in faces:
        cv2.putText(frame, "Face", (x, y - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show video feed
    cv2.imshow("Video", frame)
    cv2.waitKey(1)