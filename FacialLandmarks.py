import dlib
import cv2
import numpy as np
import imutils

def shapeToNP(shape):
    #makes numpy array of (x,y) coordinates for the 68 facial landmarks

    coords = np.zeros((68, 2), dtype="int")
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def rectangleToBox(face):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV

	x = face.left()
	y = face.top()
	w = face.right() - x
	h = face.bottom() - y

	return (x, y, w, h)

def closestFace(gray, faces):
    closestFaceBox = (0, 0, 0, 0)
    shapeClosest = np.zeros((68, 2), dtype="int")

    for face in faces:
        shape = predictor(gray, face)
        shape = shapeToNP(shape)

        (x, y, w, h) = rectangleToBox(face)

        if ((w * h) > (closestFaceBox[2] * closestFaceBox[3])):
            closestFaceBox = (x, y, w, h)
            shapeClosest = shape

    return (closestFaceBox, shapeClosest)

################################################

video = cv2.VideoCapture(0)

# detects faces
detector = dlib.get_frontal_face_detector()

# detects facial landmarks from faces detected by Detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:

    _, frame = video.read()
    height, width, _ = frame.shape

    whiteScreen = np.ones((height, width, 3))

    # grayscaling current frame for analysis
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # analyzing each face for facial landmarking
    faces = detector(gray, 1)

    #for face in faces:
    #    shape = predictor(gray, face)
    #    shape = shapeToNP(shape)
    #
    #    (x, y, w, h) = rectangleToBox(face)
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #
    #    for (x, y) in shape:
    #        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
    #        cv2.circle(whiteScreen, (x, y), 2, (0, 0, 255), -1)

    #print(closestFace(gray, faces))

    (closestFaceBox, shape) = closestFace(gray, faces)

    (x, y, w, h) = closestFaceBox

    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    for (x, y) in shape:
        cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
        cv2.circle(whiteScreen, (x, y), 2, (0, 0, 255), -1)

    cv2.imshow("video", frame)
    cv2.imshow("white", whiteScreen)
    cv2.waitKey(1)


