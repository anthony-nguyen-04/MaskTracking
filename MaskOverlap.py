import dlib
import cv2
import numpy as np
import math
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

face = cv2.imread("obama.jpg")
mask = cv2.imread("MaskPNG.png")

# detects faces
detector = dlib.get_frontal_face_detector()

# detects facial landmarks from faces detected by Detector
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grayscaling current frame for analysis
gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

# analyzing each face for facial landmarking
faces = detector(gray, 1)

(closestFaceBox, shape) = closestFace(gray, faces)

(x, y, w, h) = closestFaceBox

cv2.rectangle(face, (x, y), (x + w, y + h), (0, 255, 0), 2)

for (x, y) in shape:
    cv2.circle(face, (x, y), 4, (0, 0, 255), -1)

rightEye = ( int((shape[37][0] + shape[40][0])/2) , int((shape[37][1] + shape[40][1])/2))
leftEye = ( int((shape[43][0] + shape[46][0])/2) , int((shape[43][1] + shape[46][1])/2))

cv2.line(face, rightEye, leftEye, (255, 0, 0), 2)

degrees = 90 + math.atan( (rightEye[0] - leftEye[0]) / (rightEye[1] - leftEye[1]) ) * 180 / (math.pi)

rotated = imutils.rotate_bound(face, degrees)

print(degrees)

while True:

    key = cv2.waitKey(1)

    cv2.imshow("video", face)
    cv2.imshow("rotated", rotated)

    if key == ord('q'):
        break


