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
	# to the format (x, y, w, h) that opencv uses

	x = face.left()
	y = face.top()
	w = face.right() - x
	h = face.bottom() - y

	return (x, y, w, h)

def closestFace(gray, faces):
    # In the situation that there are multiple faces in an image, it uses
    # the closest image. Calculates which face through the biggest bounding box

    closestFaceBox = (0, 0, 0, 0)
    shapeClosest = np.zeros((68, 2), dtype="int")

    for face in faces:

        # gets the facial landmark coordinates
        shape = predictor(gray, face)
        shape = shapeToNP(shape)

        (x, y, w, h) = rectangleToBox(face)

        # If a face is "closer" than the other faces in an image,
        # store it for output

        if ((w * h) > (closestFaceBox[2] * closestFaceBox[3])):
            closestFaceBox = (x, y, w, h)
            shapeClosest = shape

    return (closestFaceBox, shapeClosest)

def maskRotateTransparent(degrees):
    # Mask.png is originally transparent, but after rotating it to fit the person,
    # it produces a black background. This makes the black background into transparent.

    maskRotated = imutils.rotate_bound(mask, degrees)

    tmp = cv2.cvtColor(maskRotated, cv2.COLOR_BGR2GRAY)
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r, _ = cv2.split(maskRotated)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)

    return maskRotated

def maskStretch(mask, shape):
    # Calculates how the distance between left-right check and nose-chin.
    # Stretches the mask to fit those dimensions.

    horizontalFaceWidth = (shape[14][0] - shape[2][0])
    verticalFaceHeight = shape[8][1] - shape[29][1]
    maskStretched = cv2.resize(mask, (horizontalFaceWidth, verticalFaceHeight))

    return maskStretched

def imagedOverlayered(image, mask, shape):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).copy()
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA).copy()

    maskHeight, maskWidth, _ = mask.shape

    xShift = shape[2][0] - 5
    yShift = shape[29][1]

    alpha_mask = mask[:, :, 3] / 255.0
    alpha_image = 1.0 - alpha_mask

    for c in range(0, 3):
        image[yShift:yShift + maskHeight, xShift:xShift+maskWidth, c] =\
            (alpha_mask * mask[:, :, c] + alpha_image * image[yShift:yShift +
                                                                     maskHeight, xShift:xShift+maskWidth, c])

    return image

################################################

# loads images
face = cv2.imread("UnmaskedImages/anthony.jpg")
#face = cv2.resize(face, (1000, 1200))
mask = cv2.imread("Mask.png", cv2.IMREAD_UNCHANGED)

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

faceOld = face.copy()

# Calculates general area of the center of each eye
rightEye = ( int((shape[37][0] + shape[40][0])/2) , int((shape[37][1] + shape[40][1])/2))
leftEye = ( int((shape[43][0] + shape[46][0])/2) , int((shape[43][1] + shape[46][1])/2))

# just drawing some lines of the triangle
cv2.line(face, (rightEye), (leftEye[0], rightEye[1]), (0, 255, 255), 5)
cv2.line(face, (leftEye[0], rightEye[1]), leftEye, (0, 255, 255), 5)
cv2.line(face, rightEye, leftEye, (255, 0, 0), 5)

# Calculates how many degrees image must be turn to the have be horizontally-straight
degrees = math.atan( (rightEye[1] - leftEye[1]) / (rightEye[0] - leftEye[0]) ) * (180 / (math.pi))
#rotated = imutils.rotate_bound(face, degrees)

maskRotated = maskRotateTransparent(degrees)
maskStretched = maskStretch(maskRotated, shape)

cv2.line(face, (shape[8][0], shape[8][1]), (shape[29][0], shape[29][1]), (255, 0, 255), 5)
cv2.line(face, (shape[2][0], shape[2][1]), (shape[14][0], shape[14][1]), (255, 0, 255), 5)


print(degrees)

while True:

    key = cv2.waitKey(1)

    cv2.imshow("video", face)
    cv2.imshow("old pics", faceOld)
    cv2.imshow("rotated", maskStretched)
    cv2.imshow("OG", imagedOverlayered(face, maskStretched, shape))

    if key == ord('q'):
        break

cv2.imwrite("dot on face.jpg", faceOld)
cv2.imwrite("lines on face.jpg", face)
cv2.imwrite("mask on face.jpg", imagedOverlayered(face, maskStretched, shape))
cv2.imwrite("mask on faceOld.jpg", imagedOverlayered(cv2.imread("UnmaskedImages/anthony.jpg"), maskStretched, shape))

