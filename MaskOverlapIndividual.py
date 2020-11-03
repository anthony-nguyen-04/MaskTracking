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

def closestFace(gray, faces, predictor):
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

    # mask.png from folder
    mask = cv2.imread("Mask.png", cv2.IMREAD_UNCHANGED)

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

    # gets points from facial landmarks to use as border for mask
    xShift = shape[2][0] - 5
    yShift = shape[29][1]

    # makes mask.png's background into transparent
    alpha_mask = mask[:, :, 3] / 255.0
    alpha_image = 1.0 - alpha_mask

    # overlaps the transparent mask over the face
    for c in range(0, 3):
        image[yShift:yShift + maskHeight, xShift:xShift+maskWidth, c] =\
            (alpha_mask * mask[:, :, c] + alpha_image * image[yShift:yShift +
                                                                     maskHeight, xShift:xShift+maskWidth, c])

    return image

def maskOverFace(name, face, detector, predictor):
    path = "E:\\Coding\\MaskTracking\\MaskTracking\\MaskedImages\\"

    # grayscaling current frame for analysis
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    # analyzing each face for facial landmarking
    faces = detector(gray, 1)
    (closestFaceBox, shape) = closestFace(gray, faces, predictor)

    # Calculates general area of the center of each eye
    rightEye = ( int((shape[37][0] + shape[40][0])/2) , int((shape[37][1] + shape[40][1])/2))
    leftEye = ( int((shape[43][0] + shape[46][0])/2) , int((shape[43][1] + shape[46][1])/2))

    # Calculates how many degrees image must be turn to the have be horizontally-straight
    degrees = math.atan( (rightEye[1] - leftEye[1]) / (rightEye[0] - leftEye[0]) ) * (180 / (math.pi))

    # rotates and stretches mask to fit the face
    maskRotated = maskRotateTransparent(degrees)
    maskStretched = maskStretch(maskRotated, shape)

    # puts adjusted mask over face
    maskedFace = imagedOverlayered(face, maskStretched, shape)

    # resizes maskedFace to 800x800
    maskedFace = cv2.resize(maskedFace, (800, 800))

    cv2.imwrite(str(path + name + "Masked.jpg"), maskedFace)

