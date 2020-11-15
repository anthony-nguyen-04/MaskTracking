import dlib
import cv2
import numpy as np
import math
import imutils
import os
from pathlib import Path

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

def closestFaceNoPredictor(faces):

    closestFaceImage = faces[0]

    for face in faces:
        (h, w) = face.shape[:2]
        (closestH, closestW) = closestFaceImage.shape[:2]

        if ((h * w) > (closestH * closestW)):
            closestFaceImage = face

    return closestFaceImage

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
    maskStretched = cv2.resize(mask, (horizontalFaceWidth+20, verticalFaceHeight+15))

    return maskStretched

def imagedOverlayered(image, mask, shape):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA).copy()
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2RGBA).copy()

    maskHeight, maskWidth, _ = mask.shape

    # gets points from facial landmarks to use as border for mask
    xShift = shape[2][0] - 5 - 10
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

def maskOverFace(name, face, detector, predictor, faceNet):
    path = Path(__file__).parent.absolute().parent.absolute()
    path = os.path.sep.join([str(path), "MaskedImages"])

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

    #######

    # setting up more efficient facial recognition
    # to crop maskedFace image to only include face
    (h, w) = face.shape[:2]
    blob = cv2.dnn.blobFromImage(face, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []

    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.85:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of the frame
            # additionally, expand bounding-box area to ensure all of face is within box
            (startX, startY) = (max(0, startX-50), max(0, startY-75))
            (endX, endY) = (min(w - 1, endX+50), min(h - 1, endY+20))

            # extract the face ROI
            faceTrimmed = maskedFace[startY:endY, startX:endX]
            faces.append(faceTrimmed)

    #######

    # resizes maskedFace to 800x800
    #maskedFace = cv2.resize(maskedFace, (800, 800))
    #cv2.imwrite(str(os.path.sep.join([path, str(name + "Masked.jpg")])), maskedFace)

    maskedFaceTrim = cv2.resize(closestFaceNoPredictor(faces), (600, 600))
    cv2.imwrite(str(os.path.sep.join([path, str(name + "Masked.jpg")])), maskedFaceTrim)

