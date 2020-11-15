import cv2, dlib, os
from CaffeMaskDetection import MaskOverlapIndividual
from pathlib import Path

def main():
    # detects faces for facial structure function
    # only using this because "predictor" only works with "detector"
    detector = dlib.get_frontal_face_detector()

    # detects facial landmarks from faces detected by Detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    path = Path(__file__).parent.absolute().parent.absolute()

    # detects face
    prototxtPath = os.path.sep.join([str(path), "CaffeMaskDetection\\deploy.prototxt"])
    weightsPath = os.path.sep.join([str(path), "CaffeMaskDetection\\res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # gets all the files within the UnmaskedFolder
    fileNamesArray = os.listdir(str(os.path.sep.join([str(path), "UnmaskedImages"])))

    for name in fileNamesArray:
        strippedName = name.split(".")[0]
        imageLocation = str(os.path.sep.join([str(path), 'UnmaskedImages', str(name)]))
        file = cv2.imread(imageLocation)

        try:
            MaskOverlapIndividual.maskOverFace(strippedName, file, detector, predictor, faceNet)
        except ZeroDivisionError:
            print("skipped: " + name)
            pass

if __name__ == "__main__":
    main()