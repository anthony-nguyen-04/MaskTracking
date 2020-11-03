import cv2, dlib, os
import MaskOverlapIndividual

def main():
    # detects faces
    detector = dlib.get_frontal_face_detector()

    # detects facial landmarks from faces detected by Detector
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    unmaskedPath = "E:\\Coding\\MaskTracking\\MaskTracking\\UnmaskedImages"

    # gets all the files within the UnmaskedFolder
    fileNamesArray = os.listdir(unmaskedPath)

    for name in fileNamesArray:
        strippedName = name.split(".")[0]
        file = cv2.imread(str(unmaskedPath + "\\" + name))

        try:
            MaskOverlapIndividual.maskOverFace(strippedName, file, detector, predictor)
        except ZeroDivisionError:
            print("skipped: " + name)
            pass

if __name__ == "__main__":
    main()