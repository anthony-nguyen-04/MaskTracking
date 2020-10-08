import cv2

video = cv2.VideoCapture(0)

while True:
    _, frame = video.read()
    cv2.imshow("video", frame)
    cv2.waitKey(1)