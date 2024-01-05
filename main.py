# cv2 collaborates with a NumPy library
import cv2
import time

# 0 for laptop main camera and 1 for usb attached/secondary camera
video = cv2.VideoCapture(0)
time.sleep(1)

while True:
	check, frame = video.read()
	cv2.imshow("My video", frame)

	key = cv2.waitKey(1)

	if key == ord("q"):
		break

video.release()