# cv2 collaborates with a NumPy library
import cv2
import time

# 0 for laptop main camera and 1 for usb attached/secondary camera
video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None

while True:
	check, frame = video.read()
	# cvtColor() is used for color conversion.
	# cvtColor(input_image, flag) where flag determines the type of conversion
	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray_frame_gau = cv2.GaussianBlur(gray_frame, (21, 21), 0)

	if first_frame is None:
		first_frame = gray_frame_gau

	delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

	# If pixel value is greater than a threshold value, it is assigned one value (maybe white)
	# else, it is assigned another value (maybe black)
	# First arg is the source image, which should be a grayscale image
	# Second arg is the threshold value used to classify the pixel values
	# Third arg is the maxval which represents the value to be given if pixel value is more than
	# (or sometimes less than) the threshold values
	thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]

	# Dilation is a process that adds pixels to the boundaries of objects in an image.
	# Dilation fills in small holes or gaps to join broken segments of lines, and to make objects more uniform
	# In dilation, the value of a pixel is changed to the maximum value of its neighbouring pixels
	dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

	# imshow is used to display an image in a window
	# First arg is window name and second arg is the image
	cv2.imshow("My video", dil_frame)

	contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in contours:
		if cv2.contourArea(contour) < 5000:
			continue
		x, y, w, h = cv2.boundingRect(contour)
		cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

	cv2.imshow("Video", frame)
	# waitKey function waits for specified milliseconds for any keyboard event
	key = cv2.waitKey(1)

	if key == ord("q"):
		break

video.release()