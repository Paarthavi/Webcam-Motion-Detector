# cv2 collaborates with a NumPy library
import cv2
import time
from emailing import send_email
import glob
import os
from threading import Thread

# 0 for laptop main camera and 1 for usb attached/secondary camera
video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
status_list = []
count = 1

def clean_folder():
	print("clean_folder function started")
	images = glob.glob("images/*.png")
	for image in images:
		os.remove(image)
	print("clean_folder function ended")

while True:
	status = 0
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
		rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
		if rectangle.any():
			status = 1
			# To store images
			cv2.imwrite(f"images/{count}.png", frame)
			count = count + 1
			all_images = glob.glob("images/*.png")
			index = int(len(all_images) / 2)
			image_with_object = all_images[index]

	status_list.append(status)
	status_list = status_list[-2:]

	if status_list[0] == 1 and status_list[1] == 0:
		email_thread = Thread(target=send_email, args=(image_with_object, ))
		email_thread.daemon = True
		clean_thread = Thread(target=clean_folder)
		clean_thread.daemon = True

		email_thread.start()

	print(status_list)

	cv2.imshow("Video", frame)
	# waitKey function waits for specified milliseconds for any keyboard event
	key = cv2.waitKey(1)

	if key == ord("q"):
		break

clean_thread.start()

video.release()