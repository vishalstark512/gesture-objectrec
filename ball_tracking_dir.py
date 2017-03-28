from collections import deque
import numpy as np
import argparse
import imutils
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the video file")
ap.add_argument("-b", "--buffer", type=int, default=32,
	help="max buffer size")

args = vars(ap.parse_args())

greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

pts = deque(maxlen=args["buffer"])
counter = 0
(dX, dY)  = (0, 0)
direction = ""

if not args.get("video", False):
	camera = cv2.VideoCapture(0)

else:
	camera = cv2.VideoCapture(args["video"])

while True:

	(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		break

	frame = imutils.resize(frame,width=600)
	blurred = cv2.GaussianBlur(frame, (11,11), 0)
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


	mask = cv2.inRange(hsv, greenLower, greenUpper) 
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)


	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
	center = None

	
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
 
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)
			pts.appendleft(center)

	for i in np.arange(1, len(pts)):
		
		if pts[i-1] is None or pts[i] is None:
			continue

		if counter >= 10 and i==1 and pts[-10] is not None:
			
			dx = pts[-10][0] - pts[i][0]
			dY = pts[-10][1] - pts[i][1]
			(dirX, dirY) = ("", "")

			if np.abs(dX) > 20:
				dirX = "East" if np.sign(dX) == 1 else "West"
 
			# ensure there is significant movement in the
			# y-direction
			if np.abs(dY) > 20:
				dirY = "North" if np.sign(dY) == 1 else "South"
 
			# handle when both directions are non-empty
			if dirX != "" and dirYd != "":
				direction = "{}-{}".format(dirY, dirX)
 
			# otherwise, only one direction is non-empty
			else:
				direction = dirX if dirX != "" else dirY


		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
 
	# show the movement deltas and the direction of movement on
	# the frame
	cv2.putText(frame, direction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 3)
	cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
		(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
	# show the frame to our screen and increment the frame counter
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	counter += 1
 
	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		break
 
# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
