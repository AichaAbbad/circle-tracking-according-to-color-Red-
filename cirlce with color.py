import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True :
    _ , frame = cap.read()

    blur_frame = cv2.GaussianBlur(frame, (3,3), 3)
    hsv = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2HSV)

    # read color
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])

    mask = cv2.inRange(hsv, low_red, high_red)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    red = cv2.bitwise_and(frame, frame, mask=mask)
    gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)


    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT , 3, 500, minRadius = 10, maxRadius = 200, param1 = 100,  param2 = 60 )
    circles = np.uint16(np.around(circles))

    if circles is not None:

        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(red, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)


    cv2.imshow('frame', frame)  # Show the frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    #cv2.imshow("gray", gray)
    cv2.imshow("mask", mask)
    cv2.imshow("red", red)

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()



