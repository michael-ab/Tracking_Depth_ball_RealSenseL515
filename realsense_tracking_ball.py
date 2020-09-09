import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs

# Author: Michael Aboulhair 

pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
profile = pipe.start(config)

frameset = pipe.wait_for_frames()
color_frame = frameset.get_color_frame()
color_init = np.asanyarray(color_frame.get_data())

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2


try:
  while True:
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    color = np.asanyarray(color_frame.get_data())
    res = color.copy()
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

    l_b = np.array([24, 133, 48])
    u_b = np.array([39, 200, 181])

    mask = cv2.inRange(hsv, l_b, u_b)
    color = cv2.bitwise_and(color, color, mask=mask)

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    ### motion detector
    d = cv2.absdiff(color_init, color)
    gray = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(th, np.ones((3, 3), np.uint8), iterations=3)
    (c, _) = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(color, c, -1, (0, 255, 0), 2)
    color_init = color

    depth = np.asanyarray(aligned_depth_frame.get_data())

    for contour in c:
        if cv2.contourArea(contour) < 1500:
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        bottomLeftCornerOfText = (x, y)

        # Crop depth data:
        depth = depth[x:x+w, y:y+h].astype(float)

        depth_crop = depth.copy()

        if depth_crop.size == 0:
          continue
        depth_res = depth_crop[depth_crop != 0]


        # Get data scale from the device and convert to meters
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_res = depth_res * depth_scale

        if depth_res.size == 0:
          continue

        dist = min(depth_res)

        cv2.rectangle(res, (x, y), (x + w, y + h), (0, 255, 0), 3)
        text = "Depth: " + str("{0:.2f}").format(dist)
        cv2.putText(res,
                    text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    lineType)


    cv2.namedWindow('RBG', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RBG', res)
    cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Depth', colorized_depth)
    cv2.namedWindow('mask', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('mask', mask)

    cv2.waitKey(1)
	
finally:
  pipe.stop()
