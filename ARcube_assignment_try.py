#!/usr/bin/python3

import cv2
import numpy as np
import pyrealsense2 as rs

# Load camera calibration data
calibration_file = "calibrationdata.tar.gz"
# Extract camera matrix and distortion coefficients from the provided calibration data
camera_matrix = np.array([[663.584552, 0.000000, 312.982328],
                          [0.000000, 662.525674, 201.728966],
                          [0.000000, 0.000000, 1.000000]])

distortion_coeffs = np.array([0.167502, -0.321948, -0.011440, -0.010280, 0.000000])

# Create an OpenCV video capture object
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Load the checkerboard image and set its size (assuming a 3x4 checkerboard)
checkerboard_size = (9, 6)
objp = np.zeros((np.prod(checkerboard_size), 3), dtype=np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())

    # Convert the image to grayscale
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


        color_image = cv2.drawChessboardCorners(color_image, checkerboard_size, corners2, ret)

        # Find the rotation and translation vectors
        ret, rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, camera_matrix, distortion_coeffs)
        
        # Project 3D points to image plane
        cube_points = np.float32([[0, 0, 0], [0, 5, 0], [8, 5, 0], [8, 0, 0],
                                  [0, 0, -5], [0, 5, -5], [8, 5, -5], [8, 0, -5]])
        imgpts, jac = cv2.projectPoints(cube_points, rvecs, tvecs, camera_matrix, distortion_coeffs)

        # Draw the cube edges
        imgpts = np.int32(imgpts).reshape(-1, 2)
        color_image = cv2.drawContours(color_image, [imgpts[:4]], -1, (0, 255, 0), 3)
        for i, j in zip(range(4), range(4, 8)):
            color_image = cv2.line(color_image, tuple(imgpts[i]), tuple(imgpts[j]), (0, 255, 0), 3)
        color_image = cv2.drawContours(color_image, [imgpts[4:]], -1, (0,255, 0), 3)

    # Display the resulting frame
    cv2.imshow('AR Cube', color_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
pipeline.stop()
cv2.destroyAllWindows()
