# ==============================
# WEEK 7: OBJECT SEGMENTATION
# EXPERIMENT 8: PERFORM OBJECT SEGMENTATION USING HIERARCHICAL BASED METHODS
# ==============================

# Step 1: Import necessary libraries
import cv2
import numpy as np

# ---------------------------------------------------
# CODE EXPLANATION
# ---------------------------------------------------

# Why do we import cv2?
# OpenCV (cv2) is used for image processing operations
# such as reading images, thresholding, contour detection,
# and displaying results.

# Why do we import NumPy?
# NumPy is used for numerical operations and handling
# image arrays.

# Step 2: Read the input image
image = cv2.imread("image.jpg")

# Why use imread()?
# imread() loads the image from the specified file path
# into memory.

# Step 3: Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Why convert to grayscale?
# Image segmentation becomes easier because grayscale
# images have only one intensity channel instead of
# three color channels.

# Step 4: Apply binary thresholding
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Why apply thresholding?
# Thresholding separates the foreground object from
# the background by converting the image into
# black and white.

# Step 5: Find contours using hierarchical retrieval
contours, hierarchy = cv2.findContours(
    thresh,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
)

# Why use findContours()?
# It detects the boundaries of objects present
# in the binary image.

# Why use RETR_TREE?
# RETR_TREE retrieves all contours and organizes them
# into a hierarchical parent-child structure.

# Why use CHAIN_APPROX_SIMPLE?
# It stores only important contour points,
# reducing memory usage.

# Step 6: Draw detected contours
cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

# Why draw contours?
# To visualize the segmented objects by drawing
# green boundaries around them.

# Step 7: Display segmented image
cv2.imshow("Hierarchical Object Segmentation", image)

# Why use imshow()?
# It displays the segmented image.

# Step 8: Wait for a key press
cv2.waitKey(0)

# Why waitKey(0)?
# Keeps the image window open until a key is pressed.

# Step 9: Close all windows
cv2.destroyAllWindows()

# Why destroyAllWindows()?
# It closes all OpenCV windows after execution.

# ---------------------------------------------------
# SAMPLE OUTPUT
# ---------------------------------------------------

# A window titled:
#
# Hierarchical Object Segmentation
#
# appears showing the original image with
# green contour boundaries drawn around
# all detected objects.
#
# Parent and child contours are retrieved
# using the hierarchical tree structure.
#
# Example:
#
# +---------------------------+
# |       ________            |
# |      /        \           |
# |     |  Object |           |
# |      \________/           |
# |                           |
# |   Green contour drawn     |
# +---------------------------+
#
# Note:
# The output depends on the input image.
# Different images produce different contours.