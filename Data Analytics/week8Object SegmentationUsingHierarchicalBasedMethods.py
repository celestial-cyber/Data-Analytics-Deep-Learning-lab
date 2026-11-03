# THEORY:
# Image segmentation is the process of dividing an image into
# multiple regions or segments.
#
# Hierarchical-based segmentation groups similar pixels into
# regions based on their characteristics.
#
# In this program, Felzenszwalb's Graph-Based Segmentation
# algorithm is used to identify different regions in an image.
#
# The segmented regions are displayed using different average
# colors to distinguish objects and areas.

# ALGORITHM:
# 1. Import the required libraries.
# 2. Load a sample image using skimage.
# 3. Apply Felzenszwalb's graph-based segmentation algorithm.
# 4. Generate segments based on image similarity.
# 5. Convert segment labels into an RGB image using label2rgb().
# 6. Display the original image.
# 7. Display the segmented image.
# 8. Compare the original and segmented images.

import matplotlib.pyplot as plt
from skimage import data
from skimage.segmentation import felzenszwalb
from skimage.color import label2rgb

# Load sample image
image = data.astronaut()

# Apply Felzenszwalb segmentation
segments = felzenszwalb(
    image, scale=100, sigma=0.5, min_size=50
)

# Create segmented image
segmented_image = label2rgb(segments, image, kind='avg')

# Display original and segmented images
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image)
ax[1].set_title('Felzenszwalb Segmentation')
ax[1].axis('off')

plt.tight_layout()
plt.show()

# EXPECTED OUTPUT:
#
# Two images are displayed side by side.
#
# 1. Original Image:
#    Displays the original astronaut image.
#
# 2. Felzenszwalb Segmentation:
#    Displays the image divided into multiple regions.
#    Each region is represented using its average color.
#
# Similar pixels are grouped into common segments.
# IMPORTANT KEY POINTS:
#
# 1. Image Segmentation:
#    Divides an image into meaningful regions.
#
# 2. Hierarchical-Based Method:
#    Groups similar regions based on their characteristics.
#
# 3. Felzenszwalb Algorithm:
#    A graph-based image segmentation algorithm.
#
# 4. skimage:
#    Python library used for image processing.
#
# 5. data.astronaut():
#    Loads a sample astronaut image.
#
# 6. felzenszwalb():
#    Performs graph-based image segmentation.
#
# 7. scale:
#    Controls the size of segments.
#
# 8. sigma:
#    Controls Gaussian smoothing before segmentation.
#
# 9. min_size:
#    Specifies the minimum size of a segment.
#
# 10. label2rgb():
#     Converts segment labels into an RGB image.
#
# 11. kind='avg':
#     Represents each segment using its average color.
#
# 12. Matplotlib:
#     Displays the original and segmented images.







# VIVA QUESTIONS:
#
# 1. What is image segmentation?
#    Image segmentation is the process of dividing an image
#    into multiple meaningful regions.
#
# 2. Which algorithm is used in this program?
#    Felzenszwalb's Graph-Based Segmentation algorithm.
#
# 3. What is the purpose of felzenszwalb()?
#    It identifies and groups similar pixels into different
#    image segments.
#
# 4. What is label2rgb() used for?
#    It converts segmentation labels into a colored RGB image.
#
# 5. What is the purpose of min_size?
#    It specifies the minimum allowed size of a segment.
#
# 6. What is the difference between image classification
#    and segmentation?
#    Classification identifies what an image contains.
#    Segmentation divides an image into meaningful regions.

