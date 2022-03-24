# AR Tag Detection and Tracking 
<p>This project involves the detection of a custom AR tag which can be used for obtaining a point of reference in the real world. Localization is very important for most robot applications and is especially so in applications such as augmented reality.</p>

There are two stages in this project:
1. Detection - Finding the AR tag from the given image sequence
2. Tracking - Keeping the tag "in view" throughout the sequence and processing image processing operations based on the tag's orientation and position.

## Detection
The task here was to to detect the tag in every frame of the video . To do this, the background in the video had to ve eliminated and FFT (Fast Fourier Transform) was later applied to this to find the edges.
Once we get the image of the AR tag separated out, we need to decode the tag to know the tag orientation and ID. This is done by decomposing the tag into an 8x8 grid and analyzing the inner 4x4 grid.

## Tracking 
Once we have the four corners of the tag, we can perform homography estimation to superimpose an image onto the tag. Here in our example, we have placed an image of Testudo and a cube over the AR tag.

## Instructions
Following are the instructions required to run the script:
- When the program is executed, you will be prompted to enter a 0 or 1: 0 gives the cube visualization and 1 gives the testudo superimposition
- Please press any key to shut the program down
- Libraries used: cv2, numpy, matplotlib, scipy, os and copy

THANK YOU!
