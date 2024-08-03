# Road Segmentation with Region Growing

## Overview

This script performs road segmentation and labeling on images using a combination of various image processing techniques including Otsu thresholding, region growing, and morphological operations. The result is an annotated image with road segments highlighted.

## Dependencies

Ensure you have the following Python packages installed:

- `numpy`
- `opencv-python`
- `matplotlib`

You can install these dependencies using pip:

```bash
pip install numpy opencv-python matplotlib
Script Overview

1. generate_random_seed_map(rows, cols, seed_density)
Generates a binary seed map for the region growing algorithm based on the specified seed density.

2. regionGrowing(inputImage, seedMap, threshold, connectivity)
Performs the region growing algorithm on the input image using the provided seed map, threshold, and connectivity.

3. Main Script Execution
Loads an image from a specified path.
Converts the image to grayscale and applies Gaussian blur.
Segments the sky and road using Otsu's thresholding and color masking.
Processes the image to remove noise and modify specific regions.
Performs region growing to segment the image into different regions.
Applies color modifications to the segmented regions and combines the results.
Saves and displays the final annotated image.
Usage

Place Your Image:
Ensure the image you want to process is placed in the ./RoadSegmentationDataset_TrainingData/images/ directory and update the image_path variable in the script to point to your image.
Run the Script:
Execute the script using Python. Open a terminal and run:
bash
Copy code
python road_segmentation.py
This will process the image and display the results.
Output:
The final result will be saved in the ./Output directory with a filename prefixed with Output_. The processed image will also be displayed using matplotlib.
Troubleshooting

FileNotFoundError: Ensure the image file exists at the specified path.
ValueError: Verify that the threshold value is within the acceptable range (0-5).
