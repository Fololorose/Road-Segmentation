import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def generate_random_seed_map(rows, cols, seed_density):
    # Generate a binary seed map based on the seed density
    seed_map = np.random.rand(rows, cols) < seed_density
    return seed_map.astype(np.int32)

def regionGrowing(inputImage, seedMap, threshold, connectivity):
    # Initialize color map and set the starting color index
    rows, cols = inputImage.shape
    colorMap = np.zeros((rows, cols, 3), dtype=np.uint8)
    currentColorIndex = 0

    # Define neighbor offsets for 4-connectivity and 8-connectivity
    if connectivity == 4:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    elif connectivity == 8:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        raise ValueError("Connectivity must be 4 or 8")

    # Generate random colors for regions
    colors = np.random.randint(0, 256, size=(100, 3), dtype=np.uint8)

    # Iterate over the seed map to grow regions
    for r in range(rows):
        for c in range(cols):
            if seedMap[r, c] == 1 and colorMap[r, c].sum() == 0:
                # Start a new region
                regionQueue = [(r, c)]
                regionColor = colors[currentColorIndex]
                colorMap[r, c] = regionColor

                # Perform BFS to grow the region
                while regionQueue:
                    x, y = regionQueue.pop(0)
                    for dx, dy in neighbors:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if (colorMap[nx, ny] == [0, 0, 0]).all() and abs(int(inputImage[nx, ny]) - int(inputImage[x, y])) <= threshold:
                                colorMap[nx, ny] = regionColor
                                regionQueue.append((nx, ny))

                # Move to the next color index
                currentColorIndex = (currentColorIndex + 1) % len(colors)

    return colorMap

try:
    # Load the color image
    image_path = './RoadSegmentationDataset_TrainingData/images/2.png'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found at path: {image_path}")

    # Convert image to grayscale
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale image for noise reduction
    blurred_grayscale_img = cv2.GaussianBlur(grayscale_img, (5, 5), 0)

    # ============================== Otsu Thresholding to Find The Sky ==============================

    # Apply Otsu's thresholding to the blurred grayscale image to segment the sky
    ret, otsu = cv2.threshold(blurred_grayscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert the Otsu result to get the sky as white
    inverted_otsu = cv2.bitwise_not(otsu)

    # Define a structuring element for morphological operations
    sE = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply closing to fill small holes in the sky region
    inverted_closed_otsu = cv2.morphologyEx(inverted_otsu, cv2.MORPH_CLOSE, sE, iterations=20)

    # Invert back to get the closed Otsu result
    closed_otsu = cv2.bitwise_not(inverted_closed_otsu)
    
    # ============================== Convert Specified Gray (Road) to White ==============================
    # Convert the original image to HSV color space
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for the gray color (road)
    light_gray = np.array([0, 0, 60])
    dark_gray = np.array([180, 55, 200])

    # Create a mask for the gray color range
    mask_gray = cv2.inRange(hsv_img, light_gray, dark_gray)

    # Combine the Otsu result and the gray mask to separate sky and road
    img_roadside = cv2.bitwise_or(otsu, mask_gray)

    # ============================== Remove Noise on Road ==============================
    # Get the dimensions of the image
    height, width = img_roadside.shape

    # Split the image into top and bottom halves
    top_half = img_roadside[:height//2, :]
    bottom_half = img_roadside[height//2:, :]

    # Apply closing to the bottom half to remove noise
    closed_bottom_half = cv2.morphologyEx(bottom_half, cv2.MORPH_CLOSE, sE, iterations=1)

    # Combine the processed bottom half with the original top half
    clean_img_roadside = np.vstack((top_half, closed_bottom_half))

    # ============================== Apply Closing to Connect the Roadside Region==============================    
    # Invert the cleaned image for further processing
    inverted_clean_img_roadside = cv2.bitwise_not(clean_img_roadside)

    # Apply closing to smooth out the black regions (non-sky, non-road)
    closed_img = cv2.morphologyEx(inverted_clean_img_roadside, cv2.MORPH_CLOSE, sE, iterations=20)

    # Replace non-sky and non-road regions with the original grayscale values
    grayscale_img_roadside = np.where(closed_img == 255, grayscale_img, closed_img)

    # ============================== Remove Video Date & Time ==============================    
    # Modify the bottom-right region of the image to gray, stopping 10 pixels before the end
    start_x, start_y = 650, 630
    end_x, end_y = width - 10, height - 10
    modified_grayscale_img = grayscale_img.copy()
    modified_grayscale_img[start_y:end_y, start_x:end_x] = 85

    # ============================== Otsu Thresholding on Modified Image ==============================
    # Apply Gaussian blur to the modified grayscale image
    blurred_grayscale_img = cv2.GaussianBlur(modified_grayscale_img, (5, 5), 0)

    # Apply Otsu's thresholding to the blurred grayscale image
    ret, otsu = cv2.threshold(blurred_grayscale_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine the Otsu result and grayscale image without sky and road
    input_img = cv2.bitwise_or(otsu, grayscale_img_roadside)

    # ============================== Region Growing ==============================
    # Generate a random seed map for region growing
    rows, cols = input_img.shape
    seed_density = 0.1
    seed_map = generate_random_seed_map(rows, cols, seed_density)

    # Define the threshold and connectivity for region growing
    threshold = 5
    connectivity = 4

    # Check if the threshold exceeds the allowed value
    if threshold > 5:
        raise ValueError("Threshold cannot be more than 5")

    # Perform region growing on the input image
    colorMap = regionGrowing(input_img, seed_map, threshold, connectivity)
    
    # ============================== Change the Segmented Road Color ==============================
    # Convert the color map to grayscale for analysis
    gray_colorMap = cv2.cvtColor(colorMap, cv2.COLOR_BGR2GRAY)

    # Calculate the most frequent color in the color map (assumed to be the road)
    (unique, counts) = np.unique(gray_colorMap, return_counts=True)
    frequent_color = unique[np.argmax(counts)]

    # Define the target color for the road (RGB: 61, 61, 245)
    target_color = [61, 61, 245]

    # Change the most frequent color in the color map to the target color
    colorMap[gray_colorMap == frequent_color] = target_color

    # ============================== Combine Region Growing Result with Otsu ==============================    
    # Expand the Otsu result to three channels
    otsu_color = np.stack([otsu] * 3, axis=-1)
    closed_otsu_color = np.stack([closed_otsu] * 3, axis=-1)

    # Replace the sky and lane markings with white in the result
    otsu_expanded = np.stack([otsu] * 3, axis=-1)
    result_with_white_sky = np.where(otsu_color == 255, otsu_expanded, colorMap)

    # Replace the white sky color with the original sky color
    result = np.where(closed_otsu_color == 255, image, result_with_white_sky)

    # ============================== Change the Land Marking Colour ==============================
    # Change the lane markings to a specific color (RGB: 221, 255, 51)
    white_pixels_mask = (result == [255, 255, 255]).all(axis=-1)
    result[white_pixels_mask] = [221, 255, 51]

    # ============================== Plotting ==============================
    # Display the final result
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title('Final Result')
    plt.imshow(result)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Extract the base name and extension
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    
    # Ensure the output directory exists
    output_dir = './Output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the output file path
    output_file_path = os.path.join(output_dir, f'Output_{name}{ext}')
    
    # Save the image
    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_file_path, result_rgb)

except FileNotFoundError as e:
    print(f"Error: {e}")
except ValueError as e:
    print(f"Error: {e}")
