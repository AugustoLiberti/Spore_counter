import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Function to count fluorescent dots
def count_fluorescent_dots(image):
    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the range for the fluorescent green color in the HSV color space
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])

    # Create a mask for the fluorescent green color
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Perform morphological operations to remove small noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

    # Count the number of contours
    dot_count = len(contours)

    return contour_image, dot_count

# Streamlit app
def main():
    st.title('Fluorescent Dots Counter')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file as an image
        image = Image.open(uploaded_file)
        image = np.array(image)

        # Process the image to count fluorescent dots
        contour_image, dot_count = count_fluorescent_dots(image)

        # Display the images and the count result
        st.image(image, caption='Original Image', use_column_width=True)
        st.image(contour_image, caption=f'Fluorescent Dots Count: {dot_count}', use_column_width=True)
        st.write(f'Total Fluorescent Dots Count: {dot_count}')

if __name__ == "__main__":
    main()
