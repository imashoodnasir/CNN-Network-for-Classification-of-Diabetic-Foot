import os
import numpy as np
import cv2
from augmentations import augment_image  # Assuming augment_image is in augmentations.py
from process_images import process_all_images  # Assuming process_all_images is in process_images.py

def main():
    input_folder = "input_images"  # Replace with your input folder path
    output_folder = "output_images"  # Replace with your output folder path

    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process all images
    print("Starting image processing...")
    process_all_images(input_folder, output_folder)
    print("Image processing completed.")

if __name__ == "__main__":
    main()