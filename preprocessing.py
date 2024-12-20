import numpy as np
import cv2

def right_flip(image):
    """
    Perform right-flip operation on the image.
    """
    return np.flip(image, axis=1)

def left_flip(image):
    """
    Perform left-flip operation on the image.
    """
    return np.flip(image, axis=0)

def rotate_15(image):
    """
    Perform 15-degree rotation operation on the image.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return rotated

def zoom(image, zoom_factor):
    """
    Perform zoom operation on the image with the specified zoom factor.
    """
    h, w = image.shape[:2]
    zoomed_h, zoomed_w = int(h * zoom_factor), int(w * zoom_factor)

    # Resize the image to the zoomed dimensions
    resized = cv2.resize(image, (zoomed_w, zoomed_h), interpolation=cv2.INTER_LINEAR)

    # Crop back to the original size
    start_h = (zoomed_h - h) // 2
    start_w = (zoomed_w - w) // 2
    cropped = resized[start_h:start_h + h, start_w:start_w + w]

    return cropped

def augment_image(image, zoom_factor=1.2):
    """
    Apply all augmentations to the input image.
    """
    augmented_images = {
        "right_flip": right_flip(image),
        "left_flip": left_flip(image),
        "rotate_15": rotate_15(image),
        "zoom": zoom(image, zoom_factor)
    }
    return augmented_images

# Example usage
if __name__ == "__main__":
    # Load an example image (replace 'example.jpg' with your image path)
    image_path = 'example.jpg'
    image = cv2.imread(image_path)

    # Perform augmentations
    augmented_images = augment_image(image)

    # Save or display augmented images
    for aug_type, aug_image in augmented_images.items():
        cv2.imwrite(f"augmented_{aug_type}.jpg", aug_image)
