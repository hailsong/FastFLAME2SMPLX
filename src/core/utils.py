import cv2
import numpy as np

def add_wireframe(source, wireframe, color=(0, 255, 255), alpha=0.7):
    """
    Overlay a wireframe (white on black) on top of a BGR source image.

    Args:
        source (np.ndarray): Source image in BGR format.
        wireframe (np.ndarray): Wireframe image (white lines on black bg), same size as source.
        color (tuple): BGR color for the wireframe overlay (default: yellow).
        alpha (float): Blending factor for the wireframe.

    Returns:
        np.ndarray: Source image with wireframe overlay.
    """
    if source.shape[:2] != wireframe.shape[:2]:
        raise ValueError("Source and wireframe images must have the same height and width.")

    # Convert wireframe to grayscale if needed
    if wireframe.ndim == 3:
        wireframe_gray = cv2.cvtColor(wireframe, cv2.COLOR_BGR2GRAY)
    else:
        wireframe_gray = wireframe

    # Create a mask where wireframe lines exist (i.e., where pixel is white)
    mask = wireframe_gray > 127  # boolean mask

    # Create color overlay with same shape as source
    overlay = source.copy()
    overlay[mask] = color

    # Blend the overlay only at wireframe locations
    blended = source.copy()
    blended[mask] = cv2.addWeighted(source[mask], 1 - alpha, overlay[mask], alpha, 0)

    return blended
