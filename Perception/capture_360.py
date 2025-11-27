import numpy as np
import cv2
import os
import time


class Capture360:
    def __init__(self, drone_controller, save_directory):
        self.drone_controller = drone_controller
        self.save_directory = save_directory

    def capture_360_view(self, cameras):
        """
        Capture a 360-degree panoramic view by capturing images from the specified cameras (front, left, back, right).
        """
        panorama_segments = []  # List to hold the images for the panoramic view

        # Ensure cameras is always a list
        if isinstance(cameras, int):
            cameras = [cameras]  # Convert single camera ID to a list

        # Iterate over the list of cameras provided in the argument.
        # This assumes that the cameras parameter is a list with camera ids [0, 1, 2, 4]
        for cam in cameras:
            images = self.drone_controller.get_images(camera=cam)  # Capture images from the current camera
            scene_bgr = images['scene_bgr']  # Extract the scene image (BGR)
            panorama_segments.append(scene_bgr)  # Add the scene image to the panorama list

        # Optionally, concatenate images horizontally to create a panoramic image
        panoramic_image = np.concatenate(panorama_segments, axis=1)  # Horizontally stack images
        return panoramic_image, panorama_segments

    def save_panoramic_image(self, panoramic_image, panorama_segments, timestamp):
        """
        Save the panoramic image with a timestamp-based folder, and also save the individual panorama segments.
        """
        # Create a directory with the timestamp as the folder name
        folder_path = os.path.join(self.save_directory, f"panorama_{timestamp}")
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Save the panoramic image in the folder
        panoramic_image_path = os.path.join(folder_path, "panoramic_image.png")
        cv2.imwrite(panoramic_image_path, panoramic_image)

        # Save the panorama segments (individual images) in the folder
        for idx, segment in enumerate(panorama_segments):
            segment_path = os.path.join(folder_path, f"segment_{idx}.png")
            cv2.imwrite(segment_path, segment)

        print(f"Panoramic image and segments saved in {folder_path}")
