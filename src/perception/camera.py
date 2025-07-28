from PIL import Image
import numpy as np
import pillow_heif
import cv2

class Camera:
    def capture(self):
        raise NotImplementedError()

class HEICCamera(Camera):
    def __init__(self, image_path):
        self.image_path = image_path
    
    def capture(self):
        heif_file = pillow_heif.read_heif(self.image_path)
        image_temp = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw"
        )
        return np.array(image_temp)
    
class WebcamCamera(Camera):
    def __init__(self, cam_index=0):
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(cam_index)
    
    def capture(self):
        ret, frame = self.cap.read()
        return frame if ret else None
    
    def release(self):
        self.cap.release()