import cv2
import requests
from .translations import translations

class BaseSugarcaneApp:
  def __init__(self):
    self.language = "English"
    self.model_name = "EfficientNetB0"
    self.camera_running = False
    self.current_frame = None
    self.frame_frozen = False
    self.cap = None
    self.cam_prediction = False
    self.server_url = "http://localhost:5000/predict"

    self.translations = translations

  def tr(self, key):
    return self.translations.get(key, {}).get(
      self.language,
      self.translations.get(key, {}).get("English", key)
    )
  
  def start_camera(self):
    self.cap = cv2.VideoCapture(0)
    self.camera_running = True
    self.frame_frozen = False
    return self.cap.isOpened()

  def stop_camera(self):
    self.camera_running = False
    if self.cap:
      self.cap.release()
    self.cap = None
    #self.current_frame = None
    #self.frame_frozen = False

  def read_frame(self):
    if self.camera_running and not self.frame_frozen and self.cap and self.cap.isOpened():
      ret, frame = self.cap.read()
      if ret:
        self.current_frame = frame
        return frame
  
  def predict(self):
    self.frame_frozen = True
    if self.camera_running:
      self.cam_prediction = True
      self.stop_camera()
    if self.current_frame is None:
      raise ValueError("No image or webcam frame available")

    success, encoded = cv2.imencode(".jpg", self.current_frame)
    if not success:
      raise ValueError("Failed to encode image")

    files = {"image": ("image.jpg", encoded.tobytes(), "image/jpeg")}
    data = {"model": self.model_name}

    response = requests.post(self.server_url, files=files, data=data)
    response.raise_for_status()

    label = response.headers.get("Prediction-Label", "Unknown")
    conf = float(response.headers.get("Confidence", "0"))

    # Return: prediction label, confidence, Grad-CAM image bytes
    return label, conf, response.content