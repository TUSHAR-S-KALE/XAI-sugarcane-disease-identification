import numpy as np
import cv2
import sys
import os

#Adding the parent directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from BaseApp.app import BaseSugarcaneApp

from kivy.uix.screenmanager import Screen
from kivy.properties import StringProperty
from kivy.clock import Clock
from kivy.app import App
from kivy.core.image import Image as CoreImage
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen, SlideTransition

shared_app = BaseSugarcaneApp()

class KivyMainScreen(Screen):
  model_select_text = StringProperty("Select Model:")

  def __init__(self, **kwargs):
    Screen.__init__(self, **kwargs)
    self.shared = shared_app
    self.font_path = os.path.join("assets", "fonts", "NotoSansDevanagari-Regular.ttf")
    Clock.schedule_once(self.initialize_ui)

  def initialize_ui(self, _):
    self.ids.top_spinner.bind(text=self.set_language)
    self.ids.model_spinner.bind(text=self.select_model)
    self.set_language(self.ids.top_spinner, "English")

  def set_language(self, spinner, lang):
    self.shared.language = lang
    font = self.font_path if lang != "English" else "Roboto"

    if self.shared.camera_running:
      self.ids.start_btn.text = self.shared.tr("Stop Camera")
    else:
      self.ids.start_btn.text = self.shared.tr("Start Camera")

    self.ids.start_btn.font_name = font
    self.ids.select_btn.text = self.shared.tr("Select an Image")
    self.ids.select_btn.font_name = font
    self.ids.predict_btn.text = self.shared.tr("Predict")
    self.ids.predict_btn.font_name = font

  def select_model(self, spinner, text):
    self.shared.model_name = text

  def toggle_camera(self, _button):
    if not self.shared.camera_running:
      try:
        if self.shared.start_camera():
          self.ids.start_btn.text = self.shared.tr("Stop Camera")
          Clock.schedule_interval(self.update_frame, 1/30)
        else:
          raise RuntimeError("Failed to open the camera")
      except Exception as e:
        self.show_popup("Error", str(e))
    else:
      self.shared.stop_camera()
      self.ids.image_panel.texture = None
      self.ids.start_btn.text = self.shared.tr("Start Camera")
      Clock.unschedule(self.update_frame)

  def update_frame(self, dt):
    try:
      if not self.shared.frame_frozen:
        frame = self.shared.read_frame()
      else:
        frame = self.shared.current_frame

      buf = cv2.flip(frame, 0).tobytes()
      source = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="bgr")
      source.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")

      self.ids.image_panel.texture = source
    
    except Exception as e:
      self.show_popup("Error", str(e))

  def freeze_frame(self, _, touch):
    if self.ids.image_panel.collide_point(*touch.pos) and self.shared.camera_running:
      self.shared.frame_frozen = True
      self.shared.cam_prediction = True
      self.shared.stop_camera()

  def select_image(self, _):
    layout = BoxLayout(orientation="vertical")
    home_dir = os.path.expanduser("~")
    filechooser = FileChooserIconView(path=home_dir, filters=["*.png", "*.jpg", "*.jpeg"])
    layout.add_widget(filechooser)

    button = Button(size_hint=(1, 0.1), text=self.shared.tr("Select"))
    font = self.font_path if self.shared.language != "English" else "Roboto"
    button.font_name = font

    def on_select(_):
      if filechooser.selection:
        selected_path = filechooser.selection[0]
        img = cv2.imread(selected_path)
        self.ids.image_panel.source = selected_path
        self.ids.image_panel.reload()

        self.shared.current_frame = img
        self.shared.frame_frozen = True
        popup.dismiss()

    button.bind(on_release=on_select)
    layout.add_widget(button)

    popup = Popup(title=self.shared.tr("Select an Image"), content=layout, size_hint=(0.9, 0.9))
    popup.open()

  def predict_image(self, _):
    try:
      label, conf, gradcam_bytes = self.shared.predict()

      conf_str = f"{conf:.4f}"

      #Sending result to ResultScreen
      result_screen = self.manager.get_screen("result")
      result_screen.ids.prediction_label.text = f"{label} ({conf_str})"

      #Loading gradcam into texture
      gradcam_np = cv2.imdecode(
        np.frombuffer(gradcam_bytes, np.uint8), cv2.IMREAD_COLOR
      )

      buf = cv2.flip(gradcam_np, 0).tobytes()
      texture = Texture.create(size=(gradcam_np.shape[1], gradcam_np.shape[0]), colorfmt="bgr")
      texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")

      result_screen.ids.result_image_panel.texture = texture

      #Switching to the result screen
      self.manager.transition = SlideTransition(direction="left")
      self.manager.current = "result"
      
    except Exception as e:
      self.show_popup("Error", str(e))

  def show_popup(self, title_key, message_key):
    title = self.shared.tr(title_key)
    message = self.shared.tr(message_key)
    font = self.font_path if self.shared.language != "English" else "Roboto"

    content = BoxLayout(orientation="vertical", padding=10, spacing=10)
    content.add_widget(Label(text=message, font_size=20, font_name=font))

    close_btn = Button(text=self.shared.tr("Close"), size_hint_y=None, height=40, font_name=font)
    content.add_widget(close_btn)

    popup = Popup(title=title, content=content, size_hint=(0.6, 0.3))
    close_btn.bind(on_release=popup.dismiss)
    popup.open()

class ResultScreen(Screen):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.shared = shared_app
  def go_back(self, _):
    self.manager.transition = SlideTransition(direction="right")
    self.manager.current = "main"
    if self.shared.cam_prediction:
        self.shared.frame_frozen = False
        self.shared.cam_prediction = False
        self.shared.start_camera()

class MainApp(App):
  def build(self):
    app = shared_app
    root = Builder.load_file("kivy_main.kv")

    root.get_screen("main").shared = app
    root.get_screen("result").shared = app

    return root

if __name__ == "__main__":
  MainApp().run()

