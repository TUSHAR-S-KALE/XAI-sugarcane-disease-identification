import numpy as np
import cv2
import sys
import os

#Adding the parent directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

from BaseApp.app import BaseSugarcaneApp

import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
import io

class TkinterApp(BaseSugarcaneApp):
  def __init__(self, root):
    super().__init__()
    self.root = root
    self.root.geometry("800x510")
    self.root.configure(bg="#f5f5f5")
    self.root.title(self.tr("Sugarcane Disease Classification"))
    
    self.build_frames()
    self.show_home()
  
  def build_frames(self):
    # Home Frame
    self.home_frame = tk.Frame(self.root, bg="#f5f5f5")
    tk.Label(self.home_frame, text=self.tr("Select Language"), font=("Arial", 10), bg="#f5f5f5").grid(row=0, column=0, padx=10, pady=10, sticky="e")

    self.lang_var = tk.StringVar(value=self.language)
    self.language_menu = ttk.Combobox(self.home_frame, textvariable=self.lang_var, state="readonly", width=15)
    self.language_menu['values'] = list(self.translations["Select Language"].keys())
    self.language_menu.grid(row=0, column=1, padx=5, sticky="w")
    self.language_menu.bind("<<ComboboxSelected>>", self.update_language)

    tk.Label(self.home_frame, text=self.tr("Select Model"), font=("Arial", 10), bg="#f5f5f5").grid(row=1, column=0, padx=10, pady=10, sticky="e")
    self.model_var = tk.StringVar(value=self.model_name)
    self.model_menu = ttk.Combobox(self.home_frame, textvariable=self.model_var, state="readonly", width=15)
    self.model_menu['values'] = ["EfficientNetB0", "ResNet50", "MobileNetV2"]
    self.model_menu.grid(row=1, column=1, padx=5, sticky="w")
    self.model_menu.bind("<<ComboboxSelected>>", self.select_model)

    self.canvas = tk.Canvas(self.home_frame, width=300, height=300, bg="white", bd=2, relief="ridge")
    self.canvas.grid(row=2, column=0, columnspan=2, pady=10)
    self.canvas.bind("<Button-1>", self.freeze_frame)

    btn_frame = tk.Frame(self.home_frame, bg="#f5f5f5")
    btn_frame.grid(row=3, column=0, columnspan=2)

    self.btn_webcam_toggle = tk.Button(btn_frame, text=self.tr("Start Camera"), command=self.toggle_webcam, width=20)
    self.btn_webcam_toggle.grid(row=0, column=0, columnspan=2, padx=10, pady=5)

    self.btn_select = tk.Button(btn_frame, text=self.tr("Select an Image"), command=self.select_image, width=15)
    self.btn_select.grid(row=1, column=0, padx=10, pady=5)

    self.btn_predict = tk.Button(btn_frame, text=self.tr("Predict"), command=self.run_prediction, width=15)
    self.btn_predict.grid(row=1, column=1, padx=10, pady=5)

    # Result Frame
    self.result_frame = tk.Frame(self.root, bg="#f5f5f5")
    self.result_canvas1 = tk.Label(self.result_frame)
    self.result_canvas1.grid(row=0, column=0, padx=10, pady=10)
    self.result_canvas2 = tk.Label(self.result_frame)
    self.result_canvas2.grid(row=0, column=1, padx=10, pady=10)

    self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 14), bg="#f5f5f5")
    self.result_label.grid(row=1, column=0, columnspan=2, pady=10)

    self.btn_back = tk.Button(self.result_frame, text=self.tr("Back to Main Screen"), command=self.show_home)
    self.btn_back.grid(row=2, column=0, columnspan=2)
  
  def update_language(self, event=None):
    self.language = self.lang_var.get()
    self.apply_language()
  
  def apply_language(self):
    self.root.title(self.tr("Sugarcane Disease Classification"))
    self.home_frame.winfo_children()[0].config(
      text=self.tr("Select Language")
    )
    self.home_frame.winfo_children()[2].config(
      text=self.tr("Select Model")
    )
    self.btn_webcam_toggle.config(text=self.tr("Start Camera") if not self.cap else self.tr("Stop Camera"))
    self.btn_select.config(text=self.tr("Select an Image"))
    self.btn_predict.config(text=self.tr("Predict"))


  def refresh_ui(self):
    self.show_home()

  def show_home(self):
    self.result_canvas1.image = None
    self.result_canvas2.image = None
    self.result_frame.pack_forget()
    self.home_frame.pack()

    self.canvas.delete("all")

    if self.cam_prediction:
      self.start_camera()
      self.update_webcam()
      self.cam_prediction = False

    #Updating labels & buttons
    self.root.title(self.tr("Sugarcane Disease Classification"))
    self.home_frame.winfo_children()[0].config(text=self.tr("Select Language"))
    self.home_frame.winfo_children()[2].config(text=self.tr("Select Model"))

    self.btn_webcam_toggle.config(text=self.tr("Start Camera") if not self.cap else self.tr("Stop Camera"))

    self.btn_select.config(text=self.tr("Select an Image"))
    self.btn_predict.config(text=self.tr("Predict"))
    
    self.result_label.config(text="")
  
  def show_result(self, orig_img, grad_img, pred_label, confidence_score):
    self.home_frame.pack_forget()
    self.result_frame.pack()

    #Original image
    orig_img = cv2.resize(orig_img, (300, 300))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_photo = ImageTk.PhotoImage(Image.fromarray(orig_img))
    self.result_canvas1.configure(image=orig_photo)
    self.result_canvas1.image = orig_photo

    #Grad-CAM image
    grad_img = grad_img.resize((300, 300))
    grad_photo = ImageTk.PhotoImage(grad_img)
    self.result_canvas2.configure(image=grad_photo)
    self.result_canvas2.image = grad_photo

    confidence_score = f"{confidence_score:.4f}"

    self.result_label.config(text=f"{self.tr('Prediction')}: {pred_label} (Confidence: {confidence_score})")
    self.btn_back.config(text=self.tr("Back to Main Screen"))
  
  def select_image(self):
    if self.cap:
      self.cap.release()
      self.cap = None
    path = filedialog.askopenfilename()
    if path:
      self.image_path = path
      img = cv2.imread(path)
      self.current_frame = img
      self.display_image(img)

  def select_model(self, event=None):
    self.model_name = self.model_var.get()

  def toggle_webcam(self):
    if not self.cap:
      if self.start_camera():
        self.update_webcam()
        self.btn_webcam_toggle.config(text=self.tr("Stop Camera"))
    else:
      self.stop_camera()
      self.canvas.delete("all")
      self.btn_webcam_toggle.config(text=self.tr("Start Camera"))

  def update_webcam(self):
    frame = self.read_frame()
    if frame is not None:
      self.display_image(frame)
      self.root.after(10, self.update_webcam)
  
  def display_image(self, img):
    self.canvas.delete("all")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (300, 300))
    photo = ImageTk.PhotoImage(Image.fromarray(img))
    self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    self.canvas.image = photo
  
  def freeze_frame(self, event=None):
    self.frame_frozen = True
    self.cam_prediction = True
    self.stop_camera()

  def run_prediction(self):
    try:
      label, conf, img_bytes = self.predict()
      grad_img = Image.open(io.BytesIO(img_bytes))
      orig_img = self.current_frame if self.current_frame is not None else cv2.imread(self.image_path)
      self.show_result(orig_img, grad_img, label, conf)
    except Exception as e:
      messagebox.showwarning("Error", f"Prediction failed: {e}")

if __name__ == "__main__":
  root = tk.Tk()
  app = TkinterApp(root)
  root.mainloop()
