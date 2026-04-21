from flask import Flask, request, jsonify, send_file, make_response
from PIL import Image
import torch
import io
import numpy as np
import cv2
from torchvision import transforms, datasets
import torchvision.models as models

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = datasets.ImageFolder(root="Dataset/Sugarcane Dataset Split/train").classes

model_paths = {
  "EfficientNetB0": "Models/efficientnet_b0.pth",
  "ResNet50": "Models/resnet50.pth",
  "MobileNetV2": "Models/mobilenet_v2.pth"
}

transform = transforms.Compose([
  transforms.Resize((224, 224)),
  transforms.ToTensor(),
  transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225]
    )
])

#Loading the model
def load_model(name):
  if name == "EfficientNetB0":
    model = models.efficientnet_b0(num_classes=len(class_names))
  elif name == "ResNet50":
    model = models.resnet50(num_classes=len(class_names))
  elif name == "MobileNetV2":
    model = models.mobilenet_v2(num_classes=len(class_names))
  model.load_state_dict(torch.load(model_paths[name], map_location=device))
  model = model.to(device)
  model.eval()
  return model

models = {name: load_model(name) for name in model_paths}

#Fetching the most optimal layer for Grad-CAM visualization
def find_conv_layer(model, name):
  if name == "EfficientNetB0":
    return model.features[-1]
  elif name == "ResNet50":
    return model.layer4[-1].conv3
  elif name == "MobileNetV2":
    return model.features[-1]
  else:
    raise ValueError("Unsupported model name")


#Generating Grad-CAM visualization
def generate_gradcam(model, model_name, img_tensor, orig_img):
  model.eval()

  activations = []
  gradients = []

  # Get correct target layer
  target_layer = find_conv_layer(model, model_name)

  #Register hooks
  #Captures activations / Activation maps of the target convolution layer
  def fwd_hook(module, inp, out):
    activations.append(out)

  #Captures gradients w.r.t activations / Gradients of the target layer activations
  def bwd_hook(module, grad_in, grad_out):
    gradients.append(grad_out[0])

  h1 = target_layer.register_forward_hook(fwd_hook)
  h2 = target_layer.register_backward_hook(bwd_hook)

  #Forward pass
  output = model(img_tensor)
  pred_class = output.argmax(dim=1).item()
  probs = torch.softmax(output, dim=1)
  confidence = probs[0, pred_class].item()

  #Backward pass
  model.zero_grad()
  output[0, pred_class].backward()

  #Removing hooks
  h1.remove()
  h2.remove()

  #Converting activations and gradients
  act = activations[0].squeeze().cpu().detach().numpy()   # (C, H, W)
  grad = gradients[0].squeeze().cpu().detach().numpy()    # (C, H, W)

  #Computing weights
  weights = np.mean(grad, axis=(1, 2))

  #Canvas to build the Grad-CAM heatmap by summing weighted feature maps
  cam = np.zeros(act.shape[1:], dtype=np.float32)

  #Grad-CAM formula
  for i, w in enumerate(weights):
    cam += w * act[i]

  #ReLU
  cam = np.maximum(cam, 0)

  #Normalize
  cam = (cam - cam.min()) / (cam.max() + 1e-8)

  #Resize to image size
  cam = cv2.resize(cam, (orig_img.shape[1], orig_img.shape[0]))
  heatmap = np.uint8(255 * cam)
  heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

  #Overlay
  overlay = (0.6 * orig_img + 0.4 * heatmap_color).astype(np.uint8)

  _, buffer = cv2.imencode(".jpg", overlay)
  return class_names[pred_class], io.BytesIO(buffer.tobytes()), confidence

#Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
  image_file = request.files.get("image")
  model_name = request.form.get("model", "EfficientNetB0")

  if not image_file:
    return jsonify({"error": "No image provided"}), 400

  img = Image.open(image_file).convert("RGB")
  img_tensor = transform(img).unsqueeze(0).to(device)
  img_tensor.requires_grad_()
  orig_np = np.array(img)

  model = models.get(model_name)
  if model is None:
    return jsonify({"error": "Invalid model name"}), 400

  pred_label, gradcam_image, confidence = generate_gradcam(model, model_name, img_tensor, orig_np)

  #Confidence check
  #if confidence < 0.85:
  #  return jsonify({"Invalid Input": "Try again"}), 400

  response = make_response(send_file(gradcam_image, mimetype='image/jpeg', as_attachment=False, download_name="gradcam.jpg"))
  response.headers["Prediction-Label"] = pred_label
  response.headers["Confidence"] = str(confidence)
  return response

if __name__ == "__main__":
  app.run(host="127.0.0.1", port=5000) #host=0.0.0.0 for accessing server from any IP address
