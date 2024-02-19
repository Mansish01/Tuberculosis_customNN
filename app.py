import torch
import requests
from PIL import Image
from torchvision import transforms as T
import gradio as gr
from models.customNN import Model

# model = torch.load(r'artifacts\run-2023-11-29-15-18-18\ckpt-FirstNeural-val=0.696-epoch=1')
# model.eval()
# model_path = r"artifacts/run-2023-12-21-16-32-32/ckpt-Model-val=0.962-epoch=6"
# model_path = r"artifacts/run-2024-02-18-00-21-29/ckpt-Model-val=0.951-epoch=8"
model_path = r"artifacts/run-2024-02-19-00-42-42/ckpt-Model-val=0.936-epoch=5"

checkpoint = torch.load(model_path)

model_state_dict  = checkpoint['model_state_dict']

model = Model(img_size= 256, num_channels=3, num_labels=2)
model.load_state_dict(model_state_dict)
model.eval()
# print(checkpoint)

labels = ['normal','tuberculosis']

transforms= T.Compose([
        T.Resize((256, 256)), 
        T.ToTensor()])
  
def predict(inp):
  threshold = 0.5
  inp = transforms(inp).unsqueeze(0)
  with torch.no_grad():
    prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    print(prediction)
    confidences = {labels[i]: float(prediction[i]) for i in range(prediction.shape[0])}
    print(confidences)
  
    sorted_confidences = sorted(confidences.items(), key=lambda x: x[1])
    # print(sorted_confidences)
    # diff = sorted_confidences[3][1] - sorted_confidences[2][1]
    # if diff < threshold:
    #   return "unable to predict"
    
  return confidences



gr.Interface(fn=predict,
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=2),
             examples=[r"/home/manish/Downloads/normal_chest.jpeg"]).launch()



























  #   # Finding top classes and their confidences
    # sorted_confidences = sorted(confidences.items(), key=lambda x: x[1], reverse=True)
    # top_classes = sorted_confidences[:4]
        
    #     # Checking if the difference between top confidences is less than threshold
    # if len(top_classes) > 1 and (top_classes[0][1] - top_classes[1][1]) < threshold:
    #         return {'Unable to predict': 1.0}  # Return 'Unable to predict' label