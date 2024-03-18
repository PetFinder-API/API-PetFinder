import time

import torch
from PIL import Image
from flask import Flask, request, jsonify
from timm import create_model
from torchvision import transforms

app = Flask(__name__)


def load_model():
    print("Load model...")
    start_time = time.time()
    model = create_model('swin_large_patch4_window7_224', pretrained=True, num_classes=1)

    # Load the entire checkpoint, not just the model state dict
    checkpoint = torch.load('model.pth', map_location=torch.device('cpu'))

    # Assuming the model's state dict is stored under 'model' in the checkpoint
    model_state_dict = checkpoint['model']

    # Load the state dict into your model
    model.load_state_dict(model_state_dict)

    model.eval()
    end_time = time.time()
    print(f"The model got loaded in {end_time - start_time} seconds")
    return model


def predict(image_path: str) -> float:
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    img = Image.open(f"images/{image_path}").convert('RGB')
    img_processed = preprocess(img)
    img_batch = img_processed.unsqueeze(0)
    prediction = model(img_batch)
    return prediction.item()


@app.route('/prediction-pet-score', methods=['POST'])
def home():
    if request.is_json:
        data = request.get_json()
        img_path = data['img_path']
        score = predict(img_path)
        result = {"score": score}
        return jsonify(result), 200
    else:
        return jsonify({"error": "Request must be JSON"}), 400


model = load_model()

if __name__ == '__main__':
    app.run(debug=True)
