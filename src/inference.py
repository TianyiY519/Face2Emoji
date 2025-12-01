import torch
from torchvision import transforms, models
from PIL import Image
import argparse

EMOJI_MAP = {
    0: "😄 Happy",
    1: "😢 Sad",
    2: "😡 Angry",
    3: "😮 Surprise",
    4: "😨 Fear",
    5: "🤢 Disgust",
    6: "😐 Neutral"
}

def load_model(weight_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def predict(image_path, model):
    transform = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path)
    img_t = transform(image).unsqueeze(0)

    outputs = model(img_t)
    _, predicted = torch.max(outputs, 1)

    print("Predicted emotion:", EMOJI_MAP[predicted.item()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="models/face2emoji_epoch10.pth")
    args = parser.parse_args()

    model = load_model(args.weights)
    predict(args.image, model)
