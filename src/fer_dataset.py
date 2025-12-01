import os
from PIL import Image
from torch.utils.data import Dataset

class FER2013Images(Dataset):
    EMOTION_TO_LABEL = {
        "angry": 0,
        "disgust": 1,
        "fear": 2,
        "happy": 3,
        "sad": 4,
        "surprise": 5,
        "neutral": 6,
    }

    def __init__(self, root, split="train", transform=None):
        self.root = os.path.join(root, split)
        self.transform = transform

        self.samples = []
        self.classes = sorted(os.listdir(self.root))

        for emotion in self.classes:
            emotion_path = os.path.join(self.root, emotion)

            # Skip non-folders
            if not os.path.isdir(emotion_path):
                continue

            label = self.EMOTION_TO_LABEL[emotion]

            for img_file in os.listdir(emotion_path):
                if img_file.endswith(".jpg") or img_file.endswith(".png"):
                    img_path = os.path.join(emotion_path, img_file)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


