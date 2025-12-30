import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class ContentStyleDataset(Dataset):
    def __init__(self, content_dir, style_dir, image_size=256):
        self.content_dir = content_dir
        self.style_dir = style_dir
        
        self.content_images = [f for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png'))]
        self.style_images = [f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.png'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), 
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.content_images)

    def __getitem__(self, idx):
        # --- 1. Charger l'image de CONTENU ---
        try:
            content_path = os.path.join(self.content_dir, self.content_images[idx])
            content_img = Image.open(content_path).convert('RGB')
            content_img.load() 
        except (OSError, IOError) as e:
            print(f"⚠️ Image Contenu corrompue ignorée : {self.content_images[idx]}")
            return self.__getitem__((idx + 1) % len(self))

        # --- 2. Charger une image de STYLE ---
        while True:
            try:
                style_name = random.choice(self.style_images)
                style_path = os.path.join(self.style_dir, style_name)
                style_img = Image.open(style_path).convert('RGB')
                style_img.load()
                break
            except (OSError, IOError) as e:
                print(f"⚠️ Image Style corrompue ignorée : {style_path}, essai d'une autre...")
        
        # --- 3. Transformations ---
        content_img = self.transform(content_img)
        style_img = self.transform(style_img)
        
        return content_img, style_img

def get_loader(content_dir, style_dir, batch_size=4, num_workers=2):
    dataset = ContentStyleDataset(content_dir, style_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Exemple d'utilisation
# loader = get_loader('./data/coco', './data/wikiart')
# content_batch, style_batch = next(iter(loader))
