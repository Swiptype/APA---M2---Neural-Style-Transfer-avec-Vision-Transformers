import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class ContentStyleDataset(Dataset):
    def __init__(self, content_dir, style_dir, image_size=256):
        self.content_dir = content_dir
        self.style_dir = style_dir
        
        # Liste des fichiers
        self.content_images = [f for f in os.listdir(content_dir) if f.endswith(('.jpg', '.png'))]
        self.style_images = [f for f in os.listdir(style_dir) if f.endswith(('.jpg', '.png'))]
        
        # Pipeline de transformation (Essentiel !)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)), # Redimensionnement forcé
            transforms.ToTensor(),                       # Conversion [0, 1]
            # Normalisation standard ImageNet (souvent utilisée pour VGG)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        # On se base sur la longueur du dataset de contenu (COCO est plus grand)
        return len(self.content_images)

    def __getitem__(self, idx):
        # 1. Charger l'image de Contenu (déterministe via idx)
        content_path = os.path.join(self.content_dir, self.content_images[idx])
        content_img = Image.open(content_path).convert('RGB')
        
        # 2. Charger une image de Style (aléatoire)
        # On veut voir différents styles pour le même contenu au fil des époques
        style_name = random.choice(self.style_images)
        style_path = os.path.join(self.style_dir, style_name)
        style_img = Image.open(style_path).convert('RGB')
        
        # 3. Appliquer les transformations
        content_img = self.transform(content_img)
        style_img = self.transform(style_img)
        
        return content_img, style_img

def get_loader(content_dir, style_dir, batch_size=4, num_workers=2):
    dataset = ContentStyleDataset(content_dir, style_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

# Exemple d'utilisation
# loader = get_loader('./data/coco', './data/wikiart')
# content_batch, style_batch = next(iter(loader))