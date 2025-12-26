import torch
from PIL import Image
from torchvision import transforms
from models.stytr2 import StyTr2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def stylize_image(content_path, style_path, model_path, output_path="output.jpg"):
    # 1. Charger le modèle
    model = StyTr2().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # 2. Préparer les images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Inverse transform pour sauvegarder l'image (dé-normalisation)
    def denorm(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
        return tensor * std + mean

    c_img = transform(Image.open(content_path).convert('RGB')).unsqueeze(0).to(DEVICE)
    s_img = transform(Image.open(style_path).convert('RGB')).unsqueeze(0).to(DEVICE)

    # 3. Génération
    with torch.no_grad():
        generated = model(c_img, s_img)
    
    # 4. Sauvegarde
    result_tensor = denorm(generated[0]).clamp(0, 1).cpu()
    transforms.ToPILImage()(result_tensor).save(output_path)
    print(f"Image stylisée sauvegardée : {output_path}")

# Exemple
# stylize_image("ma_photo.jpg", "van_gogh.jpg", "stytr2_epoch_5.pth")