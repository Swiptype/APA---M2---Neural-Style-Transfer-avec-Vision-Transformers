import torch
import torch.optim as optim
from models.stytr2 import StyTr2
from models.loss_network import LossNetwork, calc_loss
from utils.data_loader import get_loader

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4
EPOCHS = 10
BATCH_SIZE = 4
CONTENT_DIR = "./data/coco/train2017" # Adaptez vos chemins
STYLE_DIR = "./data/wikiart/train"

def train():
    print(f"Entraînement sur {DEVICE}")
    
    # 1. Initialisation
    model = StyTr2().to(DEVICE)
    loss_net = LossNetwork(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    dataloader = get_loader(CONTENT_DIR, STYLE_DIR, batch_size=BATCH_SIZE)

    # 2. Boucle
    for epoch in range(EPOCHS):
        for i, (content_imgs, style_imgs) in enumerate(dataloader):
            content_imgs = content_imgs.to(DEVICE)
            style_imgs = style_imgs.to(DEVICE)

            # Forward pass
            generated_imgs = model(content_imgs, style_imgs)

            # Calcul de la Loss
            total_loss, l_c, l_s = calc_loss(loss_net, generated_imgs, content_imgs, style_imgs)

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient Clipping (Important pour les Transformers pour éviter l'instabilité)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Batch {i} | "
                      f"Total Loss: {total_loss.item():.2f} "
                      f"(Content: {l_c.item():.2f}, Style: {l_s.item():.2f})")
        
        # Sauvegarde du modèle à chaque époque
        torch.save(model.state_dict(), f"stytr2_epoch_{epoch}.pth")
        print("Modèle sauvegardé.")

if __name__ == "__main__":
    train()