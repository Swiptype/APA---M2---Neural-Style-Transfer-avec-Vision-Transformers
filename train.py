import torch
import torch.optim as optim
from models.stytr2 import StyTr2
from models.loss_network import LossNetwork, calc_loss
from utils.data_loader import get_loader
import time

# --- CONFIGURATION OPTIMISÉE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-4 
EPOCHS = 1   
BATCH_SIZE = 4 
CONTENT_DIR = "./data/coco/train2017"
STYLE_DIR = "./data/wikiart/train"
LAMBDA_CONTENT = 1.0
LAMBDA_STYLE = 100000.0

def train():
    print(f"--- DÉBUT DE L'ENTRAÎNEMENT FINAL sur {DEVICE} ---")
    
    # 1. Initialisation
    model = StyTr2().to(DEVICE)
    loss_net = LossNetwork(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 2. Données
    dataloader = get_loader(CONTENT_DIR, STYLE_DIR, batch_size=BATCH_SIZE)
    print(f"Dataset chargé. {len(dataloader)} batchs par époque.")

    # 3. Boucle
    model.train()
    for epoch in range(EPOCHS):
        start_time = time.time()
        running_loss = 0.0
        
        for i, (content_imgs, style_imgs) in enumerate(dataloader):
            content_imgs = content_imgs.to(DEVICE)
            style_imgs = style_imgs.to(DEVICE)

            generated_imgs = model(content_imgs, style_imgs)

            total_loss, l_c, l_s = calc_loss(
                loss_net, 
                generated_imgs, 
                content_imgs, 
                style_imgs, 
                lambda_c=LAMBDA_CONTENT, 
                lambda_s=LAMBDA_STYLE
            )

            optimizer.zero_grad()
            total_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            running_loss += total_loss.item()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(dataloader)}] | "
                      f"Loss: {total_loss.item():.2f} "
                      f"(C: {l_c.item():.2f}, S: {l_s.item():.2f})")

        duration = time.time() - start_time
        avg_loss = running_loss / len(dataloader)
        print(f"--- Fin Epoch {epoch+1} en {duration:.0f}s | Moyenne Loss: {avg_loss:.2f} ---")
        
        save_path = f"stytr2_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Modèle sauvegardé : {save_path}")

if __name__ == "__main__":
    train()
