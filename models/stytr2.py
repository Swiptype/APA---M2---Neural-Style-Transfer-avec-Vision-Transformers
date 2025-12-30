import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. ENCODER ---
class SimpleEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(), 
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(), 
            nn.Conv2d(256, output_dim, kernel_size=3, stride=2, padding=1), nn.ReLU() 
        )
    def forward(self, x):
        return self.model(x)

# --- 2. DECODER ---
class SimpleDecoder(nn.Module):
    def __init__(self, input_dim=512):
        super().__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_dim, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# --- 3. BLOC TRANSFORMER ---
class StyleTransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, content_features, style_features):
        q = content_features
        k = style_features
        v = style_features

        src2, _ = self.self_attn(q, k, v)
        src = content_features + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src

# --- 4. MODELE GLOBAL ---
class StyTr2(nn.Module):
    def __init__(self, embed_dim=512, nhead=8, num_layers=2):
        super().__init__()
        
        self.encoder = SimpleEncoder(embed_dim)
        self.decoder = SimpleDecoder(embed_dim)
        
        self.transformer_layers = nn.ModuleList([
            StyleTransformerBlock(embed_dim, nhead) for _ in range(num_layers)
        ])

    def forward(self, content_img, style_img):
        c_feats = self.encoder(content_img)
        s_feats = self.encoder(style_img)
        
        b, c, h, w = c_feats.shape
        content_seq = c_feats.flatten(2).permute(2, 0, 1)
        style_seq = s_feats.flatten(2).permute(2, 0, 1)   
        
        # 3. TRANSFORMER
        output_seq = content_seq
        for layer in self.transformer_layers:
            output_seq = layer(output_seq, style_seq)
            
        # 4. REMISE EN FORME
        features_spatial = output_seq.permute(1, 2, 0).view(b, c, h, w)
        
        # 5. DECODAGE
        result = self.decoder(features_spatial)
        
        return result
