import torch
import torch.nn as nn
from torchvision.models import vgg19

class LossNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        # On charge VGG19 pré-entraîné sur ImageNet
        vgg = vgg19(pretrained=True).features
        self.vgg = vgg.to(device).eval() # Mode évaluation (pas d'entraînement du VGG)
        
        # On gèle les paramètres (on ne veut pas modifier VGG)
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        # Les couches standard pour le Style Transfer
        # Style : couches superficielles (texture) + profondes (structure)
        # Content : couche profonde (structure globale)
        self.style_layers = ['0', '5', '10', '19', '28'] # conv1_1, conv2_1...
        self.content_layers = ['21'] # conv4_2

    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[f'style_{name}'] = x
            if name in self.content_layers:
                features[f'content_{name}'] = x
        return features

def calc_gram_matrix(input_tensor):
    """
    Calcule la matrice de Gram : la corrélation entre les filtres.
    C'est la signature mathématique du 'Style'.
    """
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2)) # Produit scalaire
    return G.div(c * h * w) # Normalisation

def calc_loss(loss_network, generated, content_target, style_target, lambda_c=10.0, lambda_s=50.0):
    # 1. Extraction des features
    gen_feats = loss_network(generated)
    cont_feats = loss_network(content_target)
    styl_feats = loss_network(style_target)
    
    # 2. Content Loss (MSE sur les features)
    content_loss = 0
    for name in loss_network.content_layers:
        key = f'content_{name}'
        content_loss += torch.nn.functional.mse_loss(gen_feats[key], cont_feats[key])
        
    # 3. Style Loss (MSE sur les matrices de Gram)
    style_loss = 0
    for name in loss_network.style_layers:
        key = f'style_{name}'
        gram_gen = calc_gram_matrix(gen_feats[key])
        gram_style = calc_gram_matrix(styl_feats[key])
        style_loss += torch.nn.functional.mse_loss(gram_gen, gram_style)
        
    total_loss = lambda_c * content_loss + lambda_s * style_loss
    return total_loss, content_loss, style_loss