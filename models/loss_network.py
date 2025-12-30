import torch
import torch.nn as nn
from torchvision.models import vgg19
import torchvision.transforms as T

class LossNetwork(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = vgg19(pretrained=True).features
        self.vgg = vgg.to(device).eval()
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device))

        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.style_layers = ['0', '5', '10', '19', '28']
        self.content_layers = ['21']

    def forward(self, x):
        x = (x - self.mean) / self.std
        
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.style_layers:
                features[f'style_{name}'] = x
            if name in self.content_layers:
                features[f'content_{name}'] = x
        return features

def calc_gram_matrix(input_tensor):
    b, c, h, w = input_tensor.size()
    features = input_tensor.view(b, c, h * w)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(c * h * w)

def calc_loss(loss_network, generated, content_target, style_target, lambda_c=1.0, lambda_s=100000.0):
    gen_feats = loss_network(generated)
    cont_feats = loss_network(content_target)
    styl_feats = loss_network(style_target)
    
    content_loss = 0
    for name in loss_network.content_layers:
        key = f'content_{name}'
        content_loss += torch.nn.functional.mse_loss(gen_feats[key], cont_feats[key])
        
    style_loss = 0
    for name in loss_network.style_layers:
        key = f'style_{name}'
        gram_gen = calc_gram_matrix(gen_feats[key])
        gram_style = calc_gram_matrix(styl_feats[key])
        style_loss += torch.nn.functional.mse_loss(gram_gen, gram_style)
        
    total_loss = lambda_c * content_loss + lambda_s * style_loss
    return total_loss, content_loss, style_loss
