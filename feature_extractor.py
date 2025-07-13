import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchreid.models import build_model

class FeatureExtractor:
    def __init__(self, device='cpu'):
        self.device = device
        self.model = build_model(name='osnet_x0_25', num_classes=1000, pretrained=True)
        self.model.classifier = torch.nn.Identity()
        self.model.to(device).eval()
        
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def get_deep_features(self, crop):
        img_t = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = self.model(img_t)
            f = torch.nn.functional.normalize(f, dim=1)
        return f.cpu().squeeze(0).numpy()
    
    def get_color_hist(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hists = []
        for ch in range(3):
            h = cv2.calcHist([hsv], [ch], None, [8], [0, 256])
            hists.append(h.flatten())
        hist = np.concatenate(hists).astype(np.float32)
        return hist / (hist.sum() + 1e-6)
    
    def extract(self, crop):
        f_deep = self.get_deep_features(crop)
        f_col = self.get_color_hist(crop)
        feat = np.concatenate([f_deep, f_col])
        return feat / (np.linalg.norm(feat) + 1e-6)