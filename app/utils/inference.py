import torch
import numpy as np
from PIL import Image
from anomalib.models import Patchcore
import os

@torch.no_grad()
def load_model():
    ckpt_path = "models/patchcore_pen_cap.ckpt"
    if not os.path.exists(ckpt_path):
        return None
    
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.05,
    )
    
    # Pass weights_only=False since this is your own trusted local checkpoint
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    
    model.eval()
    return model

def score_image(model, image: Image.Image, good_thresh: float, defect_thresh: float):
    img_resized = image.resize((256, 256))
    img_tensor = torch.tensor(np.array(img_resized)).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0)
    
    outputs = model(img_tensor)
    
    score = 0.0
    if hasattr(outputs, "pred_score") and outputs.pred_score is not None:
        score = float(outputs.pred_score.item())
    elif isinstance(outputs, dict) and "pred_score" in outputs:
        score = float(outputs["pred_score"].item())
        
    if score <= good_thresh:
        status = "GOOD"
    elif good_thresh < score <= defect_thresh:
        status = "NEEDS REVIEW"
    else:
        status = "DEFECTIVE"
        
    return score, status