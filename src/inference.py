"""
CLI Inference Script for Rock & Mineral Classification.
Allows direct evaluation on spectral CSV files or mineral images using saved checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

# Add root directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models import (
    CLASS_NAMES,
    NUM_CLASSES,
    Model1DCNN,
    MLP,
    get_resnet50,
    get_mobilevit,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def load_profile_model(model_type: str, fps: int = 1):
    """
    Load 1D-CNN or MLP checkpoint for spectral profiles.
    """
    model_name = f"model_{model_type}_profiles_{fps}fps.pth"
    model_path = ROOT_DIR / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    state_dict = torch.load(model_path, map_location=DEVICE)

    if model_type == "cnn":
        model = Model1DCNN(num_classes=NUM_CLASSES)
        model.load_state_dict(state_dict)
    elif model_type == "mlp":
        input_size = state_dict['layers.0.weight'].shape[1]
        model = MLP(input_size=input_size, num_classes=NUM_CLASSES)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown profile model type: {model_type}")

    model.to(DEVICE)
    model.eval()
    return model


def load_image_model(model_type: str, fps: int = 1):
    """
    Load MobileViT or ResNet50 checkpoint for mineral images.
    """
    model_name = f"model_{model_type}_images_{fps}fps.pth"
    model_path = ROOT_DIR / "models" / model_name
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    state_dict = torch.load(model_path, map_location=DEVICE)

    if model_type == "resnet50":
        model = get_resnet50(num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(state_dict)
    elif model_type == "mobilevit":
        model = get_mobilevit(num_classes=NUM_CLASSES, pretrained=False)
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unknown image model type: {model_type}")

    model.to(DEVICE)
    model.eval()
    return model


def predict_profile(model, file_path: str):
    """
    Run inference on a Raman profile CSV file.
    """
    df = pd.read_csv(file_path, header=None)
    data = df.values.astype(np.float32)
    tensor = torch.tensor(data, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    print(f"\n--- Inference Result: {Path(file_path).name} ---")
    print(f"Total Profiles Evaluated: {len(preds)}")
    unique_preds, counts = np.unique(preds, return_counts=True)
    for cls_idx, count in zip(unique_preds, counts):
        pct = (count / len(preds)) * 100
        print(f"  - {CLASS_NAMES[cls_idx]}: {count}/{len(preds)} samples ({pct:.1f}%)")

    top_idx = unique_preds[np.argmax(counts)]
    print(f"\nFinal Consensus Diagnosis: {CLASS_NAMES[top_idx]}")


def predict_image(model, image_path: str):
    """
    Run inference on a mineral image.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=-1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    print(f"\n--- Inference Result: {Path(image_path).name} ---")
    print(f"Predicted Mineral: {CLASS_NAMES[pred_idx]} ({probs[pred_idx]*100:.2f}% confidence)")
    print("\nClass Probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  - {name}: {p*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="CLI Inference for Rock Classification with RAMAN Spectrometer")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn_1fps", "cnn_30fps", "mlp_1fps", "mlp_30fps", "resnet50_1fps", "resnet50_30fps", "mobilevit_1fps", "mobilevit_30fps"],
        help="Model architecture and capture frame rate"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to spectral CSV file or mineral image (.bmp, .jpg, .png)"
    )

    args = parser.parse_args()

    model_type, fps_str = args.model.rsplit("_", 1)
    fps = int(fps_str.replace("fps", ""))

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {input_path}")
        sys.exit(1)

    print(f"Loading {args.model} onto {DEVICE}...")
    if model_type in ["cnn", "mlp"]:
        model = load_profile_model(model_type, fps=fps)
        predict_profile(model, str(input_path))
    else:
        model = load_image_model(model_type, fps=fps)
        predict_image(model, str(input_path))


if __name__ == "__main__":
    main()
