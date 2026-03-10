"""
evaluate.py — Core evaluation script for the Image Sentiment Competition
Loads a submitted model and evaluates it on the secret test set.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset


# ─── Configuration ────────────────────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLASSES = 5
CLASS_NAMES = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
MAX_INFERENCE_TIME_S = 30  # per batch of BATCH_SIZE


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_model_class(model_code_path: str):
    """Dynamically load model class from a Python file."""
    spec = importlib.util.spec_from_file_location("user_model", model_code_path)
    module = importlib.util.module_from_spec(spec)
    
    # Inject torch & common libs into module namespace
    import torchvision
    module.__dict__["torch"] = torch
    module.__dict__["nn"] = nn
    module.__dict__["torchvision"] = torchvision
    
    spec.loader.exec_module(module)
    
    # Find the first nn.Module subclass that isn't nn.Module itself
    model_class = None
    for name in dir(module):
        obj = getattr(module, name)
        try:
            if (
                isinstance(obj, type)
                and issubclass(obj, nn.Module)
                and obj is not nn.Module
            ):
                model_class = obj
                print(f"[INFO] Found model class: {name}")
                break
        except TypeError:
            continue
    
    if model_class is None:
        raise ValueError("No nn.Module subclass found in model_class.py")
    
    return model_class


def load_model(model_class, weights_path: str) -> nn.Module:
    """Instantiate model and load state dict."""
    model = model_class()
    state_dict = torch.load(weights_path, map_location=DEVICE)
    
    # Handle common state_dict wrappers
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()
    return model


def validate_output_shape(model: nn.Module):
    """Quick sanity check: ensure model outputs (B, 5) logits."""
    dummy = torch.zeros(2, 3, 224, 224).to(DEVICE)
    with torch.no_grad():
        out = model(dummy)
    
    if out.shape != torch.Size([2, NUM_CLASSES]):
        raise ValueError(
            f"Model output shape is {out.shape}, expected (2, {NUM_CLASSES}). "
            f"Make sure your model outputs raw logits of shape (B, {NUM_CLASSES})."
        )
    print(f"[INFO] Output shape validated: {out.shape} ✓")


def evaluate_model(model: nn.Module, test_data: torch.Tensor, test_labels: torch.Tensor):
    """
    Run inference and compute metrics.
    
    Returns:
        dict with f1_score, accuracy, per_class_f1, confusion_matrix,
                  inference_time_ms, total_samples
    """
    dataset = TensorDataset(test_data, test_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    all_preds = []
    all_labels = []
    total_inference_time = 0.0
    
    with torch.no_grad():
        for batch_images, batch_labels in loader:
            batch_images = batch_images.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            
            t_start = time.perf_counter()
            logits = model(batch_images)
            t_end = time.perf_counter()
            
            elapsed = t_end - t_start
            total_inference_time += elapsed
            
            if elapsed > MAX_INFERENCE_TIME_S:
                raise TimeoutError(
                    f"Batch inference took {elapsed:.1f}s > {MAX_INFERENCE_TIME_S}s limit."
                )
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch_labels.cpu().numpy().tolist())
    
    # ─── Compute metrics ──────────────────────────────────────────────────────
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    per_class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    cm = confusion_matrix(all_labels, all_preds).tolist()
    
    # Pad per_class_f1 if some classes are missing
    while len(per_class_f1) < NUM_CLASSES:
        per_class_f1.append(0.0)
    
    avg_inference_ms = (total_inference_time / len(loader)) * 1000
    
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0
    )
    print("\n" + report)
    
    return {
        "f1_score": round(weighted_f1, 6),
        "accuracy": round(accuracy, 6),
        "per_class_f1": [round(v, 6) for v in per_class_f1],
        "confusion_matrix": cm,
        "inference_time_ms": round(avg_inference_ms, 2),
        "total_samples": len(all_labels),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a submitted model")
    parser.add_argument("--model-path", required=True, help="Path to .pth file")
    parser.add_argument("--model-code", required=True, help="Path to model_class.py")
    parser.add_argument("--test-data", required=True, help="Path to test_data.pt (Tensor)")
    parser.add_argument("--test-labels", required=True, help="Path to test_labels.pt (Tensor)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--issue-number", type=int, default=0)
    args = parser.parse_args()
    
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Loading model from: {args.model_path}")
    
    try:
        # Load test data
        test_data = torch.load(args.test_data, map_location="cpu")
        test_labels = torch.load(args.test_labels, map_location="cpu")
        print(f"[INFO] Test set: {test_data.shape}, Labels: {test_labels.shape}")
        
        # Validate data
        assert test_data.ndim == 4, "test_data must be (N, 3, 224, 224)"
        assert test_data.shape[1] == 3, "Expected 3-channel images"
        assert test_data.max() <= 1.01, "Images should be normalized to [0, 1]"
        
        # Load model
        model_class = load_model_class(args.model_code)
        model = load_model(model_class, args.model_path)
        
        # Validate output shape
        validate_output_shape(model)
        
        # Evaluate
        print("[INFO] Starting evaluation...")
        results = evaluate_model(model, test_data, test_labels)
        results["issue_number"] = args.issue_number
        results["status"] = "success"
        
        print(f"\n[RESULTS]")
        print(f"  Weighted F1 : {results['f1_score']:.4f}")
        print(f"  Accuracy    : {results['accuracy']:.4f}")
        print(f"  Inference   : {results['inference_time_ms']:.1f} ms/batch")
        
        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] Results saved to {args.output}")
        
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        traceback.print_exc()
        
        error_results = {
            "status": "error",
            "error_message": str(e),
            "issue_number": args.issue_number,
            "f1_score": 0.0,
            "accuracy": 0.0,
            "per_class_f1": [0.0] * NUM_CLASSES,
            "confusion_matrix": [],
            "inference_time_ms": 0.0,
            "total_samples": 0,
        }
        with open(args.output, "w") as f:
            json.dump(error_results, f, indent=2)
        sys.exit(1)


if __name__ == "__main__":
    main()
