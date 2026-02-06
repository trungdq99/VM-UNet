import torch
from torch.utils.data import DataLoader
import argparse
import os
import sys
import shutil
import numpy as np
import gdown
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import random

# Import internal modules (assuming this script is in the root of the project)
from models.vmunet.vmunet import VMUNet
from datasets.dataset import NPY_datasets
from configs.config_setting import setting_config
from utils import *

def download_from_drive(url, output_path):
    """Downloads a file from a Google Drive URL."""
    print(f"Downloading from Drive: {url} -> {output_path}")
    try:
        gdown.download(url, output_path, quiet=False, fuzzy=True)
        if os.path.exists(output_path):
            print(f"Successfully downloaded to {output_path}")
            return True
        else:
            print(f"Failed to download file from {url}")
            return False
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def calculate_hd95(pred, gt):
    """
    Robust HD95 calculation.
    Returns 0.0 if both are empty (perfect match).
    Returns 100.0 if one is empty and other is not (complete mismatch penalty).
    """
    if pred.sum() > 0 and gt.sum() > 0:
        try:
            return metric.binary.hd95(pred, gt)
        except Exception as e:
            print(f"HD95 calculation error: {e}")
            return 100.0
    elif pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    else:
        # Penalize if one is empty and the other is not
        return 100.0

def get_metrics(preds, gts, threshold=0.5):
    """
    Calculates metrics from predictions and ground truths memory-efficiently.
    """
    dsc_list = []
    hd95_list = []
    
    # Global confusion matrix evaluators
    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0

    for p, g in zip(preds, gts):
        # Ensure binary masks and correct shape (2D)
        # p, g are numpy arrays
        p = np.squeeze(p)
        g = np.squeeze(g)
        
        # Binarize
        p_bin = (p >= threshold).astype(np.uint8)
        g_bin = (g >= 0.5).astype(np.uint8)
        
        # 1. Update Global Terms for Confusion Matrix
        # We process flattened arrays per image, which is much smaller than dataset-wide flatten
        p_flat = p_bin.ravel()
        g_flat = g_bin.ravel()
        
        # Fast confusion matrix using bincount or boolean logic
        # 00: TN, 01: FP (pred=1, gt=0), 10: FN (pred=0, gt=1), 11: TP
        # Let's stick to simple logic for clarity or use sklearn's confusion_matrix per image
        tn, fp, fn, tp = confusion_matrix(g_flat, p_flat, labels=[0, 1]).ravel()
        
        total_TN += tn
        total_FP += fp
        total_FN += fn
        total_TP += tp

        # 2. Per-case Metrics (DSC & HD95)
        # Dice
        intersection = (p_bin * g_bin).sum()
        union = p_bin.sum() + g_bin.sum()
        if union == 0:
            dsc = 1.0 # Both empty -> Match
        else:
            dsc = 2.0 * intersection / union
        dsc_list.append(dsc)
        
        # HD95
        hd95 = calculate_hd95(p_bin, g_bin)
        hd95_list.append(hd95)

    # Calculate Global Aggregate Metrics
    total_pixels = total_TN + total_TP + total_FP + total_FN
    accuracy = (total_TN + total_TP) / total_pixels if total_pixels > 0 else 0
    sensitivity = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    specificity = total_TN / (total_TN + total_FP) if (total_TN + total_FP) > 0 else 0
    
    # Global F1/DSC based on total TP/FP/FN (Micro-averaged)
    global_dsc = 2 * total_TP / (2 * total_TP + total_FP + total_FN) if (2 * total_TP + total_FP + total_FN) > 0 else 0
    miou = total_TP / (total_TP + total_FP + total_FN) if (total_TP + total_FP + total_FN) > 0 else 0

    # Reconstruct Global Confusion Matrix
    confusion = np.array([[total_TN, total_FP], [total_FN, total_TP]])

    return {
        "Accuracy": accuracy,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "DSC": global_dsc, 
        "Mean DSC": np.mean(dsc_list),
        "HD95": np.mean(hd95_list),
        "mIoU": miou,
        "Confusion Matrix": confusion,
        "DSC_list": dsc_list,
        "HD95_list": hd95_list
    }

def generate_qualitative_results(model, dataset, my_ckpt, author_ckpt, device, output_dir, count=5):
    """Generates visual comparisons efficiently."""
    print(f"\n--- Generating Qualitative Results (Saving to {output_dir}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    indices = random.sample(range(len(dataset)), min(count, len(dataset)))
    
    # Pre-load data to avoid keeping dataset open or issues
    samples = []
    for idx in indices:
        img_tensor, mask_tensor = dataset[idx]
        if isinstance(img_tensor, np.ndarray):
            img_input = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
        else:
            img_input = img_tensor.unsqueeze(0).float().to(device)
            
        if isinstance(mask_tensor, torch.Tensor):
            gt_mask = mask_tensor.numpy()
        else:
            gt_mask = mask_tensor
        
        # Binarize ground truth mask for visualization to remove noise
        gt_mask = (gt_mask > 0.5).astype(np.float32)
        
        samples.append({
            "idx": idx,
            "img_input": img_input,
            "gt_mask": gt_mask
        })

    predictions = {}
    
    # Helper to run inference
    def run_inference(ckpt_path, name):
        results = []
        try:
            print(f"Loading {name} weights for visualization...")
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            
            # Clean state dict keys if needed (e.g. remove module. prefix)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
            model.eval()
            
            with torch.no_grad():
                for sample in samples:
                    out = model(sample["img_input"])
                    if isinstance(out, tuple): out = out[0]
                    # Model already applies sigmoid if num_classes=1 (see models/vmunet/vmunet.py)
                    pred = out.squeeze().cpu().numpy()
                    pred_bin = (pred >= 0.5).astype(np.uint8) * 255
                    results.append(pred_bin)
            return results
        except Exception as e:
            print(f"Error generating predictions for {name}: {e}")
            return None

    # Run inferences sequentially
    my_preds = run_inference(my_ckpt, "My Model")
    author_preds = run_inference(author_ckpt, "Author Model")
    
    if my_preds is None or author_preds is None:
        print("Skipping visualization due to inference errors.")
        return

    # Generate Plots
    for i, sample in enumerate(samples):
        img_disp = sample["img_input"].squeeze().cpu().numpy()
        if img_disp.ndim == 3: img_disp = img_disp.transpose(1, 2, 0)
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
        
        gt_disp = sample["gt_mask"]
        if gt_disp.ndim == 3: gt_disp = gt_disp[0]
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        
        # Input
        axes[0].imshow(img_disp, cmap='gray' if img_disp.ndim==2 else None)
        axes[0].set_title("Input Image")
        axes[0].axis('off')
        
        # GT
        axes[1].imshow(gt_disp, cmap='gray')
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')

        # Author
        axes[2].imshow(author_preds[i], cmap='gray')
        axes[2].set_title("Author Prediction")
        axes[2].axis('off')

        # My Model
        axes[3].imshow(my_preds[i], cmap='gray')
        axes[3].set_title("My Prediction")
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"comparison_{sample['idx']}.png")
        plt.savefig(save_path)
        plt.close()

def generate_quantitative_plots(my_results, author_results, output_dir):
    """Generates Heatmap and Boxplot."""
    print(f"\n--- Generating Quantitative Plots (Saving to {output_dir}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Confusion Matrix Heatmaps
    if "Confusion Matrix" in my_results:
        plt.figure(figsize=(6, 5))
        sns.heatmap(my_results["Confusion Matrix"], annot=True, fmt='g', cmap='Blues')
        plt.title("My Model Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, "my_model_confusion_matrix.png"))
        plt.close()

    if "Confusion Matrix" in author_results:
        plt.figure(figsize=(6, 5))
        sns.heatmap(author_results["Confusion Matrix"], annot=True, fmt='g', cmap='Greens')
        plt.title("Author Model Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(output_dir, "author_model_confusion_matrix.png"))
        plt.close()

    # 2. DSC Boxplot
    if "DSC_list" in my_results and "DSC_list" in author_results:
        data = [my_results["DSC_list"], author_results["DSC_list"]]
        plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=['My Model', 'Author Model'])
        plt.title('Dice Coefficient (DSC) Distribution')
        plt.ylabel('DSC')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(output_dir, "dsc_boxplot.png"))
        plt.close()

def evaluate_model(model, val_loader, criterion, device, config):
    model.eval()
    preds = []
    gts = []
    loss_list = []
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Evaluating"):
            img, msk = data
            img = img.to(device).float()
            msk = msk.to(device).float()
            # Binarize mask to remove noise (0-255 values normalized to 0-1, so >0.5 covers >127)
            msk = (msk > 0.5).float()

            out = model(img)
            # Handle tuple output if necessary (checking engine.py logic)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            # Post-processing for metrics
            msk_np = msk.squeeze(1).cpu().detach().numpy()
            # Model already applies sigmoid
            out_np = out.squeeze(1).cpu().detach().numpy()
            
            preds.append(out_np)
            gts.append(msk_np)

    metrics = get_metrics(preds, gts, config.threshold)
    metrics["Loss"] = np.mean(loss_list)
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare two VM-UNet models")
    parser.add_argument("my_model_url", type=str, help="Google Drive link for your trained model (best.pth)")
    parser.add_argument("author_model_url", type=str, help="Google Drive link for the author's model")
    args = parser.parse_args()

    # Setup directories
    download_dir = "downloaded_checkpoints"
    os.makedirs(download_dir, exist_ok=True)

    my_ckpt_path = os.path.join(download_dir, "my_best.pth")
    author_ckpt_path = os.path.join(download_dir, "author_best.pth")

    # Download models
    print("\n--- Downloading Models ---")
    if not download_from_drive(args.my_model_url, my_ckpt_path):
        print("Failed to download your model. Exiting.")
        return
    if not download_from_drive(args.author_model_url, author_ckpt_path):
        print("Failed to download author's model. Exiting.")
        return

    # Load Configuration from config_setting.py
    config = setting_config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Prepare Dataset
    print("\n--- Preparing Test Dataset ---")
    try:
        val_dataset = NPY_datasets(config.data_path, config, split="test", train=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1, # Eval usually done with batch_size 1 for accuracy
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=False
        )
        print(f"Test set size: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Make sure 'config.data_path' in configs/config_setting.py is correct.")
        return

    # Initialize Model Structure
    print("\n--- Initializing Model Architecture ---")
    model_cfg = config.model_config
    try:
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=None, # We load custom weights later
        )
        model = model.to(device)
    except Exception as e:
        print(f"Error initializing model: {e}")
        return

    criterion = config.criterion

    # Evaluation Helper
    def run_eval(name, ckpt_path):
        print(f"\n--- Evaluating {name} ---")
        print(f"Loading weights from: {ckpt_path}")
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Clean state dict keys if needed
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            
            msg = model.load_state_dict(new_state_dict, strict=False)
            if msg.missing_keys:
                print(f"Warning: Missing keys in state_dict: {msg.missing_keys}")
            
            # Verify weights changed
            param_norm = sum(p.norm().item() for p in model.parameters())
            print(f"Model L2 Norm verify: {param_norm:.4f}")

        except Exception as e:
            print(f"Error loading weights for {name}: {e}")
            return None
        
        return evaluate_model(model, val_loader, criterion, device, config)

    # Run Comparisons
    my_results = run_eval("My Model", my_ckpt_path)
    author_results = run_eval("Author's Model", author_ckpt_path)

    # Print Comparison Table
    if my_results and author_results:
        print("\n\n" + "="*60)
        print(f"{'Metric':<25} | {'My Model':<15} | {'Author Model':<15}")
        print("-" * 60)
        
        metrics_keys = ["Loss", "mIoU", "DSC", "Mean DSC", "HD95", "Accuracy", "Sensitivity (Recall)", "Specificity"]
        
        for key in metrics_keys:
            val1 = my_results.get(key, 0.0)
            val2 = author_results.get(key, 0.0)
            print(f"{key:<25} | {val1:<15.4f} | {val2:<15.4f}")
        
        print("="*60)
        print("="*60)
        
        # Generate Visualizations
        results_dir = "./results"
        images_dir = os.path.join(results_dir, "comparison_images")
        plots_dir = os.path.join(results_dir, "plots")
        
        generate_qualitative_results(model, val_dataset, my_ckpt_path, author_ckpt_path, device, images_dir)
        generate_quantitative_plots(my_results, author_results, plots_dir)
        
        print("\nEvaluation Complete.")
    else:
        print("\nEvaluation failed for one or both models.")

if __name__ == "__main__":
    main()
