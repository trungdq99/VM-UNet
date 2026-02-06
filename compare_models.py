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

def get_metrics(preds, gts, threshold=0.5):
    """Calculates metrics from predictions and ground truths."""
    preds = np.array(preds).reshape(-1)
    gts = np.array(gts).reshape(-1)

    y_pre = np.where(preds >= threshold, 1, 0)
    y_true = np.where(gts >= 0.5, 1, 0)

    confusion = confusion_matrix(y_true, y_pre)
    # Handle cases where confusion matrix might not be 2x2 (e.g. only one class present)
    if confusion.shape == (2, 2):
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1]
    else:
        # Fallback or simplified calculation if needed, though for binary seg it usually fits
        # If ground truth is all 0s and pred is all 0s, confusion is [[N, 0], [0, 0]]
        # This is strictly for 0 and 1 classes.
        TN = confusion[0,0] if confusion.shape[0] > 0 else 0
        TP = 0
        FP = 0
        FN = 0
        # This part is simplified; usually with sufficient data 2x2 is expected.
    
    total = float(np.sum(confusion))
    accuracy = float(TN + TP) / total if total != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
    miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

    
    # Calculate per-case metrics (DSC and HD95)
    dsc_list = []
    hd95_list = []
    
    
    for p, g in zip(preds, gts):
        # Ensure binary masks and correct shape (2D)
        p = np.squeeze(p)
        g = np.squeeze(g)
        
        p_bin = (p >= threshold).astype(np.uint8)
        g_bin = (g >= 0.5).astype(np.uint8)
        
        # Dice per case
        intersection = (p_bin * g_bin).sum()
        union = p_bin.sum() + g_bin.sum()
        if union == 0:
            dsc = 1.0 # Both empty is a match
        else:
            dsc = 2.0 * intersection / union
        dsc_list.append(dsc)
        
        # HD95 per case
        if p_bin.sum() > 0 and g_bin.sum() > 0:
            try:
                # metric.binary.hd95 is imported via from utils import * (which imports from medpy)
                hd95 = metric.binary.hd95(p_bin, g_bin)
            except Exception as e:
                print(f"HD95 calculation error: {e}")
                hd95 = 0.0
        else:
            hd95 = 0.0
        hd95_list.append(hd95)

    return {
        "Accuracy": accuracy,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "DSC": f1_or_dsc, # Keep global F1 as "DSC" or overwrite? Let's keep global for table, but use list for plot. 
        "Mean DSC": np.mean(dsc_list),
        "HD95": np.mean(hd95_list),
        "mIoU": miou,
        "Confusion Matrix": confusion,
        "DSC_list": dsc_list,
        "HD95_list": hd95_list
    }

def generate_qualitative_results(model, dataset, my_ckpt, author_ckpt, device, output_dir, count=5):
    """Generates visual comparisons for a few random samples."""
    print(f"\n--- Generating Qualitative Results (Saving to {output_dir}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    indices = random.sample(range(len(dataset)), min(count, len(dataset)))
    
    # Prepare model loading helper
    def load_weights(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            return True
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")
            return False

    for idx in indices:
        img_tensor, mask_tensor = dataset[idx] # Assuming dataset returns (img, mask) tensors or arrays
        # Check if dataset returns tensors or numpy (NPY_datasets usually returns tensors via transforms)
        # However, looking at utils.py myToTensor, it returns tensors. 
        
        # Ensure we have a batch dimension
        if isinstance(img_tensor, np.ndarray):
            img_input = torch.from_numpy(img_tensor).unsqueeze(0).float().to(device)
        else:
            img_input = img_tensor.unsqueeze(0).float().to(device)
            
        # Ground Truth processing
        if isinstance(mask_tensor, torch.Tensor):
            gt_mask = mask_tensor.numpy()
        else:
            gt_mask = mask_tensor
        
        # Squeeze dimensions for visualization (H, W)
        if gt_mask.ndim == 3: gt_mask = gt_mask[0] # C,H,W -> H,W
            
        # Predict with My Model
        if not load_weights(my_ckpt): continue
        with torch.no_grad():
            my_out = model(img_input)
            if isinstance(my_out, tuple): my_out = my_out[0]
            my_pred = torch.sigmoid(my_out).squeeze().cpu().numpy()
            my_pred_bin = (my_pred >= 0.5).astype(np.uint8) * 255

        # Predict with Author Model
        if not load_weights(author_ckpt): continue
        with torch.no_grad():
            author_out = model(img_input)
            if isinstance(author_out, tuple): author_out = author_out[0]
            author_pred = torch.sigmoid(author_out).squeeze().cpu().numpy()
            author_pred_bin = (author_pred >= 0.5).astype(np.uint8) * 255

        # Process Input Image for Display
        img_disp = img_input.squeeze().cpu().numpy()
        if img_disp.ndim == 3: img_disp = img_disp.transpose(1, 2, 0) # C,H,W -> H,W,C
        # Normalize to 0-1 for plt
        img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min())
        
        # Process GT
        gt_disp = gt_mask
        
        # Create Plot
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
        axes[2].imshow(author_pred_bin, cmap='gray')
        axes[2].set_title("Author Prediction")
        axes[2].axis('off')

        # My Model
        axes[3].imshow(my_pred_bin, cmap='gray')
        axes[3].set_title("My Prediction")
        axes[3].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"comparison_{idx}.png")
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

            out = model(img)
            # Handle tuple output if necessary (checking engine.py logic)
            if isinstance(out, tuple):
                out = out[0]
            
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            # Post-processing for metrics
            msk_np = msk.squeeze(1).cpu().detach().numpy()
            out_np = out.squeeze(1).cpu().detach().numpy()
            
            gts.append(msk_np)
            preds.append(out_np)

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
    print("\n--- Preparing Validation Dataset ---")
    try:
        val_dataset = NPY_datasets(config.data_path, config, train=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1, # Eval usually done with batch_size 1 for accuracy (matches train.py)
            shuffle=False,
            pin_memory=True,
            num_workers=config.num_workers,
            drop_last=False
        )
        print(f"Validation set size: {len(val_dataset)}")
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
            # Helper to handle if state_dict is inside a key (common in training scripts)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load with strict=False to ignore extra keys like 'total_ops', 'total_params'
            msg = model.load_state_dict(state_dict, strict=False)
            if msg.missing_keys:
                print(f"Warning: Missing keys in state_dict: {msg.missing_keys}")
            # we can ignore msg.unexpected_keys as they are likely the cause of the previous error

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
