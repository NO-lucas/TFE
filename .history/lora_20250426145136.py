import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from utils import cls_acc, get_function
import pytorch_warmup as warmup
import torch.nn.functional as F
from transformers.modeling_outputs import ImageClassifierOutput
import cv2
import torch

from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
import wandb
import large_image
import openslide
import polars as pl
from torchvision import transforms
from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
    load_lora
)


class ClipWrapper(nn.Module):
    def __init__(self, clip_model, model_linear):
        super(ClipWrapper, self).__init__()
        self.clip_model = clip_model
        self.model_linear = model_linear

    def forward(self, x):
        # clip_model returns a tuple, we unpack it
        clip_output = self.clip_model(x)
        # Select only the first element of the tuple
        image_features = clip_output[0]
        # Apply the linear layer on the image features
        output = self.model_linear(image_features)
        return output


def get_number_trainable_parameters(model_name, clip_model):
    n_param = np.sum([p.numel() for p in get_lora_parameters(clip_model)])
    return n_param


def get_feature_size(model, input_size):
    model.eval()

    # Move model to GPU if available
    device = next(model.parameters()).device

    with torch.no_grad():
        # Create a sample input tensor and move it to the same device as the model
        sample_input = torch.randn(1, *input_size).to(device)
        features = model(sample_input)

        if isinstance(features, ImageClassifierOutput):
            features = features.logits

        return np.prod(features.size()[1:])


def evaluate_uni(args, clip_model, loader):

    clip_model.eval()

    acc = 0.0
    loss_epoch = 0.0
    tot_samples = 0

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()
            image_features = clip_model(images)

            if isinstance(image_features, ImageClassifierOutput):
                image_features = image_features.logits

            loss = F.cross_entropy(image_features, target)
            loss_epoch += loss.item() * target.shape[0]
            acc += cls_acc(image_features, target) * target.shape[0]
            tot_samples += target.shape[0]

    acc /= tot_samples
    loss_epoch /= tot_samples
    return acc, loss_epoch


def run_uni(args, clip_model, logit_scale, train_loader, val_loader, test_loader):
    """Classifier experiment - backbone freezed and classification layer added on the top of it"""

    
    return
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def evaluate_lora_uni(args, clip_model, loader, test=False, name=''):
    clip_model.eval()

    acc = 0.0
    loss_epoch = 0.0
    tot_samples = 0
    errors_matrix = {}
    total_per_class = {}
    hesitation_data = {}
    colors = []
    prob_values = []
    correct_classifications = {}
    classes_x = ["1", "2", "3", "41", "42", "51", "52", "54", "57"]

    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            images, target = images.cuda(), target.cuda()

            if args.model_name in ["clip"]:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model(images)
            else:
                image_features = clip_model(images)

            if isinstance(image_features, ImageClassifierOutput):
                image_features = image_features.logits

            loss = F.cross_entropy(image_features.float(), target)
            loss_epoch += loss.item() * target.shape[0]
            acc += cls_acc(image_features, target) * target.shape[0]
            tot_samples += target.shape[0]

            if test:
                preds = image_features.argmax(dim=1)
                probs = torch.softmax(image_features, dim=1)
                top2_preds = probs.argsort(dim=1, descending=True)[:, :2]

                for label, pred, prob, top2 in zip(target.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy(), top2_preds.cpu().numpy()):
                    total_per_class[label] += 1
                    if label != pred:
                        errors_matrix[label][pred] += 1
                        hesitation_data[label].append(prob[pred])

                        if label in top2:  
                            colors.append("green")
                        else:
                            colors.append("blue")
                    else:
                        
                        prob_values.append(prob[label])
                        correct_classifications[label].append(prob[label])

    try:
        wandb.log({"val_loss": loss_epoch / tot_samples, "val_accuracy": acc / tot_samples})
    except:
        pass
    acc /= tot_samples
    loss_epoch /= tot_samples

    if test:
        classes = sorted(total_per_class.keys())
        confusion_matrix = [[errors_matrix[true_class].get(pred_class, 0) for pred_class in classes] for true_class in classes]

        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted class")
        plt.ylabel("Correct class")
        plt.title("Matrix of misclassifications")
        plt.savefig(f"{args.model_name}_{name}_errors.png")
        plt.show()

        all_x, all_y, means_x, means_y = [], [], [], []
        for cls, probs in hesitation_data.items():
            all_x.extend([cls] * len(probs))
            all_y.extend(probs)
            means_x.append(cls)
            means_y.append(np.mean(probs))

        
        plt.figure(figsize=(10, 5))
        plt.scatter(all_x, all_y, alpha=0.8, c=colors, label="Prediction errors (green = right 2nd pred, blue = wrong)")
        plt.scatter(means_x, means_y, color="red", marker="o", s=100, label="Means by class")


        all_x_correct, all_y_correct, means_x_correct, means_y_correct = [], [], [], []
        for cls, prob_list in correct_classifications.items():
            all_x_correct.extend([cls] * len(prob_list))
            all_y_correct.extend(probs)
            means_x_correct.append(cls)
            means_y_correct.append(np.mean(prob_list))
        
        plt.scatter(means_x_correct, means_y_correct, color="black", marker="o", s=100, label="Means by class")
        plt.xlabel("Correct class")
        plt.ylabel("Probability given to the predicted class")
        plt.title("How much the model is hesitating predicting errors")
        plt.legend()
        plt.xticks(ticks=list(range(len(classes_x))), labels=classes_x)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"{args.model_name}_{name}_hesitation_scatter.png")
        plt.show()

    return acc, loss_epoch



def run_uni_lora(args, clip_model, logit_scale, train_loader, val_loader, test_loader):

    WANDB = True
    VALIDATION = True
    acc_val = 0.0
    best_val_loss = float("inf")
    best_model_path = None  

    patience = 500
    no_improve_epochs = 0
    best_acc_val = 0.0


    if WANDB:
        name_run = f"{args.model_name}_{args.lr}_{args.r}_{args.seed}_{args.shots}_{args.n_iters}_{args.position}_{args.encoder}"
        wandb.init(
            project="lora_" + str(args.dataset) + "_" + str(args.shots), name=name_run
        )
        config = wandb.config
        config.model_name = args.model_name
        config.lr = args.lr
        config.rank = args.r
        config.seed = args.seed
        config.shots = args.shots
        config.n_iters = args.n_iters
        config.position = args.position
        config.encoder = args.encoder
        config.params = args.params
        config.dataset = args.dataset
        config.weight_decay = 1e-1
        config.beta1 = 0.9
        config.beta2 = 0.999
        config.logit_scale = logit_scale
        config.batch_size = args.batch_size
        config.dropout_rate = args.dropout_rate
    
    if args.model_name in ["vit_google", "clip"]:
        num_features = 512
    elif args.model_name in ["quilt", "biomedclip"]:
        num_features = 512
    elif args.model_name in ["uni"]:
        num_features = get_feature_size(clip_model, (3, 224, 224))

    model_linear = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(num_features, args.num_classes)
    ).cuda()

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    mark_only_lora_as_trainable(clip_model)
    trainable_parameters_ = get_lora_parameters(clip_model)

    for _, param in model_linear.named_parameters():
        trainable_parameters_.append(param)

    if args.model_name in ["clip", "quilt", "biomedclip"]:
        clip_model_ = nn.Sequential(clip_model.visual, model_linear)
    elif args.model_name in ["uni"]:
        clip_model_ = nn.Sequential(clip_model, model_linear)
    elif args.model_name in ["vit_google"]:
        setattr(clip_model, "classifier", model_linear)
        clip_model_ = clip_model
    else:
        raise RuntimeError(
            "Wrong model name used. Try clip, uni, biomedclip, vit_google or quilt."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters_,
        weight_decay=1e-1,
        betas=(0.9, 0.999),
        lr=args.lr,
    )

    num_steps = args.n_iters * args.shots
    warmup_period = 50
    total_iters = warmup_period + num_steps if args.shots > 0 else num_steps

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_steps, eta_min=1e-6
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

    scaler = torch.cuda.amp.GradScaler()
    count_iters = 0
    count_epochs = 0

    while count_iters < total_iters:
        print("Iteration number:", count_iters)
        clip_model_.train()
        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.0

        for i, (images, target) in enumerate(tqdm(train_loader)):
            images, target = images.cuda(), target.cuda()
            if args.model_name in ["clip"]:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = clip_model_(images)
            else:
                output = clip_model_(images)

            if isinstance(output, ImageClassifierOutput):
                output = output.logits

            loss = F.cross_entropy(output, target)
            acc_train += cls_acc(output, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    scheduler.step()

            count_iters += 1
            if count_iters == total_iters:
                break

        if WANDB:
            wandb.log({"epoch_train_loss": loss_epoch / tot_samples,
                       "epoch_train_accuracy": acc_train / tot_samples})

        acc_train /= tot_samples
        loss_epoch /= tot_samples

        current_lr = scheduler.get_last_lr()[0]
        print(f"OptLR: {optimizer.param_groups[0]['lr']:.6f}, LR: {current_lr:.6f}, Acc: {acc_train:.4f}, Loss: {loss_epoch:.4f}")

        # **Validation**
        if VALIDATION:
            count_epochs += 1
            acc_val, loss_val = evaluate_lora_uni(args, clip_model_, val_loader)
            print(f"**** Val accuracy: {acc_val:.2f}, Val loss: {loss_val:.4f} ****")

            if WANDB:
                wandb.log({"val_loss": loss_val, "val_accuracy": acc_val})

            # **Sauvegarde du meilleur modèle**
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                best_model_path = f'best_clip_model_{args.shots}_{args.dataset}.pth'
                torch.save(clip_model_.state_dict(), best_model_path)
                print(f"🔥 New best model saved: {best_model_path} (Val loss: {best_val_loss:.4f})")

            # **Early Stopping**
            if acc_val > best_acc_val:
                best_acc_val = acc_val
                no_improve_epochs = 0  # Reset if better val acc
            else:
                no_improve_epochs += 1  # epochs without enhancing

            if no_improve_epochs >= patience:
                print(f"⏹️ Early stopping launched")
                break 


    print("Testing with last clip model ...")
    acc_test, _ = evaluate_lora_uni(args, clip_model_, test_loader, test=True)
    print(f"**** Final test accuracy for last clip model : {acc_test:.2f} ****")
    # **Loading best model before test**
    if best_model_path:
        print(f"🔄 Loading best model from {best_model_path} for final evaluation")
        clip_model_.load_state_dict(torch.load(best_model_path))

    acc_test, _ = evaluate_lora_uni(args, clip_model_, test_loader, test=True)
    print(f"**** Final test accuracy: {acc_test:.2f} ****")

    json_path = (
        f"./Results/lora_{args.dataset}_{args.model_name}_{args.seed}_{args.shots}_{args.lr}_{args.r}_results.json"
    )

    with open(json_path, "w") as f:
        json.dump({"val_acc": acc_val, "test_acc": acc_test}, f)

    args.save_path = json_path.replace(".json", ".pt")
    save_lora(args, list_lora_layers)

    print(f"Best model saved at: {best_model_path}")
    return best_model_path




def create_background_mask(slide_path, level=7, save_path="bg_mask.npy", debug=False):
    slide = openslide.OpenSlide(slide_path)
    print(slide.dimensions)
    thumbnail = np.array(slide.read_region((0, 0), level, slide.level_dimensions[level]))[:, :, :3]
    gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)

    # threshold
    _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

    # cleaning very small background
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    # Find countours
    contours, hierarchy = cv2.findContours(clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # empty mask
    mask = np.zeros(clean.shape, dtype=np.uint8)

    # draw contours
    cv2.drawContours(mask, contours, -1, color=1, thickness=cv2.FILLED)
    final_mask = 1 - mask

    np.save(save_path, final_mask)

    if debug:
        debug_img = thumbnail.copy()
        debug_img[final_mask == 1] = [0, 255, 0]
        cv2.imwrite("debug_mask.png", debug_img)

        plt.figure(figsize=(12, 6))
        plt.imshow(debug_img)
        plt.title("background mask")
        plt.axis("off")
 
    
    return final_mask


def is_background_patch(x, y, mask, res_ratio, patch_size):
    """Check if the center of the mask is in the background"""
    cx = int((x + patch_size // 2) / res_ratio)
    cy = int((y + patch_size // 2) / res_ratio)
    if cx >= mask.shape[1] or cy >= mask.shape[0]:
        return False
    return mask[cy, cx] == 1
import glob

def run_process_wsi(args, clip_model, output_path, test_loader, only_test=False):
    # ndpi_paths = glob.glob("wsi_images/*.ndpi")
    ndpi_paths = ["C:/Users/lucas/AAA_MEMOIRE/Code_Memoire/img/database/09C07888.ndpi",
                  "C:/Users/lucas/AAA_MEMOIRE/Code_Memoire/img/database/11C01217.ndpi"]
    for slide_path in ndpi_paths:
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        print(f"processing {slide_name}...")

        slide = openslide.OpenSlide(slide_path)
        low_res = 4
        low_scale_factor = (slide.level_dimensions[0][0]/ slide.level_dimensions[low_res][0])
        high_res = 2
        high_scale_factor = (slide.level_dimensions[0][0]/ slide.level_dimensions[high_res][0])

        width_low_res = slide.level_dimensions[low_res][0]
        height_low_res = slide.level_dimensions[low_res][1]

        width_high_res = slide.level_dimensions[high_res][0]
        height_high_res = slide.level_dimensions[high_res][1]

        level = 7

        patch_size = 224
    
        res_ratio = 2**(level - low_res)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
        ])

        os.makedirs(f"wsi_results_low_res/{slide_name}", exist_ok=True)
        os.makedirs(f"wsi_results_high_res/{slide_name}", exist_ok=True)

        bg_mask_path = f"{slide_name}_background_mask.npy"
        
        bg_mask = create_background_mask(slide_path, level=level, save_path=bg_mask_path, debug=True)

        # Loading low res model
        num_features = 512
        model_linear = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(num_features, args.num_classes)).cuda()
        apply_lora(args, clip_model)
        clip_model = clip_model.cuda()
        mark_only_lora_as_trainable(clip_model)
        for _, param in model_linear.named_parameters():
            get_lora_parameters(clip_model).append(param)
        clip_model_ = nn.Sequential(clip_model.visual, model_linear)
        clip_model_.load_state_dict(torch.load("best_models/low_res.pth"))

        low_res_preds = []

        print("Prediction low res")
        with torch.no_grad():
            for x in tqdm(range(0, width_low_res, patch_size), desc=f"LowRes X - {slide_name}"):
                data = []
                for y in range(0, height_low_res, patch_size):
                    if is_background_patch(x, y, bg_mask, res_ratio, patch_size):
                        continue
                    tile = slide.read_region((int(x*low_scale_factor), int(y*low_scale_factor)), low_res, (224, 224)).convert("RGB")
                    tile_tensor = transform(tile).unsqueeze(0).cuda()

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model_(tile_tensor)
                    logits = image_features
                    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                    prediction = torch.argmax(logits, dim=1).item()

                    data.append({
                        "x": x,
                        "y": y,
                        "prediction": prediction,
                        "probabilities": probs.tolist(),
                        "embedding": image_features.cpu().numpy().tolist()
                    })
                    low_res_preds.append((x, y, prediction))

                df = pl.DataFrame(data)
                df.write_parquet(f"wsi_results_low_res/{slide_name}/wsi_lowres_results_x{x}.parquet")

        # High res
        print("Raffinement high resolution...")
        model_linear = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(num_features, 9)).cuda()
        clip_model_ = nn.Sequential(clip_model.visual, model_linear)
        clip_model_.load_state_dict(torch.load("best_models/high_res.pth"))

        with torch.no_grad():
            confirmed_41_coords = set()

            for x, y, pred in tqdm(low_res_preds, desc=f"HighRes refinement - {slide_name}"):
                if is_background_patch(x, y, bg_mask, res_ratio, patch_size):
                    continue

                
                if pred == 3:
                    # Random pacthe
                    confirmations = 0
                    for _ in range(3):
                        # choosing random patche in the low res one
                        offset_x = np.random.randint(0, patch_size)
                        offset_y = np.random.randint(0, patch_size)
                        hx = int((x + offset_x / patch_size) * high_scale_factor)
                        hy = int((y + offset_y / patch_size) * high_scale_factor)
                        tile = slide.read_region((hx, hy), high_res, (224, 224)).convert("RGB")
                        tile_tensor = transform(tile).unsqueeze(0).cuda()

                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            image_features = clip_model_(tile_tensor)
                        logits = image_features
                        prediction = torch.argmax(logits, dim=1).item()

                        if prediction == 3:
                            confirmations += 1

                    if confirmations >= 1:
                        confirmed_41_coords.add((x, y))  # We validate the patche
                        # register pred
                        df = pl.DataFrame([{
                            "x": x,
                            "y": y,
                            "confirmed_prediction": 41,
                            "note": "confirmed from high res"
                        }])
                        df.write_parquet(f"wsi_results_high_res_sampling/{slide_name}/confirmed_41_patch_{x}_{y}.parquet")

                elif pred == 4:
                
                    for _ in range(3):
                        offset_x = np.random.randint(0, patch_size)
                        offset_y = np.random.randint(0, patch_size)
                        hx = int((x + offset_x / patch_size) * high_scale_factor)
                        hy = int((y + offset_y / patch_size) * high_scale_factor)
                        tile = slide.read_region((hx, hy), high_res, (224, 224)).convert("RGB")
                        tile_tensor = transform(tile).unsqueeze(0).cuda()

                        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                            image_features = clip_model_(tile_tensor)
                        logits = image_features
                        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                        prediction = torch.argmax(logits, dim=1).item()

                        if prediction in [3,4,5,6,7,8]:
                            df = pl.DataFrame([{
                                "x": x,
                                "y": y,
                                "refined_prediction": prediction,
                                "probabilities": probs.tolist(),
                                "embedding": image_features.cpu().numpy().tolist()
                            }])
                            df.write_parquet(f"wsi_results_high_res_sampling/{slide_name}/refined_patch_{x}_{y}_pred{prediction}.parquet")

        print(f"Pipeline over for {slide_name}.\n")


import os
import time
import glob
import torch
import openslide
import pandas as pd
import numpy as np
import polars as pl
from torchvision import transforms
from tqdm import tqdm
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import os
import time
import glob
import torch
import openslide
import pandas as pd
import numpy as np
from torchvision import transforms
from tqdm import tqdm
import joblib
import torch.nn as nn

def compute_entropy(probs):
    return -np.sum(probs * np.log(probs + 1e-8))

def pipeline(args, clip_model, output_path):
    ndpi_paths = [
        "C:/Users/lucas/AAA_MEMOIRE/Code_Memoire/img/database/09C07888.ndpi",
        "C:/Users/lucas/AAA_MEMOIRE/Code_Memoire/img/database/11C01217.ndpi"
    ]
    
    for slide_path in ndpi_paths:
        slide_name = os.path.splitext(os.path.basename(slide_path))[0]
        print(f"Processing {slide_name} ...")
        
        slide = openslide.OpenSlide(slide_path)
        low_res = 2
        low_scale_factor = (slide.level_dimensions[0][0] / slide.level_dimensions[low_res][0])

        width_low_res = slide.level_dimensions[low_res][0]
        height_low_res = slide.level_dimensions[low_res][1]

        level = 7
        patch_size = 224
        res_ratio = 2 ** (level - low_res)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
        ])

        os.makedirs(f"wsi_results_low_res/{slide_name}", exist_ok=True)

        bg_mask_path = f"{slide_name}_background_mask.npy"
        bg_mask = create_background_mask(slide_path, level=level, save_path=bg_mask_path, debug=True)

        # Charger modèle de prédiction CLIP
        num_features = 512
        model_linear = nn.Sequential(nn.Flatten(start_dim=1), nn.Linear(num_features, 9)).cuda()
        apply_lora(args, clip_model)
        clip_model = clip_model.cuda()
        mark_only_lora_as_trainable(clip_model)
        for _, param in model_linear.named_parameters():
            get_lora_parameters(clip_model).append(param)
        clip_model_ = nn.Sequential(clip_model.visual, model_linear)
        clip_model_.load_state_dict(torch.load("best_models/high_res.pth"))

        # Charger SVM et scaler
        best_svm_model = joblib.load('best_SVM_models/svm_model.joblib')
        scaler = joblib.load('best_SVM_models/scaler.joblib')

        data_for_features = []

        t0 = time.time()

        print("Prediction on high resolution patches...")
        with torch.no_grad():
            for x in tqdm(range(0, width_low_res, patch_size), desc=f"LowRes X - {slide_name}"):
                for y in range(0, height_low_res, patch_size):
                    if is_background_patch(x, y, bg_mask, res_ratio, patch_size):
                        continue
                    tile = slide.read_region((int(x * low_scale_factor), int(y * low_scale_factor)), low_res, (224, 224)).convert("RGB")
                    tile_tensor = transform(tile).unsqueeze(0).cuda()

                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        image_features = clip_model_(tile_tensor)

                    logits = image_features
                    probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                    prediction = torch.argmax(logits, dim=1).item()

                    # Calcul de l'entropie ici
                    entropy = compute_entropy(probs)

                    # Collecte toutes les infos nécessaires
                    data_for_features.append({
                        "x": x,
                        "y": y,
                        "prediction": prediction,
                        "entropy": entropy
                    })

        t1 = time.time()
        print(f"Time looping over the slide : {t1 - t0:.2f} seconds")

        ### Reformatting + SVM
        t2 = time.time()

        # Mise en forme pour la SVM
        df_patches = pd.DataFrame(data_for_features)

        features_dict = {}
        for idx, row in df_patches.iterrows():
            img_id = slide_name  # identifiant unique pour une slide
            if img_id not in features_dict:
                features_dict[img_id] = np.zeros((9, 2))  # 9 classes × (count, mean entropy)
            features_dict[img_id][row["prediction"], 0] += 1  # count : combien de patches pour chaque classe
            features_dict[img_id][row["prediction"], 1] += row["entropy"]  # somme des entropies

        # Moyenne des entropies par classe
        for img_id in features_dict:
            for class_idx in range(9):
                count = features_dict[img_id][class_idx, 0]
                if count > 0:
                    features_dict[img_id][class_idx, 1] /= count  # moyenne d'entropie

        X_slide = np.array([v.flatten() for v in features_dict.values()])

        # Standardisation
        X_slide_scaled = scaler.transform(X_slide)

        # Prédiction finale avec SVM
        y_slide_pred = best_svm_model.predict(X_slide_scaled)

        print(f"✅ Predicted class for {slide_name}: {y_slide_pred[0]}")

        t3 = time.time()
        print(f"⏱️ Temps pour reformattage + prédiction SVM : {t3 - t2:.2f} secondes")

        print(f"Pipeline over for {slide_name}.\n")




def run_uni_lora_percent(
    args, clip_model, logit_scale, train_loader, val_loader, test_loader
):

    VALIDATION = True
    acc_val = 0.0

    if args.model_name in ["vit_google", "clip"]:
        num_features = 768
    elif args.model_name in ["quilt", "biomedclip"]:
        num_features = 512
    elif args.model_name in ["uni"]:
        num_features = get_feature_size(clip_model, (3, 224, 224))

    model_linear = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(num_features, args.num_classes)
    ).cuda()

    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()

    mark_only_lora_as_trainable(clip_model)
    trainable_parameters_ = get_lora_parameters(clip_model)
    for _, param in model_linear.named_parameters():
        trainable_parameters_.append(param)

    if args.model_name in ["clip", "quilt", "biomedclip"]:
        clip_model_ = nn.Sequential(clip_model.visual, model_linear)
    elif args.model_name in ["uni"]:
        clip_model_ = nn.Sequential(clip_model, model_linear)
    elif args.model_name in ["vit_google"]:
        setattr(clip_model, "classifier", model_linear)
        clip_model_ = clip_model
    else:
        raise RuntimeError(
            "Wrong model name used. Try clip, uni, biomedclip, vit_google or quilt."
        )

    optimizer = torch.optim.AdamW(
        trainable_parameters_,
        weight_decay=1e-1,
        betas=(0.9, 0.999),
        lr=args.lr,
    )

    num_steps = args.n_iters * len(train_loader)
    warmup_period = 50
    total_iters = warmup_period + num_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, num_steps, eta_min=1e-6
    )
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period)

    # training LoRA
    scaler = torch.cuda.amp.GradScaler()

    count_iters = 0

    while count_iters < total_iters:
        clip_model_.train()

        acc_train = 0
        tot_samples = 0
        loss_epoch = 0.0

        for i, (images, target) in enumerate(tqdm(train_loader)):

            images, target = images.cuda(), target.cuda()
            if args.model_name in ["clip"]:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    output = clip_model_(images)
            else:
                output = clip_model_(images)

            if isinstance(output, ImageClassifierOutput):
                output = output.logits

            loss = F.cross_entropy(output, target)
            acc_train += cls_acc(output, target) * target.shape[0]
            loss_epoch += loss.item() * target.shape[0]
            tot_samples += target.shape[0]

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    scheduler.step()

            count_iters += 1

            if count_iters == total_iters:
                break

        acc_train /= tot_samples
        loss_epoch /= tot_samples

        current_lr = scheduler.get_last_lr()[0]
        for param_group in optimizer.param_groups:
            optimizer_lr = param_group["lr"]
        print(
            " OptLR: {:.6f}, LR: {:.6f}, Acc: {:.4f}, Loss: {:.4f}".format(
                optimizer_lr, current_lr, acc_train, loss_epoch
            )
        )

        # Eval
        if VALIDATION:
            acc_val, loss_val = evaluate_lora_uni(args, clip_model_, val_loader)
            print("**** Val accuracy: {:.2f}. ****\n".format(acc_val))

    acc_test, _ = evaluate_lora_uni(args, clip_model_, test_loader)
    print("**** Final test accuracy: {:.2f}. ****\n".format(acc_test))

    json_path = (
        "./Results/lora_"
        + str(args.dataset)
        + "_"
        + str(args.model_name)
        + "_"
        + str(args.seed)
        + "_"
        + str(args.shots)
        + "_"
        + str(args.lr)
        + "_"
        + str(args.r)
        + str(args.percentage)
        + "_percent_results.json"
    )

    with open(
        json_path,
        "w",
    ) as f:
        json.dump({"val_acc": acc_val, "test_acc": acc_test}, f)

    args.save_path = json_path.replace(".json", ".pt")
    save_lora(args, list_lora_layers)

    return
