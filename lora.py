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
    errors_matrix = defaultdict(lambda: defaultdict(int))  # Matrice d'erreurs
    total_per_class = defaultdict(int)  # Nombre total d'exemples par classe
    hesitation_data = defaultdict(list)  # Stocke les probabilités d'hésitation par vraie classe
    colors = []  # Stocke la couleur des points pour le scatter plot
    prob_values = []  # Liste pour stocker les probabilités des bonnes classifications
    correct_classifications = defaultdict(list)  # Stocke les bonnes classifications par classe
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
                probs = torch.softmax(image_features, dim=1)  # Convertir en probabilités
                top2_preds = probs.argsort(dim=1, descending=True)[:, :2]  # Deux meilleures prédictions

                for label, pred, prob, top2 in zip(target.cpu().numpy(), preds.cpu().numpy(), probs.cpu().numpy(), top2_preds.cpu().numpy()):
                    total_per_class[label] += 1
                    if label != pred:
                        errors_matrix[label][pred] += 1  # Enregistre l'erreur
                        hesitation_data[label].append(prob[pred])  # Probabilité accordée à la mauvaise classe

                        # Vérification de la 2e meilleure prédiction
                        if label in top2:  
                            colors.append("green")  # ✅ Le modèle hésitait avec la vraie classe
                        else:
                            colors.append("blue")  # ❌ Mauvaise prédiction sans hésitation avec la vraie classe
                    else:
                        # Si la prédiction est correcte, on ajoute ce point à une autre liste
                        # colors.append("black")  # ✅ Prédiction correcte
                        prob_values.append(prob[label])  # Probabilité de la bonne prédiction
                        correct_classifications[label].append(prob[label])  # Ajoute à la bonne classe

    try:
        wandb.log({"val_loss": loss_epoch / tot_samples, "val_accuracy": acc / tot_samples})
    except:
        pass
    acc /= tot_samples
    loss_epoch /= tot_samples

    if test:
        print("📊 Génération des visualisations d'erreur...")

        # Heatmap des erreurs
        classes = sorted(total_per_class.keys())  # Classes en int
        confusion_matrix = [[errors_matrix[true_class].get(pred_class, 0) for pred_class in classes] for true_class in classes]

        plt.figure(figsize=(10, 7))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Reds", xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted class")
        plt.ylabel("Correct class")
        plt.title("Matrix of misclassifications")
        plt.savefig(f"{args.model_name}_{name}_errors.png")
        plt.show()

        # Scatter plot des hésitations avec couleurs conditionnelles
        all_x, all_y, means_x, means_y = [], [], [], []
        for cls, probs in hesitation_data.items():
            all_x.extend([cls] * len(probs))  # X : Vraie classe (int)
            all_y.extend(probs)  # Y : Probabilité accordée à la mauvaise classe
            means_x.append(cls)
            means_y.append(np.mean(probs))  # Moyenne des hésitations pour chaque classe

        # Affichage des bons et mauvais points sur le même scatter plot
        plt.figure(figsize=(10, 5))
        plt.scatter(all_x, all_y, alpha=0.8, c=colors, label="Prediction errors (green = right 2nd pred, blue = wrong)")
        plt.scatter(means_x, means_y, color="red", marker="o", s=100, label="Means by class")


        all_x_correct, all_y_correct, means_x_correct, means_y_correct = [], [], [], []
        for cls, prob_list in correct_classifications.items():
            all_x_correct.extend([cls] * len(prob_list))  # X : Vraie classe (int)
            all_y_correct.extend(probs)  # Y : Probabilité accordée à la mauvaise classe
            means_x_correct.append(cls)
            means_y_correct.append(np.mean(prob_list))
        
        plt.scatter(means_x_correct, means_y_correct, color="black", marker="o", s=100, label="Means by class")
        plt.xlabel("Correct class")
        plt.ylabel("Probability given to the predicted class")
        plt.title("How much the model is hesitating predicting errors")
        plt.legend()
        plt.xticks(ticks=list(range(len(classes_x))), labels=classes_x)  # Labels pour X
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.savefig(f"{args.model_name}_{name}_hesitation_scatter.png")
        plt.show()

        print(f"📊 Graphiques enregistrés sous : {args.model_name}_{name}_errors.png et {args.model_name}_{name}_hesitation_scatter.png")

    return acc, loss_epoch



def run_uni_lora(args, clip_model, logit_scale, train_loader, val_loader, test_loader):

    WANDB = True
    VALIDATION = True
    acc_val = 0.0
    best_val_loss = float("inf")  # Initialisation avec une valeur très grande
    best_model_path = None  

    patience = 500  # Nombre d'epochs sans amélioration avant arrêt
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
                no_improve_epochs = 0  # Reset si amélioration
            else:
                no_improve_epochs += 1  # Compte les epochs sans amélioration

            if no_improve_epochs >= patience:
                print(f"⏹️ Early stopping déclenché après {patience} epochs sans amélioration.")
                break  # Arrêt de la boucle d'entraînement


    print("Testing with last clip model ...")
    acc_test, _ = evaluate_lora_uni(args, clip_model_, test_loader, test=True)
    print(f"**** Final test accuracy for last clip model : {acc_test:.2f} ****")
    # **Chargement du meilleur modèle avant test**
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



from tqdm import tqdm
def run_process_wsi(args, clip_model, output_path, test_loader, only_test=False):
    """
    Parcours les patches d'une image WSI, extrait les embeddings et effectue des prédictions.
    Args:
        args: arguments de configuration.
        clip_model: modèle CLIP pré-entraîné.
        output_path: chemin pour enregistrer les résultats.
    """

    slide = openslide.OpenSlide("image_wsi/11C01217.ndpi")
    

    # # Définir le classifieur
    num_features = 512
    model_linear = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(num_features, args.num_classes)
    ).cuda()
    
    list_lora_layers = apply_lora(args, clip_model)
    clip_model = clip_model.cuda()
    mark_only_lora_as_trainable(clip_model)
    trainable_parameters_ = get_lora_parameters(clip_model)

    for _, param in model_linear.named_parameters():
        trainable_parameters_.append(param)
    
    clip_model_ = nn.Sequential(clip_model.visual, model_linear)
    
    # Boucle sur tous les fichiers de poids dans le dossier best_models
    weight_folder = "best_models"
    for weights_name in os.listdir(weight_folder):
        weight_path = os.path.join(weight_folder, weights_name)
        
        if weights_name.endswith(".pth"):  # Vérifier que c'est bien un fichier de poids
            try:
                clip_model_.load_state_dict(torch.load(weight_path))
                print(f"Model {weights_name} loaded")
                acc_test, _ = evaluate_lora_uni(args, clip_model_, test_loader, test=True, name=weights_name)
                print(f"**** Final test accuracy for {weights_name}: {acc_test:.2f} ****")
            except Exception as e:
                print(f"Error loading {weights_name}: {e}")
    
    if only_test:
        print("Only the test was performed")
        return
    
    weight_path = 'best_models/model_82_8_600_100.pth'
    weight_path_51_52 = "best_51_52/best_clip_model_600_cyto_51_52.pth"
    clip_model_.load_state_dict(torch.load(weight_path))
    # specialized_model = torch.load(weight_path_51_52)
    # specialized_model.eval().cuda()

    patch_size = 224

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275]),
    ])
    width, height = slide.dimensions

    print("Device utilisé :", "GPU" if torch.cuda.is_available() else "CPU")
    os.makedirs("wsi_results", exist_ok=True)

    i = 0
    with torch.no_grad():
        for x in tqdm(range(0, width, patch_size), desc="Processing tiles (x)"):
            data = []
            for y in tqdm(range(0, height, patch_size), desc="Processing tiles (y)", leave=False):
                tile = slide.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
                tile_tensor = transform(tile).unsqueeze(0).cuda()

                # Extraire les features avec CLIP
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    image_features = clip_model_(tile_tensor)
                logits = image_features
                probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()
                prediction = torch.argmax(logits, dim=1).item()

                # # Vérification si la classe prédit est 51 ou 52 (indices 5 et 6 dans le vecteur de probas)
                # if prediction in [99999]:
                #     clip_model_.load_state_dict(torch.load(weight_path_51_52))
                #     with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                #         specialized_logits = specialized_model(tile_tensor)
                #     specialized_probs = torch.softmax(specialized_logits, dim=1).cpu().numpy().flatten()
                #     specialized_prediction = torch.argmax(specialized_logits, dim=1).item()
                    
                #     # Mise à jour des probabilités
                #     probabilities[5] = specialized_probs[0]  # Probabilité de la classe 51
                #     probabilities[6] = specialized_probs[1]  # Probabilité de la classe 52
                #     probabilities /= probabilities.sum()  # Normalisation
                    
                #     # Mise à jour de la prédiction
                #     prediction = 5 if specialized_prediction == 0 else 6
                #     image_features = specialized_logits  # Mise à jour des embeddings
                #     clip_model_.load_state_dict(torch.load(weight_path))
                # Sauvegarde des résultats
                data.append({
                    "x": x,
                    "y": y,
                    "prediction": prediction,
                    "probabilities": probabilities.tolist(),
                    "embedding": image_features.cpu().numpy().tolist()
                })
            
            # Convertir en DataFrame Polars
            df = pl.DataFrame(data)
            output_path = f"wsi_results/wsi_image_results_{i}.parquet"
            i += 1
            df.write_parquet(output_path)


            


   

    print(f"Résultats enregistrés dans {output_path}")




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
