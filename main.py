import os
import timm
import clip
import torch
import open_clip
from datasets import build_dataset
import torchvision.transforms as transforms
from datasets.utils import build_data_loader
from lora import run_uni, run_uni_lora, run_uni_lora_percent, run_process_wsi, pipeline, run_lora_optuna, evaluate_lora_uni
from run_utils import set_random_seed, get_arguments
from transformers import AutoModelForImageClassification, AutoImageProcessor
from transformers import ViTImageProcessor, ViTForImageClassification
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
import logging
import sys
from copy import deepcopy
from features import (
    features_extractor,
    FeaturesDataset,
    textual_extractor,
)

import torch.nn as nn
from loralib.utils import (
    mark_only_lora_as_trainable,
    apply_lora,
    get_lora_parameters,
    save_lora,
    load_lora
)


class HuggingfaceTransformWrapper:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)


def main():

    # biomedclip, quilt, pubmedclip, vit, dinobloom B et dinobloom L

    args = get_arguments()

    set_random_seed(args.seed)

    # -------------------------------- Models --------------------------------
    # _, preprocess = clip.load(args.backbone)
    tokenizer = None

    if args.model_name == "clip":
        model_clip, preprocess = clip.load(args.backbone)
        tokenizer = clip.tokenize

    elif args.model_name == "quilt":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:wisdomik/QuiltNet-B-32"
        )
        tokenizer = open_clip.get_tokenizer("hf-hub:wisdomik/QuiltNet-B-32")

    elif args.model_name == "uni":
        from huggingface_hub import login
        login(token="YOUR_TOKEN")

        model_clip = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )

        # transform = transforms.Compose(
        #     [
        #         transforms.Resize(224),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     ]
        # )

        # data_config = timm.data.resolve_model_data_config(model_clip)
        # preprocess = timm.data.create_transform(**data_config, is_training=False)
        preprocess = create_transform(**resolve_data_config(model_clip.pretrained_cfg, model=model_clip))

    elif args.model_name == "vit_google":
        model_clip = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
 
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        preprocess = HuggingfaceTransformWrapper(processor)

    elif args.model_name == "biomedclip":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
    
    elif args.model_name == "pubmedclip":
        from transformers import CLIPModel
        _, preprocess = clip.load(args.backbone)
 
        model_clip = CLIPModel.from_pretrained(
            "flaviagiammarino/pubmed-clip-vit-base-patch32"
        )

    elif args.model_name == "dinobloom":
        model_clip = timm.create_model(
            model_name="hf-hub:1aurent/vit_large_patch14_224.dinobloom",
            pretrained=True,
        )

        # Preprocess for ViT-Google
        data_config = timm.data.resolve_model_data_config(model_clip)
        preprocess = timm.data.create_transform(**data_config, is_training=False)

    else:
        raise RuntimeError(
            "Wrong model name used. Try clip, uni, biomedclip, vit_google or quilt."
        )
    
    input_size = data_config["input_size"][-2] if args.model_name == "dinobloom" else 224
    da_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=input_size,
                scale=(0.7, 1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    )

    train_tranform = transforms.Compose(
        [preprocess, da_transform]
    )

    train_tranform = transforms.Compose(
        [da_transform, preprocess]
    )

    model_clip.eval()
    model_clip.cuda()
    logit_scale = 100

    # ---------------------------- Prepare dataset ----------------------------
    print("Preparing dataset.")

    level_name = (args.level).replace("_", "")

    if args.task == "classifier":
        features_csv_train = os.path.join(
            args.root_path, args.dataset + "_" + args.model_name + "_features_train.npz"
        )
        features_csv_val = os.path.join(
            args.root_path, args.dataset + "_" + args.model_name + "_features_val.npz"
        )
        features_csv_test = os.path.join(
            args.root_path, args.dataset + "_" + args.model_name + "_features_test.npz"
        )

        if (
            not os.path.exists(features_csv_train)
            or not os.path.exists(features_csv_val)
            or not os.path.exists(features_csv_test)
        ):
            dataset = build_dataset(args.dataset, args.root_path, -1, args.level)

        textual_csv_train = os.path.join(
            args.root_path, args.dataset + "_" + args.model_name + "_textual_train.npz"
        )

        if not os.path.exists(textual_csv_train) and args.textual == "True":
            dataset = build_dataset(args.dataset, args.root_path, -1, args.level)

            textual_extractor(args, dataset, model_clip, tokenizer)

        val_loader = build_data_loader(
            data_source=dataset.val,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        train_loader = None
        if not args.eval_only:

            train_loader = build_data_loader(
                data_source=dataset.train_x,
                batch_size=args.batch_size,
                tfm=train_tranform,
                is_train=True,
                shuffle=True,
                num_workers=5,
            )

        features_extractor(args, model_clip, train_loader, val_loader, test_loader)

        train_dataset = FeaturesDataset(
            os.path.join(
                args.root_path,
                args.dataset + "_" + args.model_name + "_features_train.npz",
            )
        )
        val_dataset = FeaturesDataset(
            os.path.join(
                args.root_path,
                args.dataset + "_" + args.model_name + "_features_val.npz",
            )
        )
        test_dataset = FeaturesDataset(
            os.path.join(
                args.root_path,
                args.dataset + "_" + args.model_name + "_features_test.npz",
            )
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=5,
            shuffle=True,
            pin_memory=True,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=5,
            shuffle=True,
            pin_memory=True,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            num_workers=5,
            shuffle=True,
            pin_memory=True,
        )

    elif args.task == "lora":

        if args.dataset == "hicervix":
            pt_path = (
                "./"
                + str(args.dataset)
                + "_"
                + str(args.seed)
                + "_"
                + str(args.shots)
                + "_"
                + str(level_name)
                + ".pt"
            )

            if not os.path.exists(pt_path):
                # Doing this to save time.
                os.system(
                    f"python3 dataset_hicervix.py --seed_launch {args.seed} --shots_launch {args.shots} --level_launch {args.level}"
                )

            dataset = torch.load(pt_path, weights_only=False)
        else:
            dataset = build_dataset(args.dataset, args.root_path, args.shots)

        val_loader = build_data_loader(
            data_source=dataset.val,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        train_loader = build_data_loader(
            data_source=dataset.train_x,
            batch_size=args.batch_size,
            tfm=train_tranform,
            is_train=True,
            shuffle=True,
            num_workers=5,
        )

    elif args.task == "percentage_lora":

        assert args.percentage > 0, "The percentage should be greater than zero."

        if args.dataset == "hicervix":
            pt_path = (
                "./"
                + str(args.dataset)
                + "_"
                + str(args.seed)
                + "_"
                + str(args.shots)
                + "_"
                + str(level_name)
                + "_"
                + str(args.percentage)
                + "_percent.pt"
            )

            if not os.path.exists(pt_path):
                # Doing this to save time.
                os.system(
                    f"python3 dataset_hicervix.py --seed_launch {args.seed} --shots_launch {args.shots} --level_launch {args.level} --percent_launch {args.percentage}"
                )

            dataset = torch.load(pt_path, weights_only=False)
        else:
            print("Percentage experiment was not implemented for the other datasets.")

        val_loader = build_data_loader(
            data_source=dataset.val,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        train_loader = build_data_loader(
            data_source=dataset.train_x,
            batch_size=args.batch_size,
            tfm=train_tranform,
            is_train=True,
            shuffle=True,
            num_workers=5,
        )
    elif args.task =="inference":
        test_loaders = []

        for res in [1]:

            dataset = build_dataset(args.dataset, args.root_path, args.shots)


            test_loaders.append(build_data_loader(
                data_source=dataset.test,
                batch_size=256,
                is_train=False,
                tfm=preprocess,
                shuffle=False,
                num_workers=5,
            ))

    elif args.task == "image_classifier":
      
        dataset = build_dataset(args.dataset, args.root_path, args.shots)


        val_loader = build_data_loader(
            data_source=dataset.val,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=256,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=5,
        )

        train_loader = build_data_loader(
            data_source=dataset.train_x,
            batch_size=args.batch_size,
            tfm=train_tranform,
            is_train=True,
            shuffle=True,
            num_workers=5,
        )


    elif args.task == "pipeline":
        test_loader= None
    elif args.task == "optuna":
        
        dataset = build_dataset(args.dataset, args.root_path, args.shots)

        val_loader = build_data_loader(
            data_source=dataset.val,
            batch_size=128,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=2,
        )

        test_loader = build_data_loader(
            data_source=dataset.test,
            batch_size=128,
            is_train=False,
            tfm=preprocess,
            shuffle=False,
            num_workers=2,
        )

        train_loader = build_data_loader(
            data_source=dataset.train_x,
            batch_size=16,
            tfm=train_tranform,
            is_train=True,
            shuffle=True,
            num_workers=2,
        )

    else:
        print("We are in the wrong situation")


    print(f"\n Running {args.task} ... \n")
    # Classifier experiment
    if args.task == "classifier":
        run_uni(args, model_clip, logit_scale, train_loader, val_loader, test_loader)

    # LoRA experiment
    elif args.task == "lora":
        run_uni_lora(
            args, model_clip, logit_scale, train_loader, val_loader, test_loader
        )

    # Percentage - LoRA experiment
    elif args.task == "percentage_lora":
        run_uni_lora_percent(
            args, model_clip, logit_scale, train_loader, val_loader, test_loader
        )
    elif args.task =="inference":
        args.lr = 0.000053439003527267916
        args.weight_decay = 0.0012938768880920027
        # args.eta_min = eta_min
        args.n_iters = 4
        args.r = 4
        # args.batch_size = batch_size
        args.dropout_rate = 0.09087453174846884	

        model_dir = "best_optuna_models/"
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')][1]

        model_files = ["res1_data_aug2.pth"]

        num_features = 512


        

        for clip_model_name, test_loader, r in zip(model_files, test_loaders, [4]):
                
            model_linear = nn.Sequential(
                nn.Flatten(start_dim=1), nn.Linear(num_features, args.num_classes)
            ).cuda()
            args.r = r
            model_clip_copy, _ = clip.load(args.backbone)


            list_lora_layers = apply_lora(args, model_clip_copy)
            model_clip_copy = model_clip_copy.cuda()
            mark_only_lora_as_trainable(model_clip_copy)
            trainable_parameters_ = get_lora_parameters(model_clip_copy)

            for _, param in model_linear.named_parameters():
                trainable_parameters_.append(param)

            clip_model_ = nn.Sequential(model_clip_copy.visual, model_linear)



            clip_model_.load_state_dict(torch.load(model_dir + clip_model_name))
            
            acc_test, _ = evaluate_lora_uni(args, clip_model_, test_loader)

            print(f"**** Final test accuracy: {acc_test:.2f} ****")


    elif args.task == "image_classifier":
        
        run_process_wsi(args, model_clip,'/wsi_image_results', train_loader, val_loader, test_loader)

    elif args.task == "pipeline":
        
        pipeline(args, model_clip,'/wsi_image_results')
    elif args.task == "optuna":

        best_global_acc = [0.0]
        study_name = f"res1_text_newdataset_{args.model_name}_notext"  # Unique identifier of the study.
        

        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-1)
            # eta_min = trial.suggest_loguniform("eta_min", 1e-6, 1e-3)
            n_iters = trial.suggest_int("n_iters", 50, 300)
            r = trial.suggest_categorical("r", [2, 4, 8, 16, 24, 36])
            # batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.2)

            # Configurer les arguments du modèle
            args.lr = lr
            args.weight_decay = weight_decay
            # args.eta_min = eta_min
            args.n_iters = n_iters
            args.r = r
            # args.batch_size = batch_size
            args.dropout_rate = dropout_rate

            # Entraîner le modèle
            
            model_clip_copy = deepcopy(model_clip)
            val_acc = run_lora_optuna(args, model_clip_copy, logit_scale, train_loader, val_loader, test_loader, trial, best_global_acc, study_name)

            return val_acc
        
        optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
        
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=30),
            pruner=HyperbandPruner(min_resource=10, max_resource=80, reduction_factor=3),
            storage=storage_name,  # Specify the storage URL here.
            study_name=study_name,
            load_if_exists = True
        )
        study.optimize(objective, n_trials=180)


    else:
        print("Wrong task name")


if __name__ == "__main__":
    main()
