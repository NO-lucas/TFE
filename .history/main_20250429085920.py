import os
import timm
import clip
import torch
import open_clip
from datasets import build_dataset
import torchvision.transforms as transforms
from datasets.utils import build_data_loader
from lora import run_uni, run_uni_lora, run_uni_lora_percent, run_process_wsi, pipeline, run_lora_optuna
from run_utils import set_random_seed, get_arguments
from transformers import AutoModelForImageClassification, AutoImageProcessor
from features import (
    features_extractor,
    FeaturesDataset,
    textual_extractor,
)


def main():

    args = get_arguments()

    set_random_seed(args.seed)

    # -------------------------------- Models --------------------------------
    _, preprocess = clip.load(args.backbone)
    tokenizer = None

    if args.model_name == "clip":
        model_clip, preprocess = clip.load(args.backbone)

    elif args.model_name == "quilt":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:wisdomik/QuiltNet-B-32"
        )

    elif args.model_name == "uni":
        model_clip = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )

    elif args.model_name == "vit_google":
        _ = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model_clip = AutoModelForImageClassification.from_pretrained(
            "google/vit-base-patch16-224"
        )

    elif args.model_name == "biomedclip":
        model_clip, preprocess, _ = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )
        tokenizer = open_clip.get_tokenizer(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
        )

    else:
        raise RuntimeError(
            "Wrong model name used. Try clip, uni, biomedclip, vit_google or quilt."
        )

    model_clip.eval()
    model_clip.cuda()
    logit_scale = 100

    # ---------------------------- Prepare dataset ----------------------------
    print("Preparing dataset.")

    train_tranform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=224,
                scale=(0.08, 1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

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
    elif args.task == "image_classifier":
        
        test_loader = None
    
    elif args.task == "pipeline":
        
        
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
    elif args.task == "image_classifier":
        
        run_process_wsi(args, model_clip,'/wsi_image_results', test_loader)
    
    elif args.task == "pipeline":
        
        pipeline(args, model_clip,'/wsi_image_results')
    elif args.task == "optuna":
        import optuna
        from optuna.pruners import HyperbandPruner
        from optuna.samplers import TPESampler

        def objective(trial):
            # Suggest hyperparameters
            lr = trial.suggest_loguniform("lr", 1e-6, 1e-3)
            weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)
            eta_min = trial.suggest_loguniform("eta_min", 1e-6, 1e-3)
            n_iters = trial.suggest_int("n_iters", 100, 1000)
            r = trial.suggest_categorical("r", [2, 4, 8, 16])
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            dropout_rate = trial.suggest_uniform("dropout_rate", 0.0, 0.5)

            # Configurer les arguments du modèle
            args.lr = lr
            args.weight_decay = weight_decay
            args.eta_min = eta_min
            args.n_iters = n_iters
            args.r = r
            args.batch_size = batch_size
            args.dropout_rate = dropout_rate

            # Entraîner le modèle
            val_acc = run_lora_optuna(args, model_clip, logit_scale, train_loader, val_loader, test_loader)

            return val_acc
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(n_startup_trials=2),
            pruner=HyperbandPruner()
        )
        study.optimize(objective, n_trials=100)



    else:
        print("Wrong task name")


if __name__ == "__main__":
    main()
