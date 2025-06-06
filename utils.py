import clip
import torch
from tqdm import tqdm


def get_function(model_name, clip_model, tokenizer=None):
    MODEL_NAME = {}
    if model_name == "clip":
        MODEL_NAME = {
            "vision": clip_model.encode_image,
            "text": clip_model.encode_text,
            "token": clip.tokenize,
        }

    elif model_name == "quilt":
        MODEL_NAME = {
            "vision": clip_model.encode_image,
            "text": clip_model.encode_text,
            "token": clip.tokenize,
        }

    elif model_name == "biomedclip":
        MODEL_NAME = {
            "vision": clip_model.visual,
            "text": clip_model.text,
            "token": tokenizer,
        }

    elif model_name == "uni":

        MODEL_NAME = {"vision": clip_model, "text": None, "token": None}

    elif model_name == "vit_google":

        MODEL_NAME = {"vision": clip_model, "text": None, "token": None}

    return MODEL_NAME["vision"], MODEL_NAME["text"], MODEL_NAME["token"]


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[:topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]

    return acc


def clip_classifier(classnames, template, clip_model, model_name, tokenizer):

    vision, text, token = get_function(model_name, clip_model, tokenizer)

    with torch.no_grad():

        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace("_", " ")
            texts = [t.format(classname) for t in template]
            texts = token(texts).cuda()
            class_embeddings = text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()

    return clip_weights


def pre_load_features(clip_model, loader, model_name, tokenizer):

    vision, text, token = get_function(model_name, clip_model, tokenizer)
    features, labels = [], []

    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm(loader)):
            images, target = images.cuda(), target.cuda()
            image_features = vision(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            features.append(image_features.cpu())
            labels.append(target.cpu())

        features, labels = torch.cat(features), torch.cat(labels)

    return features, labels
