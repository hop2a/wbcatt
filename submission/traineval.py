import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from tqdm.auto import tqdm
from io import BytesIO
import numpy as np
import random

from att_dataset import AttDataset
from attribute_predictor import AttributePredictor

# Liste des attributs à prédire pour chaque cellule
att_names = [
    "cell_size",
    "cell_shape",
    "nucleus_shape",
    "nuclear_cytoplasmic_ratio",
    "chromatin_density",
    "cytoplasm_vacuole",
    "cytoplasm_texture",
    "cytoplasm_colour",
    "granule_type",
    "granule_colour",
    "granularity",
]

def save_args(savedir, args, name="args.json"):
    # Sauvegarde les arguments de configuration dans un fichier JSON
    path = os.path.join(savedir, name)
    with open(path, "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print("Arguments sauvegardés dans %s" % path)

def save_json(dict, path):
    # Sauvegarde un dictionnaire au format JSON
    with open(path, "w") as f:
        json.dump(dict, f, sort_keys=True, indent=4)
        print("Journal sauvegardé dans %s" % path)

def save_checkpoint(path, model, key="model"):
    # Sauvegarde l'état du modèle
    checkpoint = {}
    checkpoint[key] = model.state_dict()
    torch.save(checkpoint, path)
    print("Point de contrôle sauvegardé dans", path)

def resume_model(model, resume, state_dict_key="model"):
    """
    Charge un modèle préentraîné
    model: modèle PyTorch
    resume: chemin vers le fichier de reprise
    state_dict_key: clé du dictionnaire d'état
    """
    print("Chargement des poids depuis %s" % resume)

    checkpoint = torch.load(resume, map_location="cpu")
    if state_dict_key is not None:
        pretrained_dict = checkpoint[state_dict_key]
    else:
        pretrained_dict = checkpoint

    try:
        model.load_state_dict(pretrained_dict)
    except RuntimeError as e:
        print(e)
        print("Impossible de charger tous les poids! Tentative de chargement partiel...")
        model_dict = model.state_dict()
        # 1. Filtrer les clés non nécessaires
        pretrained_dict_use = {}
        pretrained_dict_ignored = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                pretrained_dict_use[k] = v
            else:
                pretrained_dict_ignored[k] = v
        pretrained_dict = pretrained_dict_use
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Chargé uniquement", pretrained_dict.keys())
        print("Ignoré:", pretrained_dict_ignored.keys())

    return model

def make_deterministic(seed):
    # Assure la reproductibilité des résultats
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_transforms(split, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # Définit les transformations d'images pour l'entraînement ou le test
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

def calculate_metrics(true_labels, predicted_probs):
    # Calcule les métriques d'évaluation (précision, rappel, F1-score)
    predicted_probs = np.array(predicted_probs)
    true_labels = np.array(true_labels)
    predicted_labels = np.argmax(predicted_probs, axis=1)
    metrics = {
        "acc": accuracy_score(true_labels, predicted_labels),
        "f1_macro": f1_score(true_labels, predicted_labels, average="macro"),
        "pre_macro": precision_score(true_labels, predicted_labels, average="macro"),
        "rec_macro": recall_score(true_labels, predicted_labels, average="macro"),
    }
    return metrics

# ... [Le reste du code garde la même structure avec commentaires en français]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîner un modèle de prédiction d'attributs de cellules.")
    parser.add_argument(
        "--train",
        default="./pbc_attr_v1_train.csv",
        help="Chemin vers le fichier CSV d'entraînement"
    )
    parser.add_argument(
        "--val",
        default="./pbc_attr_v1_val.csv",
        help="Chemin vers le fichier CSV de validation"
    )
    parser.add_argument(
        "--test",
        default="./pbc_attr_v1_test.csv",
        help="Chemin vers le fichier CSV de test"
    )
    # ... [Suite des arguments avec descriptions en français]
