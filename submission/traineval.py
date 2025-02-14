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

# Liste des attributs à analyser pour chaque cellule
att_names = [
    "cell_size",           # Taille de la cellule
    "cell_shape",          # Forme de la cellule
    "nucleus_shape",       # Forme du noyau
    "nuclear_cytoplasmic_ratio",  # Ratio noyau/cytoplasme
    "chromatin_density",   # Densité de la chromatine
    "cytoplasm_vacuole",   # Vacuoles dans le cytoplasme
    "cytoplasm_texture",   # Texture du cytoplasme
    "cytoplasm_colour",    # Couleur du cytoplasme
    "granule_type",        # Type de granules
    "granule_colour",      # Couleur des granules
    "granularity",         # Granularité
]

def save_args(savedir, args, name="args.json"):
    """
    Sauvegarde les arguments de configuration dans un fichier JSON
    
    Args:
        savedir: Répertoire de sauvegarde
        args: Arguments à sauvegarder
        name: Nom du fichier (par défaut 'args.json')
    """
    path = os.path.join(savedir, name)
    with open(path, "w") as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    print("Arguments sauvegardés dans %s" % path)

def save_json(dict_data, path):
    """
    Sauvegarde un dictionnaire au format JSON
    
    Args:
        dict_data: Dictionnaire à sauvegarder
        path: Chemin du fichier de sortie
    """
    with open(path, "w") as f:
        json.dump(dict_data, f, sort_keys=True, indent=4)
        print("Données sauvegardées dans %s" % path)

def save_checkpoint(path, model, key="model"):
    """
    Sauvegarde l'état du modèle dans un point de contrôle
    
    Args:
        path: Chemin de sauvegarde
        model: Modèle à sauvegarder
        key: Clé du dictionnaire d'état
    """
    checkpoint = {}
    checkpoint[key] = model.state_dict()
    torch.save(checkpoint, path)
    print("Point de contrôle sauvegardé dans", path)

def resume_model(model, resume, state_dict_key="model"):
    """
    Charge un modèle préentraîné à partir d'un point de contrôle
    
    Args:
        model: Modèle PyTorch
        resume: Chemin vers le fichier de reprise
        state_dict_key: Clé du dictionnaire d'état
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
        print("Chargé uniquement:", pretrained_dict.keys())
        print("Ignoré:", pretrained_dict_ignored.keys())

    return model

def make_deterministic(seed):
    """
    Configure tous les générateurs aléatoires pour assurer la reproductibilité
    
    Args:
        seed: Graine aléatoire
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_transforms(split, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Renvoie les transformations d'images pour l'entraînement ou le test
    
    Args:
        split: 'train' ou 'test'
        mean: Moyenne pour la normalisation
        std: Écart-type pour la normalisation
    """
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

def get_image_encoder(name, pretrained=True):
    """
    Crée et configure l'encodeur d'images
    
    Args:
        name: Nom du modèle ('resnet50', 'vgg16', etc.)
        pretrained: Utiliser des poids préentraînés
    """
    weights = "DEFAULT" if pretrained else None
    model = getattr(torchvision.models, name)(weights=weights)
    
    # Configuration de la couche finale selon l'architecture
    if name.startswith("vgg"):
        model.classifier[6] = nn.Identity()
    elif name.startswith("resnet"):
        model.fc = nn.Identity()
    elif name.startswith("vit"):
        model.heads = nn.Identity()
    elif name.startswith("convnext"):
        model.classifier[-1] = nn.Identity()
    else:
        raise ValueError(f"Encodeur d'images non supporté: {name}")
    
    # Détermine la dimension de sortie
    with torch.inference_mode():
        out = model(torch.randn(5, 3, 224, 224))
    assert out.dim() == 2
    assert out.size(0) == 5
    image_encoder_output_dim = out.size(1)
    
    return model, image_encoder_output_dim

def calculate_metrics(true_labels, predicted_probs):
    """
    Calcule les métriques d'évaluation
    
    Args:
        true_labels: Étiquettes réelles
        predicted_probs: Probabilités prédites
    """
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

def save_predictions_to_csv(predictions, log_dir, filename, dataloader_val):
    """
    Sauvegarde les prédictions dans un fichier CSV
    """
    decoded_predictions = []
    for j, preds in enumerate(predictions):
        attribute = dataloader_val.dataset.attribute_columns[j]
        encoder = dataloader_val.dataset.attribute_encoders[attribute]
        decoder = {v: k for k, v in encoder.items()}
        decoded_preds = [decoder[np.argmax(p)] for p in preds]
        decoded_predictions.append(decoded_preds)
    
    data = {"image_path": dataloader_val.dataset.df["path"]}
    data.update(
        {
            attribute: preds
            for attribute, preds in zip(
                dataloader_val.dataset.attribute_columns, decoded_predictions
            )
        }
    )
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(log_dir, filename), index=False)

def evaluate(model, dataloader):
    """
    Évalue le modèle sur un jeu de données
    """
    model.eval()
    num_attributes = len(dataloader.dataset.attribute_columns)
    all_predictions = [[] for _ in range(num_attributes)]
    all_probabilities = [[] for _ in range(num_attributes)]
    all_true_labels = [[] for _ in range(num_attributes)]
    all_image_paths = []
    
    with torch.inference_mode():
        for i, data in enumerate(dataloader):
            images, attributes = data["image"], data["attributes"]
            images, attributes = images.cuda(), attributes.cuda()
            model_outputs = model(images)
            
            for j, model_output in enumerate(model_outputs):
                model_output = torch.softmax(model_output, dim=1)
                all_probabilities[j].extend(model_output.cpu().tolist())
                all_predictions[j].extend(
                    torch.argmax(model_output, dim=1).cpu().tolist()
                )
                all_true_labels[j].extend(attributes[:, j].cpu().tolist())
            all_image_paths.extend(data["img_path"])
    
    # Calcul des métriques
    _, initial_metrics = next(
        iter(dataloader.dataset.attribute_columns)
    ), calculate_metrics([0, 1, 1], [[0.98, 0.02], [0.02, 0.98], [0.60, 0.40]])
    
    overall_metrics = {metric: 0.0 for metric in initial_metrics.keys()}
    per_attribute_metrics = {
        column: {} for column in dataloader.dataset.attribute_columns
    }
    
    for j, attribute in enumerate(dataloader.dataset.attribute_columns):
        metrics = calculate_metrics(all_true_labels[j], all_probabilities[j])
        for metric in overall_metrics.keys():
            overall_metrics[metric] += metrics[metric]
        per_attribute_metrics[attribute] = metrics
    
    for metric in overall_metrics.keys():
        overall_metrics[metric] /= num_attributes
    
    return {
        "overall_metrics": overall_metrics,
        "per_attribute_metrics": per_attribute_metrics,
        "all_image_paths": all_image_paths,
        "all_probabilities": all_probabilities,
        "all_predictions": all_predictions,
    }

def main(args):
    """
    Fonction principale d'entraînement et d'évaluation
    """
    print("Configuration:", args)
    make_deterministic(args.seed)

    # Création du répertoire de logs
    log = {}
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    log_path = os.path.join(args.logdir, "log.json")
    save_json(log, log_path)
    save_args(args.logdir, args)

    # Configuration de l'encodeur d'images
    image_encoder, image_encoder_output_dim = get_image_encoder(args.backbone)

    # Préparation des jeux de données
    dataset_train = AttDataset(
        args.train,
        att_names,
        image_dir=args.image_dir,
        transform=get_transforms("train"),
        multiply=args.epoch_multiply,
    )
    
    dataset_val = AttDataset(
        args.val,
        att_names,
        image_dir=args.image_dir,
        transform=get_transforms("test"),
        attribute_encoders=dataset_train.attribute_encoders,
    )
    
    dataset_test = AttDataset(
        args.test,
        att_names,
        image_dir=args.image_dir,
        transform=get_transforms("test"),
        attribute_encoders=dataset_train.attribute_encoders,
    )

    # Vérification de la cohérence des attributs
    attribute_sizes = [
        len(encoder) for encoder in dataset_train.attribute_encoders.values()
    ]
    
    for column in dataset_val.attribute_columns + dataset_test.attribute_columns:
        for value in sorted(dataset_val.df[column].unique()):
            assert value in dataset_train.attribute_encoders[column], \
                f"Valeur d'attribut '{value}' dans la colonne '{column}' non trouvée dans le jeu d'entraînement"

    # Création des dataloaders
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        pin_memory=True,
        drop_last=True,
    )
    
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        pin_memory=True,
        drop_last=False,
    )
    
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=(args.workers > 0),
        pin_memory=True,
        drop_last=False,
    )

    # Création et configuration du modèle
    model = AttributePredictor(attribute_sizes, image_encoder_output_dim, image_encoder)
    if args.resume is not None:
        model = resume_model(model, args.resume, state_dict_key="model")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.decay
    )

    model.cuda()
    best_val_metric = 0
    model_data_in_memory = None
    best_epoch = 0

    # Boucle d'entraînement
    training_logs = []
    for epoch in range(args.epochs):
        if not args.use_eval_mode:
            model.train()
        running_loss = 0.0
        num_processed_samples = 0
        
        # Barre de progression pour chaque époque
        with tqdm(
            dataloader_train, desc=f"Époque {epoch + 1}/{args.epochs}", unit="batch"
        ) as t:
            for i, data in enumerate(t):
                images, attribute_targets = data["image"], data["attributes"]
                images, attribute_targets = images.cuda(), attribute_targets.cuda()
                
                # Réinitialisation des gradients
                optimizer.zero_grad()
                
                # Forward pass
                attribute_outputs = model(images)
                
                # Calcul de la perte
                loss = 0
                for idx, (output, target) in enumerate(
                    zip(attribute_outputs, attribute_targets.t())
                ):
                    loss += criterion(output, target)
                
                # Moyenne de la perte sur tous les attributs
                loss = loss / len(attribute_outputs)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Mise à jour des statistiques
                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                num_processed_samples += batch_size
                
                # Mise à jour de la barre de progression
                t.set_postfix(train_loss=(running_loss / num_processed_samples))
        
        # Calcul de la perte moyenne sur l'époque
        training_loss = running_loss / num_processed_samples
        
        # Évaluation sur l'ensemble de validation
        evaluation_results = evaluate(model, dataloader_val)
        overall_metrics, per_attribute_metrics, _, _, _ = evaluation_results.values()
        
        print(
            f"Époque {epoch + 1}, Métriques globales sur validation: "
            + ", ".join([f"{k}: {(100*v):.2f}" for k, v in overall_metrics.items()])
        )
        
        # Vérification si c'est le meilleur modèle
        val_metric = overall_metrics[args.eval_metric]
        if val_metric > best_val_metric:
            best_epoch = epoch
            best_val_metric = val_metric
            # Sauvegarde du modèle en mémoire
            model_data_in_memory = BytesIO()
            torch.save(model.state_dict(), model_data_in_memory, pickle_protocol=-1)
            model_data_in_memory.seek(0)

        # Enregistrement des logs d'entraînement
        training_log = {
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "evaluation": evaluation_results,
        }
        training_logs.append(training_log)
        log["training"] = training_logs
        save_json(log, log_path)

    print("Meilleure époque:", best_epoch + 1)
    
    # Évaluation du meilleur modèle sur l'ensemble de test
    if model_data_in_memory is not None:
        model_in_cpu = torch.load(model_data_in_memory, map_location="cpu")
        model_data_in_memory.close()
        model.load_state_dict(model_in_cpu)
        model.cuda()
        
    evaluation_results = evaluate(model, dataloader_test)
    overall_metrics_best = evaluation_results["overall_metrics"]
    best_decoded_predictions = evaluation_results["all_probabilities"]
    
    print(
        f"Époque {epoch + 1}, Métriques globales du meilleur modèle sur test: "
        + ", ".join([f"{k}: {(100*v):.2f}" for k, v in overall_metrics_best.items()])
    )
    
    # Sauvegarde des prédictions
    save_predictions_to_csv(
        best_decoded_predictions,
        args.logdir,
        "bestval_epoch_predictions.csv",
        dataloader_test,
    )
    
    log["bestval"] = evaluation_results
    save_json(log, log_path)
    
    # Sauvegarde du meilleur modèle
    best_model_path = os.path.join(args.logdir, "best_model.pth")
    save_checkpoint(best_model_path, model, key="model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne un modèle de prédiction d'attributs de cellules.")
    
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
    
    parser.add_argument(
        "--image_dir",
        default="./data/PBC/",
        help="Répertoire racine contenant les fichiers images"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Nombre d'époques d'entraînement"
    )
    
    parser.add_argument(
        "--epoch_multiply",
        type=int,
        default=1,
        help="Nombre de répétitions du dataset dans chaque époque"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Taux d'apprentissage"
    )
    
    parser.add_argument(
        "--decay",
        type=float,
        default=0.01,
        help="Décroissance des poids"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Taille du batch"
    )
    
    parser.add_argument(
        "--eval_metric",
        default="f1_macro",
        help="Métrique d'évaluation pour la sélection du modèle"
    )
    
    parser.add_argument(
        "--backbone",
        default="resnet50",
        help="Choix de l'encodeur d'images",
        choices=["vgg16", "resnet50", "convnext_tiny", "vit_b_16"]
    )
    
    parser.add_argument(
        "--use_eval_mode",
        action="store_true",
        help="Utiliser model.eval() même pour l'entraînement. Améliore souvent les performances avec ResNet."
    )
    
    parser.add_argument(
        "--resume",
        default=None,
        help="Chemin vers un modèle pré-entraîné"
    )
    
    parser.add_argument(
        "--logdir",
        default="./log",
        help="Répertoire pour sauvegarder les logs d'expérience"
    )
    
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Graine aléatoire"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Nombre de workers pour torch.utils.data.DataLoader"
    )
    
    args = parser.parse_args()
    main(args)
        
