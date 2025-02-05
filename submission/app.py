from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from attribute_predictor import AttributePredictor

app = Flask(__name__)
CORS(app)

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
//A MODIFIER ICI
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

# Création des dossiers nécessaires
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Configuration du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

# Liste des attributs à prédire
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

# Dictionnaire de décodage des attributs (à adapter selon votre modèle)
attribute_decoders = {
    "cell_size": {0: "small", 1: "medium", 2: "large"},
    "cell_shape": {0: "round", 1: "oval", 2: "irregular"},
    "nucleus_shape": {0: "round", 1: "oval", 2: "irregular"},
    # Ajoutez les autres attributs selon votre modèle
}

def get_image_encoder(name="resnet50", pretrained=True):
    """
    Crée et configure l'encodeur d'images
    """
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, name)(weights=weights)
    
    if name.startswith("resnet"):
        model.fc = nn.Identity()
        output_dim = 2048  # Pour ResNet50
    elif name.startswith("vgg"):
        model.classifier[6] = nn.Identity()
        output_dim = 4096  # Pour VGG16
    elif name.startswith("convnext"):
        model.classifier[-1] = nn.Identity()
        output_dim = 768  # Pour ConvNext Tiny
    else:
        raise ValueError(f"Encodeur non supporté: {name}")
    
    return model, output_dim

def load_model():
    """
    Charge le modèle s'il n'est pas déjà en mémoire
    """
    global model
    if model is None:
        try:
            print("Chargement du modèle...")
            
            # Vérification de l'existence du fichier
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Modèle non trouvé: {MODEL_PATH}")
            
            # Chargement du checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Création de l'encodeur
            image_encoder, output_dim = get_image_encoder("resnet50")
            
            # Création du modèle
            # Vous devrez ajuster ces valeurs selon votre modèle
            attribute_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]  # exemple
            model = AttributePredictor(attribute_sizes, output_dim, image_encoder)
            
            # Chargement des poids
            model.load_state_dict(checkpoint['model'])
            model.to(device)
            model.eval()
            
            print("Modèle chargé avec succès!")
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

def allowed_file(filename):
    """
    Vérifie si le type de fichier est autorisé
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    Prétraite l'image pour le modèle
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        return image_tensor
    except Exception as e:
        raise ValueError(f"Erreur lors du traitement de l'image: {str(e)}")

def decode_predictions(outputs):
    """
    Décode les sorties du modèle en prédictions intelligibles
    """
    predictions = {}
    
    for i, output in enumerate(outputs):
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        attribute = att_names[i]
        decoded_value = "Unknown"
        if attribute in attribute_decoders:
            decoded_value = attribute_decoders[attribute].get(pred_class, "Unknown")
        
        predictions[attribute] = {
            'class': decoded_value,
            'raw_class': pred_class,
            'confidence': float(confidence)
        }
    
    return predictions

@app.route('/health', methods=['GET'])
def health_check():
    """
    Point de terminaison pour vérifier que l'API fonctionne
    """
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Point de terminaison principal pour l'analyse d'images
    """
    # Vérification du fichier
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'Type de fichier non autorisé'}), 400
    
    try:
        # Sauvegarde temporaire du fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Chargement du modèle si nécessaire
        load_model()
        
        # Prétraitement de l'image
        image_tensor = process_image(filepath)
        image_tensor = image_tensor.to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Décodage des prédictions
        predictions = decode_predictions(outputs)
        
        # Nettoyage
        os.remove(filepath)
        
        return jsonify({
            'status': 'success',
            'filename': filename,
            'predictions': predictions
        })
        
    except Exception as e:
        # Nettoyage en cas d'erreur
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """
    Gestionnaire global d'erreurs
    """
    return jsonify({
        'status': 'error',
        'error': str(e)
    }), 500

if __name__ == '__main__':
    # Configuration du logger
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Démarrage du serveur
    print(f"Démarrage du serveur sur le port 5000...")
    print(f"Utilisation du dispositif: {device}")
    app.run(debug=True, host='0.0.0.0', port=5000)
