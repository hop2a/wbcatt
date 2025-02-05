from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from attribute_predictor import AttributePredictor

app = Flask(__name__)
CORS(app)

# Configuration des chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pth')

# Création des dossiers nécessaires
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configuration Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Configuration du modèle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None

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

def get_image_encoder(name="resnet50", pretrained=True):
    """Configure l'encodeur d'images"""
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, name)(weights=weights)
    
    if name.startswith("resnet"):
        model.fc = nn.Identity()
        output_dim = 2048
    elif name.startswith("vgg"):
        model.classifier[6] = nn.Identity()
        output_dim = 4096
    else:
        raise ValueError(f"Encodeur non supporté: {name}")
    
    return model, output_dim

def load_model():
    """Charge le modèle"""
    global model
    if model is None:
        try:
            print("Chargement du modèle...")
            
            # Chargement du checkpoint
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            
            # Configuration de l'encodeur
            image_encoder, output_dim = get_image_encoder("resnet50")
            
            # Création du modèle avec les dimensions appropriées
            attribute_sizes = [3] * len(att_names)  # 3 classes pour chaque attribut
            model = AttributePredictor(attribute_sizes, output_dim, image_encoder)
            
            # Chargement des poids
            model.load_state_dict(checkpoint['model'])
            model.to(device)
            model.eval()
            
            print("Modèle chargé avec succès!")
            
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            raise

def process_image(image_path):
    """Prétraite l'image"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

def decode_predictions(outputs):
    """Décode les prédictions"""
    predictions = {}
    
    # Décodage pour chaque attribut
    for i, attribute in enumerate(att_names):
        probs = torch.softmax(outputs[i], dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
        
        predictions[attribute] = {
            'class': pred_class,
            'confidence': float(confidence)
        }
    
    return predictions

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier envoyé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
    if file and file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
        try:
            # Sauvegarde temporaire du fichier
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Chargement du modèle si nécessaire
            load_model()
            
            # Prétraitement et prédiction
            image_tensor = process_image(filepath)
            image_tensor = image_tensor.to(device)
            
            with torch.no_grad():
                outputs = model(image_tensor)
            
            # Décodage des prédictions
            predictions = decode_predictions(outputs)
            
            # Nettoyage
            os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'predictions': predictions
            })
            
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'status': 'error',
                'error': str(e)
            }), 500
            
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

if __name__ == '__main__':
    print(f"Démarrage du serveur sur le port 5000...")
    print(f"Utilisation du dispositif: {device}")
    app.run(debug=True, host='0.0.0.0', port=5000)
