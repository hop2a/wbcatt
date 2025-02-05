from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import torch
from PIL import Image
from torchvision import transforms
import attribute_predictor  # votre module de prédiction
import torch.nn as nn

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'best_model.pth'  # chemin vers votre modèle entraîné

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Charger le modèle
model = None  # Sera initialisé à la première requête
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    global model
    if model is None:
        # Initialiser votre modèle ici
        model = attribute_predictor.AttributePredictor(...)  # vos paramètres
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        model.eval()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """Prétraitement de l'image selon vos besoins"""
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

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Vérifier si un fichier a été envoyé
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier n\'a été envoyé'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
        
    if file and allowed_file(file.filename):
        # Sauvegarder le fichier
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Charger le modèle si ce n'est pas déjà fait
            load_model()
            
            # Prétraiter l'image
            image_tensor = process_image(filepath)
            image_tensor = image_tensor.to(device)
            
            # Faire la prédiction
            with torch.no_grad():
                outputs = model(image_tensor)
            
            # Convertir les sorties en prédictions
            predictions = {}
            att_names = [
                "cell_size", "cell_shape", "nucleus_shape",
                "nuclear_cytoplasmic_ratio", "chromatin_density",
                "cytoplasm_vacuole", "cytoplasm_texture",
                "cytoplasm_colour", "granule_type",
                "granule_colour", "granularity"
            ]
            
            for i, output in enumerate(outputs):
                probs = torch.softmax(output, dim=1)
                pred_class = torch.argmax(probs, dim=1).item()
                predictions[att_names[i]] = {
                    'class': pred_class,
                    'confidence': probs[0][pred_class].item()
                }
            
            # Nettoyer le fichier temporaire
            os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'predictions': predictions
            })
            
        except Exception as e:
            return jsonify({
                'error': f'Erreur lors de l\'analyse: {str(e)}'
            }), 500
            
    return jsonify({'error': 'Type de fichier non autorisé'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)