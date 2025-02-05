import sys
import pkg_resources
import subprocess

def check_cuda():
    """Vérifie si CUDA est disponible"""
    try:
        import torch
        print("\nVérification CUDA:")
        print(f"CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Version CUDA: {torch.version.cuda}")
            print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"Erreur lors de la vérification CUDA: {str(e)}")

def check_dependencies():
    """Vérifie si toutes les dépendances sont installées"""
    requirements = [
        'numpy',
        'pandas',
        'scikit-learn',
        'torch',
        'torchvision',
        'tqdm',
        'Pillow',
        'flask',
        'flask-cors',
        'werkzeug',
        'gunicorn',
        'python-dotenv',
        'matplotlib',
        'seaborn',
        'opencv-python',
        'albumentations',
        'pathlib',
        'python-json-logger',
        'tensorboard'
    ]

    print("Vérification des dépendances installées:")
    missing = []
    
    for package in requirements:
        try:
            dist = pkg_resources.get_distribution(package)
            print(f"{package}: {dist.version} ✓")
        except pkg_resources.DistributionNotFound:
            print(f"{package}: Non installé ✗")
            missing.append(package)
    
    return missing

def main():
    print("Vérification de l'environnement Python...")
    print(f"Version Python: {sys.version}")
    
    missing_packages = check_dependencies()
    check_cuda()
    
    if missing_packages:
        print("\nPaquets manquants:")
        print('\n'.join(f"- {pkg}" for pkg in missing_packages))
        print("\nPour installer les paquets manquants, exécutez:")
        print("pip install -r requirements.txt")
    else:
        print("\nToutes les dépendances sont installées! ✓")

if __name__ == '__main__':
    main()