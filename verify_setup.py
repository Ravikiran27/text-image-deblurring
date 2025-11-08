"""
Setup Verification Script

This script verifies that the project is correctly set up and all dependencies are installed.

Usage:
    python verify_setup.py
"""

import sys
import os


def check_python_version():
    """Check if Python version is 3.10+"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 10:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be 3.10 or higher")
        return False


def check_dependencies():
    """Check if required packages are installed"""
    required_packages = {
        'tensorflow': 'TensorFlow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'sklearn': 'scikit-learn',
        'skimage': 'scikit-image',
        'matplotlib': 'Matplotlib',
        'streamlit': 'Streamlit',
        'PIL': 'Pillow'
    }
    
    print("\nChecking dependencies:")
    all_installed = True
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✅ {name} is installed")
        except ImportError:
            print(f"❌ {name} is NOT installed")
            all_installed = False
    
    return all_installed


def check_gpu():
    """Check if GPU is available"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if len(gpus) > 0:
            print(f"\n✅ GPU detected: {len(gpus)} device(s)")
            for gpu in gpus:
                print(f"   - {gpu.name}")
            return True
        else:
            print("\n⚠️  No GPU detected. Training will use CPU (slower)")
            return False
    except Exception as e:
        print(f"\n⚠️  Could not check GPU: {e}")
        return False


def check_project_structure():
    """Check if project directories exist"""
    required_dirs = [
        'data/blur',
        'data/orig',
        'models',
        'utils',
        'app',
        'saved_models'
    ]
    
    print("\nChecking project structure:")
    all_exist = True
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/ exists")
        else:
            print(f"❌ {dir_path}/ does NOT exist")
            all_exist = False
    
    return all_exist


def check_files():
    """Check if required files exist"""
    required_files = [
        'train.ipynb',
        'test.py',
        'requirements.txt',
        'README.md',
        'models/vgg16_autoencoder.py',
        'models/resnet_autoencoder.py',
        'utils/dataset_loader.py',
        'utils/metrics.py',
        'app/streamlit_app.py'
    ]
    
    print("\nChecking required files:")
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} does NOT exist")
            all_exist = False
    
    return all_exist


def test_model_import():
    """Test if model modules can be imported"""
    print("\nTesting model imports:")
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        from models.vgg16_autoencoder import build_vgg16_autoencoder
        print("✅ VGG16 autoencoder module can be imported")
        
        from models.resnet_autoencoder import build_resnet_autoencoder
        print("✅ ResNet50 autoencoder module can be imported")
        
        from utils.dataset_loader import DeblurDataset
        print("✅ Dataset loader module can be imported")
        
        from utils.metrics import calculate_psnr, calculate_ssim
        print("✅ Metrics module can be imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


def main():
    """Run all verification checks"""
    print("="*60)
    print("PROJECT SETUP VERIFICATION")
    print("="*60)
    
    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'GPU': check_gpu(),
        'Project Structure': check_project_structure(),
        'Required Files': check_files(),
        'Module Imports': test_model_import()
    }
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    for check, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{check}: {status}")
    
    all_passed = all(results.values())
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All checks passed! Project is ready to use.")
        print("\nNext steps:")
        print("1. Add your dataset to data/blur/ and data/orig/")
        print("2. Open train.ipynb in Jupyter/Kaggle to train the model")
        print("3. Run 'streamlit run app/streamlit_app.py' to launch the web app")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nTo install dependencies, run:")
        print("   pip install -r requirements.txt")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
