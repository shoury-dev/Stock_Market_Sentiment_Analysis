import os
import sys
import subprocess
import platform

def run_command(command):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python requirements...")
    
    success, stdout, stderr = run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    if success:
        print("✅ Requirements installed successfully!")
    else:
        print(f"❌ Error installing requirements: {stderr}")
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("📚 Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✅ NLTK data downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {str(e)}")
        return False

def create_env_file():
    """Create .env file from template if it doesn't exist"""
    print("🔧 Setting up environment file...")
    
    if not os.path.exists('.env'):
        if os.path.exists('.env.example'):
            # Copy example file
            if platform.system() == "Windows":
                success, _, _ = run_command("copy .env.example .env")
            else:
                success, _, _ = run_command("cp .env.example .env")
            
            if success:
                print("✅ Created .env file from template")
                print("⚠️  Please edit .env file and add your API keys!")
                return True
            else:
                print("❌ Failed to create .env file")
                return False
        else:
            print("❌ .env.example file not found")
            return False
    else:
        print("✅ .env file already exists")
        return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/models',
        'logs',
        'exports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Directories created successfully!")
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("🧪 Testing imports...")
    
    required_modules = [
        'pandas', 'numpy', 'requests', 'nltk', 'textblob', 
        'vaderSentiment', 'streamlit', 'plotly', 'yfinance'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All required modules imported successfully!")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up Stock Market Sentiment Analyzer...")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Run setup steps
    steps = [
        ("Creating directories", create_directories),
        ("Installing requirements", install_requirements),
        ("Downloading NLTK data", download_nltk_data),
        ("Setting up environment file", create_env_file),
        ("Testing imports", test_imports)
    ]
    
    all_successful = True
    
    for step_name, step_function in steps:
        print(f"\n{step_name}...")
        success = step_function()
        if not success:
            all_successful = False
            print(f"❌ {step_name} failed!")
        else:
            print(f"✅ {step_name} completed!")
    
    print("\n" + "=" * 50)
    
    if all_successful:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Edit the .env file and add your API keys")
        print("2. Run: streamlit run streamlit_app.py")
        print("3. Open your browser to the displayed URL")
        print("\n🔗 Useful commands:")
        print("   - Start Streamlit app: streamlit run streamlit_app.py")
        print("   - Install additional packages: pip install <package_name>")
        print("   - Update requirements: pip freeze > requirements.txt")
    else:
        print("❌ Setup completed with errors!")
        print("Please check the error messages above and try again.")

if __name__ == "__main__":
    main()
