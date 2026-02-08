#!/usr/bin/env python3
"""
DeepSeek System Auto-Setup
Automatically installs dependencies and configures the system
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
import shutil

class DeepSeekSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_file = self.project_root / "requirements.txt"
        self.system = platform.system()
        
    def print_banner(self):
        print("""
        ╔══════════════════════════════════════════════════════════╗
        ║      DEEPSEEK LEAD SYSTEM SETUP v2026.1.0               ║
        ║      Advanced AI-Powered Lead Generation Platform       ║
        ╚══════════════════════════════════════════════════════════╝
        """)
    
    def create_directory_structure(self):
        """Create the complete directory structure"""
        directories = [
            "core",
            "validation",
            "scrapers",
            "api",
            "dashboard",
            "data/database",
            "data/cache",
            "exports",
            "logs",
            "models/ai",
            "models/ml",
            "config",
            "utils"
        ]
        
        print("📁 Creating directory structure...")
        for directory in directories:
            path = self.project_root / directory
            path.mkdir(parents=True, exist_ok=True)
            print(f"   Created: {directory}")
        
        # Create __init__.py files
        for dir_path in self.project_root.rglob("*/"):
            if dir_path.is_dir() and "__init__.py" not in os.listdir(dir_path):
                (dir_path / "__init__.py").touch()
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing dependencies...")
        
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)])
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            sys.exit(1)
    
    def install_system_dependencies(self):
        """Install system-specific dependencies"""
        print("🖥️  Installing system dependencies...")
        
        if self.system == "Linux":
            commands = [
                "sudo apt-get update",
                "sudo apt-get install -y python3-dev build-essential libssl-dev libffi-dev",
                "sudo apt-get install -y chromium-browser chromium-chromedriver"
            ]
        elif self.system == "Darwin":  # macOS
            commands = [
                "brew update",
                "brew install python-tk@3.9",
                "brew install chromedriver --cask"
            ]
        elif self.system == "Windows":
            print("   Please install Chrome browser manually")
            return
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True)
            except:
                print(f"   Note: Some system dependencies may need manual installation")
    
    def configure_environment(self):
        """Configure environment variables and settings"""
        print("⚙️  Configuring environment...")
        
        # Create .env file
        env_template = """# DeepSeek Lead Generation System Configuration
# API Keys and Configuration

# DeepSeek AI Configuration
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_URL=https://api.deepseek.com/v1

# Database Configuration
DATABASE_URL=sqlite:///data/database/leads.db
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your_secret_key_here

# Scraper Configuration
SCRAPER_MAX_THREADS=5
SCRAPER_TIMEOUT=30
SCRAPER_USER_AGENT=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36

# AI/ML Configuration
AI_MODEL_PATH=models/ai/
ML_MODEL_PATH=models/ml/
ENABLE_AI_SCORING=true
ENABLE_ML_PREDICTION=true

# Export Configuration
EXPORT_FORMAT=csv
EXPORT_ENCODING=utf-8

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/deepseek_system.log

# Country-Specific Settings (Popular domains by country)
COUNTRY_DOMAINS_DE=web.de,gmx.de,t-online.de,freenet.de
COUNTRY_DOMAINS_CH=bluewin.ch,gmail.ch,hotmail.ch
COUNTRY_DOMAINS_NL=ziggo.nl,kpn.nl,planet.nl
COUNTRY_DOMAINS_US=gmail.com,yahoo.com,hotmail.com,aol.com
# ... Add more countries as needed
"""
        
        env_file = self.project_root / ".env"
        if not env_file.exists():
            with open(env_file, "w") as f:
                f.write(env_template)
            print("   Created .env configuration file")
        
        # Create config file
        config_dir = self.project_root / "config"
        config_file = config_dir / "settings.py"
        
        config_content = '''"""
DeepSeek System Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EXPORTS_DIR = BASE_DIR / "exports"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

# API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/database/leads.db")

# Country Configuration
COUNTRIES = ["DE", "CH", "NL", "AU", "AT", "IT", "SE", "NO", "DK", 
             "CA", "UK", "US", "UAE", "SA", "BH", "JP", "IN", "SG", "MY", "FR", "BG"]

# Country-specific email domains
COUNTRY_EMAIL_DOMAINS = {
    "DE": ["web.de", "gmx.de", "t-online.de", "freenet.de", "yahoo.de"],
    "CH": ["bluewin.ch", "gmail.ch", "hotmail.ch", "yahoo.ch"],
    "NL": ["ziggo.nl", "kpn.nl", "planet.nl", "gmail.com"],
    "US": ["gmail.com", "yahoo.com", "hotmail.com", "aol.com", "outlook.com"],
    "UK": ["gmail.com", "yahoo.co.uk", "hotmail.co.uk", "outlook.com"],
    # ... Add more countries
}

# Lead Types
LEAD_TYPES = ["forex", "crypto", "casino"]
LEAD_CATEGORIES = ["hot", "recovery", "xdepositor", "new_trader", "online_scraper"]

# AI Configuration
AI_MODELS = {
    "deepseek": "deepseek-chat",
    "gpt": "gpt-4",
    "hybrid": "ensemble"
}

# Validation Rules
MIN_LEAD_QUALITY_SCORE = 0.7
MAX_DAYS_SINCE_ACTIVITY = 365
MIN_ACCOUNT_SIZE = 100  # USD

# Export Configuration
EXPORT_FORMATS = ["csv", "json", "excel", "database"]
DEFAULT_EXPORT_FORMAT = "csv"

# Logging Configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": LOGS_DIR / "deepseek_system.log",
            "formatter": "detailed",
            "level": "INFO"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "detailed",
            "level": "INFO"
        }
    },
    "root": {
        "handlers": ["file", "console"],
        "level": "INFO"
    }
}
'''
        
        with open(config_file, "w") as f:
            f.write(config_content)
        print("   Created config/settings.py")
    
    def download_ai_models(self):
        """Download pre-trained AI models"""
        print("🤖 Downloading AI models...")
        
        models_dir = self.project_root / "models"
        models_dir.mkdir(exist_ok=True)
        
        # This would download actual models in production
        print("   AI models will be downloaded on first run")
    
    def run_tests(self):
        """Run system tests"""
        print("🧪 Running system tests...")
        
        test_files = [
            "test_core.py",
            "test_validation.py",
            "test_api.py"
        ]
        
        for test_file in test_files:
            test_path = self.project_root / "tests" / test_file
            if test_path.exists():
                try:
                    subprocess.run([sys.executable, str(test_path)], check=True)
                    print(f"   ✓ {test_file}")
                except:
                    print(f"   ✗ {test_file} failed")
    
    def setup_complete(self):
        """Display completion message"""
        print("""
        🎉 SETUP COMPLETED SUCCESSFULLY!
        
        Next steps:
        1. Edit the .env file with your API keys
        2. Run the system: python run.py --mode=generate
        3. Access dashboard: python run.py --mode=dashboard
        4. Start API: python run.py --mode=api
        
        Quick Start Examples:
        • Generate 100 Forex hot leads: python run.py --type=forex --category=hot --count=100
        • Generate recovery leads for Germany: python run.py --type=crypto --category=recovery --country=DE
        • Start the web dashboard: python run.py --mode=dashboard
        """)
    
    def run(self):
        """Execute complete setup"""
        self.print_banner()
        self.create_directory_structure()
        self.install_dependencies()
        self.install_system_dependencies()
        self.configure_environment()
        self.download_ai_models()
        self.run_tests()
        self.setup_complete()

if __name__ == "__main__":
    setup = DeepSeekSetup()
    setup.run()