import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
ROOT_DIR = Path(__file__).parent
CODE_DIR = ROOT_DIR / "code"
DATA_DIR = ROOT_DIR / "baseline_data"
OUTPUT_DIR = ROOT_DIR / "outputs"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# Model configurations
DEFAULT_MODEL = "gpt-4"
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Data processing settings
BATCH_SIZE = 32
NUM_WORKERS = 4

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FILE = OUTPUT_DIR / "experiment.log" 