# src/extensions.py
from flask_mysqldb import MySQL
from sentence_transformers import SentenceTransformer
import warnings
import logging
import os

# Suppress warnings and reduce logging verbosity
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Suppress tokenizers warning

# MySQL extension (to be initialized with app later)
mysql = MySQL()

# Load Sentence Transformer model once at startup
print("="*60)
print("Loading AI model (paraphrase-MiniLM-L6-v2)...")
print("This only happens once at startup...")
SENTENCE_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("✓ Model loaded successfully! Ready for answer evaluation.")
print("="*60)