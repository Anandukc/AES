# NLP Prerequisites Guide

## What You Need to Install

To use NLP (Natural Language Processing) in this Answer Evaluation System, you need to install **two types** of components:

### 1. Python Libraries (Installed via pip)
These are the code packages:
- **nltk** - Natural Language Toolkit
- **sentence-transformers** - For semantic text analysis
- **scikit-learn** - Machine learning algorithms
- **torch** (PyTorch) - Deep learning backend

### 2. Data Packages (Downloaded separately)
After installing NLTK, you must download data files:
- **stopwords** - List of common words to ignore
- **punkt** - For breaking text into sentences/words
- **wordnet** - Dictionary for word meanings
- **vader_lexicon** - For sentiment analysis

### 3. Pre-trained Models (Downloaded automatically)
- **paraphrase-MiniLM-L6-v2** - AI model for understanding text meaning

---

## Quick Installation (3 Methods)

### Method 1: Automated Setup (Recommended)
```bash
# Install libraries first
pip install -r requirements.txt

# Run automated setup
python setup_nlp.py
```

### Method 2: Manual Setup
```bash
# Step 1: Install libraries
pip install -r requirements.txt

# Step 2: Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

# Step 3: Download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"
```

### Method 3: Interactive (with GUI)
```python
import nltk
nltk.download()  # Opens a window to select packages
```

---

## Why Do We Need These?

### NLTK (Natural Language Toolkit)
```python
# What it does:
from nltk.tokenize import word_tokenize
word_tokenize("Python is great")  
# Output: ['Python', 'is', 'great']

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize("running")  
# Output: 'run' (base form)
```

### Sentence Transformers
```python
# What it does:
# Understands that these mean the same thing:
"Python is a programming language"
"Python is a language for programming"
# Both get similar "meaning scores"
```

### Why Download Separately?
- **Libraries** = The tools/programs
- **Data** = The dictionaries/word lists the tools need
- **Models** = Pre-trained AI brains that understand text

It's like:
- Installing Microsoft Word (library)
- But needing dictionaries for spell-check (data)
- And grammar rules for suggestions (models)

---

## Verification

### Test if everything is installed:
```bash
python test_nlp_setup.py
```

### Expected output:
```
==================================================
NLP Setup Verification
==================================================
✓ NLTK stopwords loaded
✓ NLTK tokenizer working
✓ NLTK lemmatizer working
✓ NLTK sentiment analyzer working
✓ Sentence Transformers working
  Model loaded: paraphrase-MiniLM-L6-v2
  Embedding dimension: 384
✓ Scikit-learn working

==================================================
Summary:
3/3 tests passed
✓ All NLP components ready!
==================================================
```

---

## Troubleshooting

### Error: "No module named 'nltk'"
```bash
pip install nltk
```

### Error: "Resource stopwords not found"
```python
import nltk
nltk.download('stopwords')
```

### Error: "Can't load model paraphrase-MiniLM-L6-v2"
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer('paraphrase-MiniLM-L6-v2')
# Wait 1-2 minutes for download
```

### Error: "CERTIFICATE_VERIFY_FAILED"
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# Then try downloading again
```

---

## Download Sizes & Times

| Component | Size | Time (approx) |
|-----------|------|---------------|
| NLTK library | 10 MB | 10-30 sec |
| NLTK data packages | 50 MB | 30-60 sec |
| PyTorch | 750 MB | 3-5 min |
| Sentence Transformers | 200 MB | 1-3 min |
| Pre-trained model | 90 MB | 1-2 min |
| **Total** | **~1.1 GB** | **6-12 min** |

*Times depend on internet speed*

---

## Where Files Are Stored

### NLTK Data:
- **Windows**: `C:\Users\YourName\AppData\Roaming\nltk_data`
- **Mac/Linux**: `~/nltk_data`

### Sentence Transformer Models:
- **All OS**: `~/.cache/torch/sentence_transformers/`

### Check locations:
```python
import nltk
print("NLTK data:", nltk.data.path)

import os
print("Model cache:", os.path.expanduser('~/.cache/torch/sentence_transformers/'))
```

---

## What Happens During Answer Evaluation?

When a student submits an answer, the system:

1. **Tokenizes** the answer (breaks into words) - uses NLTK
2. **Lemmatizes** words (converts to base form) - uses NLTK WordNet
3. **Removes stopwords** (ignores "the", "a", "is") - uses NLTK stopwords
4. **Calculates sentiment** (positive/negative tone) - uses NLTK VADER
5. **Generates embeddings** (converts text to numbers) - uses Sentence Transformers
6. **Compares semantically** (understands meaning) - uses cosine similarity from scikit-learn
7. **Scores the answer** (0-10) - weighted combination of all metrics

All these steps need the libraries and data to be installed!

---

## First Run Behavior

**First time running the app:**
- Takes 30-60 seconds to load models
- Downloads models if not cached
- Initializes NLTK components

**Subsequent runs:**
- Much faster (2-3 seconds)
- Models loaded from cache

---

## Need Help?

1. Run the test script: `python test_nlp_setup.py`
2. Check detailed docs: `DOCUMENTATION.md`
3. See error messages and solutions in troubleshooting section

---

## Minimum System Requirements

- **RAM**: 4GB (8GB recommended)
- **Disk Space**: 2GB free
- **Internet**: Required for initial download
- **Python**: 3.8 or higher

---

**Quick Start:**
```bash
pip install -r requirements.txt
python setup_nlp.py
python test_nlp_setup.py
python admin.py
```

That's it! 🚀
