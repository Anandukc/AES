#!/usr/bin/env python3
"""
Automated NLP Setup Script for Answer Evaluation System
This script downloads all required NLTK data and models
"""

import sys

def download_nltk_data():
    """Download all required NLTK data packages"""
    print("\n" + "="*60)
    print("Downloading NLTK Data Packages...")
    print("="*60)
    
    try:
        import nltk
        
        packages = [
            ('stopwords', 'English stopwords list'),
            ('punkt', 'Sentence/word tokenizer'),
            ('wordnet', 'WordNet lemmatizer'),
            ('vader_lexicon', 'Sentiment analysis lexicon'),
            ('omw-1.4', 'Open Multilingual Wordnet (optional)')
        ]
        
        for package, description in packages:
            print(f"\nDownloading {package}: {description}")
            try:
                nltk.download(package, quiet=False)
                print(f"✓ {package} downloaded successfully")
            except Exception as e:
                print(f"✗ Error downloading {package}: {e}")
        
        print("\n✓ NLTK data download complete!")
        return True
        
    except ImportError:
        print("✗ NLTK not installed. Run: pip install nltk")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def download_sentence_transformer_model():
    """Download the sentence transformer model"""
    print("\n" + "="*60)
    print("Downloading Sentence Transformer Model...")
    print("="*60)
    print("Model: paraphrase-MiniLM-L6-v2")
    print("Size: ~90MB")
    print("This may take 1-2 minutes...\n")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("\n✓ Model downloaded successfully!")
        print(f"✓ Model cached at: ~/.cache/torch/sentence_transformers/")
        
        # Test the model
        test_text = "This is a test sentence"
        embedding = model.encode([test_text])
        print(f"✓ Model test passed (embedding dimension: {embedding.shape[1]})")
        
        return True
        
    except ImportError:
        print("✗ sentence-transformers not installed. Run: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def verify_installation():
    """Verify all components are working"""
    print("\n" + "="*60)
    print("Verifying Installation...")
    print("="*60)
    
    all_ok = True
    
    # Test NLTK
    try:
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        stops = stopwords.words('english')
        tokens = word_tokenize("Test")
        lemmatizer = WordNetLemmatizer()
        sia = SentimentIntensityAnalyzer()
        
        print("✓ NLTK components working")
    except Exception as e:
        print(f"✗ NLTK verification failed: {e}")
        all_ok = False
    
    # Test Sentence Transformers
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        print("✓ Sentence Transformers working")
    except Exception as e:
        print(f"✗ Sentence Transformers verification failed: {e}")
        all_ok = False
    
    # Test Scikit-learn
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        print("✓ Scikit-learn working")
    except Exception as e:
        print(f"✗ Scikit-learn verification failed: {e}")
        all_ok = False
    
    return all_ok

def main():
    print("="*60)
    print("NLP Setup Script for Answer Evaluation System")
    print("="*60)
    print("\nThis script will:")
    print("1. Download NLTK data packages (~50MB)")
    print("2. Download Sentence Transformer model (~90MB)")
    print("3. Verify installation")
    print("\nTotal download: ~140MB")
    print("Estimated time: 3-5 minutes")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Setup cancelled.")
        sys.exit(0)
    
    # Step 1: Download NLTK data
    nltk_ok = download_nltk_data()
    
    # Step 2: Download Sentence Transformer model
    model_ok = download_sentence_transformer_model()
    
    # Step 3: Verify installation
    verify_ok = verify_installation()
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    print(f"NLTK Data: {'✓ OK' if nltk_ok else '✗ FAILED'}")
    print(f"Sentence Transformer Model: {'✓ OK' if model_ok else '✗ FAILED'}")
    print(f"Verification: {'✓ OK' if verify_ok else '✗ FAILED'}")
    print("="*60)
    
    if nltk_ok and model_ok and verify_ok:
        print("\n✓ Setup completed successfully!")
        print("You can now run: python admin.py")
    else:
        print("\n✗ Setup incomplete. Please check errors above.")
        print("\nManual installation commands:")
        print("  pip install -r requirements.txt")
        print("  python -c \"import nltk; nltk.download('all')\"")
    
    print("\nFor detailed documentation, see: DOCUMENTATION.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
