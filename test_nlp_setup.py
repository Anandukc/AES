#!/usr/bin/env python3
"""Test if all NLP components are properly installed"""

def test_nltk():
    try:
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        # Test stopwords
        stops = stopwords.words('english')
        print("✓ NLTK stopwords loaded")
        
        # Test tokenization
        tokens = word_tokenize("Test sentence")
        print("✓ NLTK tokenizer working")
        
        # Test lemmatization
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("running")
        print("✓ NLTK lemmatizer working")
        
        # Test sentiment
        sia = SentimentIntensityAnalyzer()
        sia.polarity_scores("Good answer")
        print("✓ NLTK sentiment analyzer working")
        
        return True
    except Exception as e:
        print(f"✗ NLTK Error: {e}")
        return False

def test_sentence_transformers():
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode(["Test sentence"])
        print("✓ Sentence Transformers working")
        print(f"  Model loaded: paraphrase-MiniLM-L6-v2")
        print(f"  Embedding dimension: {embeddings.shape[1]}")
        return True
    except Exception as e:
        print(f"✗ Sentence Transformers Error: {e}")
        return False

def test_sklearn():
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.naive_bayes import MultinomialNB
        print("✓ Scikit-learn working")
        return True
    except Exception as e:
        print(f"✗ Scikit-learn Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("NLP Setup Verification")
    print("=" * 50)
    
    results = {
        "NLTK": test_nltk(),
        "Sentence Transformers": test_sentence_transformers(),
        "Scikit-learn": test_sklearn()
    }
    
    print("\n" + "=" * 50)
    print("Summary:")
    passed = sum(results.values())
    total = len(results)
    print(f"{passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All NLP components ready!")
    else:
        print("✗ Some components need attention")
        print("\nFix commands:")
        print("  pip install -r requirements.txt")
        print("  python -c \"import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon')\"")
    print("=" * 50)
