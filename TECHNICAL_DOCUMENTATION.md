# TECHNICAL DOCUMENTATION
## AI-Powered Answer Quality Analyze System - Implementation Details

**Version**: 1.0 (Production Ready)  
**Date**: February 2026  
**Accuracy**: 94% on tuned data, 86% on unseen data

---

## TABLE OF CONTENTS
1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [Data Flow Pipeline](#3-data-flow-pipeline)
4. [Preprocessing Layer](#4-preprocessing-layer)
5. [Evaluation Metrics Engine](#5-evaluation-metrics-engine)
6. [Intelligent Enhancement Systems](#6-intelligent-enhancement-systems)
7. [Scoring Algorithm](#7-scoring-algorithm)
8. [API Implementation](#8-api-implementation)
9. [Performance Metrics](#9-performance-metrics)
10. [Code Structure](#10-code-structure)

---

## 1. ARCHITECTURE OVERVIEW

### 1.1 System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     WEB INTERFACE (Flask)                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │          Student/Teacher Input Forms                  │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              PREPROCESSING LAYER                             │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │   Text     │  │  Number    │  │ Tokenize & │           │
│  │ Cleaning   │→ │Normalize   │→ │ Lemmatize  │           │
│  └────────────┘  └────────────┘  └────────────┘           │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           EVALUATION METRICS ENGINE (9 Metrics)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  AI Semantic (80%)  │  TF-IDF (2%)  │  NB (2%)      │  │
│  │  Partial Match (5%) │  Relevance (8%) │ Others (3%) │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│          INTELLIGENT ENHANCEMENT LAYER                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐           │
│  │  Synonym   │  │  Antonym   │  │ Semantic   │           │
│  │   Boost    │  │  Penalty   │  │  Mismatch  │           │
│  └────────────┘  └────────────┘  └────────────┘           │
│  ┌────────────┐  ┌────────────┐                            │
│  │Containment │  │Contradiction│                           │
│  │   Check    │  │  Detection  │                           │
│  └────────────┘  └────────────┘                            │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SCORING ALGORITHM                               │
│         (Weighted Average + Boosts + Penalties)             │
└───────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              DATABASE STORAGE (MySQL)                        │
│            Score + Timestamp + Student Info                 │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Design Pattern
- **Pattern**: Layered Architecture with Pipeline Processing
- **Approach**: Hybrid AI + Rule-based System
- **Processing**: Sequential with conditional branching

---

## 2. TECHNOLOGY STACK

### 2.1 Core Technologies
```python
# Web Framework
Flask==3.1.2
Flask-MySQLdb==2.0.0

# AI/ML Libraries
sentence-transformers==2.2.2  # Transformer models
scikit-learn==1.3.0           # Classical ML algorithms
nltk==3.8.1                   # Natural Language Toolkit

# Data Processing
pandas==2.0.3
numpy==1.24.3

# Database
MySQL (via mysql-connector-python)
```

### 2.2 AI Model Specification
```python
Model: paraphrase-MiniLM-L6-v2
Architecture: Transformer (BERT-based)
Embedding Dimension: 384
Training Data: 1B+ sentence pairs
Purpose: Semantic similarity understanding
Load Time: ~2-3 seconds (once at startup)
Inference Time: ~100-200ms per answer pair
```

### 2.3 NLTK Resources
```python
- punkt: Tokenization
- wordnet: Synonym/Antonym detection
- vader_lexicon: Sentiment analysis
- omw-1.4: Open Multilingual WordNet
- stopwords: Common word filtering
```

---

## 3. DATA FLOW PIPELINE

### 3.1 Request Flow
```python
# Step 1: Input Reception
POST /evaluate_answer
├── question_id
├── expected_answer
└── student_answer

# Step 2: Preprocessing
expected_normalized = preprocess_pipeline(expected_answer)
student_normalized = preprocess_pipeline(student_answer)

# Step 3: Early Exit Checks
if student_answer.empty():
    return 0
if exact_match(expected, student):
    return 10
if numbers_match(expected, student):
    return 9

# Step 4: Metric Calculation (Parallel)
metrics = calculate_all_metrics(expected, student)

# Step 5: Weighted Aggregation
base_score = weighted_average(metrics, weights)

# Step 6: Enhancement Application
final_score = apply_boosts_and_penalties(base_score, expected, student)

# Step 7: Response
return round(final_score, 0)  # Integer 0-10
```

### 3.2 Processing Time Breakdown
```
Text Preprocessing:     20-30ms   (5%)
AI Model Inference:     100-200ms (50%)
Classical Metrics:      50-75ms   (20%)
Enhancement Checks:     50-75ms   (20%)
Score Calculation:      10-20ms   (5%)
──────────────────────────────────────
Total Average:          ~230-400ms
```

---

## 4. PREPROCESSING LAYER

### 4.1 Number Normalization System

#### 4.1.1 Number Word Mapping
```python
NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', ..., 'twenty': '20',
    'thirty': '30', ..., 'hundred': '100', 'thousand': '1000'
}

COMPOUND_NUMBERS = {
    'twenty-five': '25', 'twenty five': '25', 'twentyfive': '25',
    'forty-five': '45', ..., 'ninety-nine': '99'
}
```

#### 4.1.2 Algorithm: normalize_numbers()
```python
Input: "nineteen forty five"
Process:
  1. Replace compounds: "twenty-five" → "25"
  2. Detect decade + ones: "twenty" + "five" → "2" + "5" → "25"
  3. Detect year pattern: "nineteen" + "forty" + "five" → "19" + "4" + "5"
  4. Post-process: "19 4 5" → "1945" (regex: r'\b(19|18|20)\s+(\d{2})\b')
  5. Combine digits: "2 5" → "25" (regex: r'\b([2-9])\s+([0-9])\b')
Output: "1945"
```

#### 4.1.3 Algorithm: numbers_match()
```python
Purpose: Check if numerical values match across formats

Steps:
  1. Normalize both answers
  2. Extract all numbers using regex: r'\b\d+\.?\d*\b'
  3. Handle special symbols: °C → degrees celsius
  4. Compare number sets
  5. Check decimal tolerance: |float(a) - float(b)| < 0.01
  6. Check prefix matching: "3.14" matches "3.14159"

Returns:
  - True: Numbers match
  - False: Different numbers (wrong answer)
  - None: No numbers to compare
```

### 4.2 Text Cleaning

#### 4.2.1 Algorithm: clean_text()
```python
Purpose: Normalize text for consistent comparison

Operations:
  1. Lowercase conversion
  2. Concatenated word splitting:
     - Pattern: r'([a-z])([A-Z])' → r'\1 \2'
     - "electronicdevice" → "electronic device"
  3. Common word boundary detection:
     - (electronic)(device), (four)(legged), etc.
  4. Whitespace normalization: multiple spaces → single space
  5. Strip leading/trailing spaces

Example:
  Input:  "ElectronicDevice   for  computing"
  Output: "electronic device for computing"
```

### 4.3 Text Preprocessing

#### 4.3.1 Algorithm: preprocess_text()
```python
Purpose: Tokenize, lemmatize, and clean for analysis

Pipeline:
  1. clean_text(text)  # Normalize
  2. word_tokenize()   # NLTK tokenization
  3. Filter alphanumeric only
  4. WordNetLemmatizer.lemmatize()
     - "running", "runs", "ran" → "run"
     - "better", "good" → "good"

Returns: List of lemmatized tokens

Example:
  Input:  "The cats are running quickly"
  Output: ['cat', 'run', 'quickly']
```

#### 4.3.2 Stop Word Removal
```python
STOPWORDS = {'a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 
             'be', 'been', 'being', 'of', 'in', 'on', 'at', 'to', 
             'for', 'with', 'by'}

# Applied selectively in:
- partial_match(): Token comparison
- relevance_score(): Keyword extraction
- containment_check(): Content inclusion
```

---

## 5. EVALUATION METRICS ENGINE

### 5.1 Metric Implementation Details

#### 5.1.1 Exact Match (Weight: 1%)
```python
def exact_match(expected_answer, student_answer):
    expected_normalized = clean_text(expected_answer)
    student_normalized = clean_text(student_answer)
    return int(expected_normalized == student_normalized)

Returns: 1 (match) or 0 (no match)
```

#### 5.1.2 Partial Match with Fuzzy Matching (Weight: 5%)
```python
def fuzzy_token_match(token1, token2, threshold=0.85):
    # Uses difflib.SequenceMatcher
    ratio = SequenceMatcher(None, token1, token2).ratio()
    return ratio >= threshold

def partial_match(expected_answer, student_answer):
    1. Preprocess both answers (tokenize, lemmatize)
    2. Remove stopwords from both
    3. For each expected token:
         Find fuzzy match in student tokens (85% threshold)
         Count matches
    4. Return: matched_count / total_expected_tokens

Example:
  Expected: "Pacific Ocean"  → ['pacific', 'ocean']
  Student:  "Pacifc Ocean"   → ['pacifc', 'ocean']
  Result: 2/2 = 1.0 (fuzzy match handles "Pacifc")
```

#### 5.1.3 TF-IDF Cosine Similarity (Weight: 2%)
```python
def cosine_similarity_score(expected_answer, student_answer):
    1. Create TfidfVectorizer with:
         - Custom tokenizer: preprocess_text()
         - Lowercase: True
         - Stop words: 'english'
    2. Fit and transform both answers
    3. Calculate cosine similarity:
         similarity = (A · B) / (||A|| × ||B||)
    4. Return: similarity score (0-1)

TF-IDF Formula:
  TF(t) = count(t) / total_words
  IDF(t) = log(total_docs / docs_containing(t))
  TF-IDF(t) = TF(t) × IDF(t)
```

#### 5.1.4 Sentence Transformer AI - Enhanced Match (Weight: 40%)
```python
def enhanced_sentence_match(expected_answer, student_answer):
    1. Clean and normalize both texts
    2. Generate embeddings using SENTENCE_MODEL:
         embeddings = model.encode([text])
         # Returns 384-dimensional vector
    3. Calculate cosine similarity in embedding space
    4. Return: similarity (0-1)

Technical Details:
  Model: sentence-transformers/paraphrase-MiniLM-L6-v2
  - 6-layer MiniLM (lightweight BERT variant)
  - Trained on paraphrase detection
  - Semantic understanding without keyword matching
  
Example:
  Expected: "Ability to learn"
  Student:  "Capacity to understand"
  Keyword Match: Low (~20%)
  Semantic Match: High (~75%)
```

#### 5.1.5 Semantic Similarity (Weight: 40%)
```python
def semantic_similarity_score(expected_answer, student_answer):
    # Same implementation as enhanced_sentence_match()
    # Redundant check for reliability
    # Uses same SENTENCE_MODEL with same preprocessing
    
    1. clean_text() on both
    2. model.encode() for embeddings
    3. cosine_similarity() between embeddings
    4. Return: similarity (0-1)

Combined AI Weight: 40% + 40% = 80%
```

#### 5.1.6 Multinomial Naive Bayes (Weight: 2%)
```python
def multinomial_naive_bayes_score(expected_answer, student_answer):
    1. Create CountVectorizer with:
         - Custom tokenizer
         - Lowercase
         - Stop words removed
    2. Transform both answers to count vectors
    3. Train MultinomialNB classifier:
         X = [expected_vector, student_vector]
         y = [0, 1]  # Labels
    4. Predict probability of student answer
    5. Return: probability score (0-1)

Bayes Theorem:
  P(class|features) = P(features|class) × P(class) / P(features)
```

#### 5.1.7 Coherence Score (Weight: 2%)
```python
def coherence_score(expected_answer, student_answer):
    len_expected = len(word_tokenize(clean_text(expected_answer)))
    len_student = len(word_tokenize(clean_text(student_answer)))
    
    if len_student >= len_expected:
        return 0.9  # Don't penalize detailed answers
    else:
        return len_student / len_expected

Rationale: Longer, detailed answers should not be penalized
```

#### 5.1.8 Relevance Score (Weight: 8%)
```python
def relevance_score(expected_answer, student_answer):
    1. Tokenize both answers
    2. Remove stopwords
    3. For each expected token:
         Check fuzzy match in student tokens (85% threshold)
         Count matches
    4. Return: matched / total_expected

Similar to partial_match but without lemmatization
```

#### 5.1.9 Sentiment Analysis (Weight: 0%)
```python
def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()  # VADER
    sentiment_score = sia.polarity_scores(text)['compound']
    return (sentiment_score + 1) / 2  # Normalize to [0, 1]

Status: Implemented but not used (weight=0)
Reason: Sentiment not relevant for factual correctness
```

### 5.2 Weighted Average Calculation
```python
def weighted_average_score(scores, weights):
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

Weight Distribution:
  AI Semantic:      80% (Enhanced 40% + Semantic 40%)
  Relevance:        8%
  Partial Match:    5%
  TF-IDF:           2%
  Naive Bayes:      2%
  Coherence:        2%
  Exact Match:      1%
  Sentiment:        0%
  ────────────────────
  Total:            100%
```

---

## 6. INTELLIGENT ENHANCEMENT SYSTEMS

### 6.1 Synonym Detection System

#### 6.1.1 Manual Synonym Dictionary
```python
SYNONYM_PAIRS = {
    # Building/Structure (6 pairs)
    ('house', 'building'), ('house', 'structure'), ('house', 'home'),
    ('house', 'residence'), ('building', 'structure'), ('residence', 'home'),
    
    # Vehicle (3 pairs)
    ('car', 'vehicle'), ('car', 'automobile'), ('vehicle', 'automobile'),
    
    # Intelligence (11 pairs)
    ('smart', 'intelligent'), ('intelligence', 'ability'), 
    ('capacity', 'ability'), ('understand', 'comprehend'),
    ('reason', 'think'), ('learn', 'understand'), etc.
    
    # Temperature (3 pairs)
    ('warm', 'hot'), ('temperature', 'hot'), ('high', 'temperature'),
    
    # Living/People (4 pairs)
    ('live', 'reside'), ('people', 'human'), ('person', 'human'),
    
    # Government (2 pairs)
    ('government', 'system'), ('govern', 'rule'),
    
    # Travel (1 pair)
    ('travel', 'transportation')
}

Total: 40+ bidirectional pairs
```

#### 6.1.2 Algorithm: check_synonyms()
```python
Purpose: Detect if two words are synonyms

Method 1: Manual Dictionary
  - Fast lookup in predefined pairs
  - Domain-specific synonyms

Method 2: WordNet
  - synsets = wordnet.synsets(word)
  - Check if words share any synset
  - Check lemma overlap between synsets

Returns: Boolean (True if synonyms)

Example:
  check_synonyms('intelligent', 'smart') → True
  check_synonyms('house', 'building')    → True
```

#### 6.1.3 Algorithm: synonym_boost()
```python
Purpose: Calculate boost score for synonym usage

Steps:
  1. Preprocess both answers (tokens > 2 chars)
  2. For each expected token:
       Find synonym in student tokens
       Count matches
  3. Calculate ratio: synonym_count / expected_tokens
  4. Return: boost ratio (0-1)

Applied in evaluate():
  if synonym_boost > 0.2:  # 20%+ synonyms
      if synonym_boost >= 0.5:  # Strong usage
          final_score = max(final_score, 7)
          boost = synonym_boost * 0.5  # Up to 50% boost
      else:
          boost = synonym_boost * 0.4  # Up to 40% boost
      final_score = min(10, final_score * (1 + boost))
```

### 6.2 Antonym Detection System

#### 6.2.1 Manual Antonym Dictionary
```python
ANTONYM_PAIRS = {
    # Temperature (3 pairs)
    ('hot', 'cold'), ('warm', 'cold'), ('cool', 'hot'),
    
    # Size (4 pairs)
    ('big', 'small'), ('large', 'small'), ('large', 'tiny'),
    
    # Speed (3 pairs)
    ('fast', 'slow'), ('quick', 'slow'),
    
    # Direction (3 pairs)
    ('up', 'down'), ('high', 'low'), ('rise', 'fall'),
    
    # Light/Time (3 pairs)
    ('light', 'dark'), ('day', 'night'), ('sunlight', 'darkness'),
    
    # Morality (4 pairs)
    ('good', 'bad'), ('right', 'wrong'), ('truth', 'lie'),
    ('truth', 'lies'), ('truth', 'deception'), ('facts', 'lies'),
    
    # Emotion (6 pairs)
    ('love', 'hate'), ('love', 'hatred'), ('affection', 'hatred'),
    ('happy', 'sad'), ('happiness', 'sadness'), ('joy', 'sorrow'),
    
    # Success/Failure (3 pairs)
    ('success', 'failure'), ('achieve', 'fail'), ('win', 'lose'),
    
    # Courage/Fear (4 pairs)
    ('brave', 'cowardly'), ('courageous', 'fearful'), 
    ('brave', 'fearful'), ('courage', 'fear'),
    
    # Health (4 pairs)
    ('health', 'sickness'), ('health', 'disease'), 
    ('healthy', 'sick'), ('well', 'ill'),
    
    # Peace/War (7 pairs)
    ('peace', 'war'), ('peace', 'violence'), ('peaceful', 'violent'),
    ('calm', 'violent'), ('absence', 'war'), ('absence', 'violence'),
    ('conflict', 'peace'), ('war', 'calm'),
    
    # Wealth/Poverty (5 pairs)
    ('wealth', 'poverty'), ('rich', 'poor'), ('wealthy', 'poor'),
    ('money', 'nothing'), ('money', 'poverty'),
    
    # Knowledge/Ignorance (5 pairs)
    ('knowledge', 'ignorance'), ('knowing', 'ignorant'),
    ('wise', 'ignorant'), ('understanding', 'ignorance'),
    
    # Safety/Danger (6 pairs)
    ('safe', 'dangerous'), ('safe', 'risky'), ('safety', 'danger'),
    ('secure', 'risky'), ('free', 'dangerous'),
    
    # Clean/Dirty (4 pairs)
    ('clean', 'dirty'), ('clean', 'filthy'), ('pure', 'impure'),
    
    # Life/Death (3 pairs)
    ('life', 'death'), ('alive', 'dead'), ('living', 'dead'),
    
    # Creation/Destruction (3 pairs)
    ('create', 'destroy'), ('build', 'demolish'), ('make', 'break'),
    
    # Physical (2 pairs)
    ('gills', 'lungs'), ('lungs', 'gills'),
    
    # Fire/Ice (4 pairs)
    ('frozen', 'fire'), ('ice', 'fire'), ('cold', 'fire'), ('water', 'fire')
}

Total: 50+ bidirectional pairs
```

#### 6.2.2 Algorithm: check_antonyms()
```python
Purpose: Detect if two words are antonyms

Method 1: Manual Dictionary
  - Comprehensive bidirectional pairs
  - Domain-specific opposites

Method 2: WordNet
  - For each synset of word1:
      For each lemma:
          Check if lemma.antonyms() contains word2

Returns: Boolean (True if antonyms)

Example:
  check_antonyms('hot', 'cold')    → True
  check_antonyms('peace', 'war')   → True
  check_antonyms('ice', 'fire')    → True
```

#### 6.2.3 Algorithm: antonym_penalty()
```python
Purpose: Calculate penalty for using opposite meanings

Steps:
  1. Extract tokens from both answers (including short words)
  2. Also split original text (catch 'war', 'sad', etc.)
  3. Check all token combinations for antonyms
  4. Count antonym matches
  5. Calculate penalty ratio
  6. If any antonyms: penalty × 2.0 (doubled strength)

Applied in evaluate():
  if antonym_penalty_value > 0:
      multiplier = max(0.1, 1 - (penalty * 1.8))  # Up to 90% reduction
      final_score = final_score * multiplier

Example:
  Expected: "hot"
  Student:  "cold"
  Penalty: 1.0 (100%)
  Multiplier: 0.1 (90% reduction)
  If base_score = 5, final = 5 × 0.1 = 0.5 → rounds to 1
```

### 6.3 Semantic Mismatch Detection

#### 6.3.1 Algorithm: check_semantic_mismatch()
```python
Purpose: Detect factually wrong answers in same category

Categories:
  1. Numbers (via numbers_match())
  2. Animals (mammals, birds, fish, reptiles, insects)
  3. Cities (50+ major cities)
  4. Countries (20+ countries)
  5. People (presidents, authors, artists, scientists, explorers)
  6. Languages (14+ languages)
  7. Planets (9 planets including Pluto)
  8. Colors (basic colors, length ≤ 3 words)

Logic:
  - If both have category members but DIFFERENT → Mismatch
  
Returns: Boolean (True if wrong category member detected)

Examples:
  Expected: "Tokyo"     Student: "Beijing"    → True (cities)
  Expected: "Lincoln"   Student: "Washington" → True (presidents)
  Expected: "cat"       Student: "bird"       → True (animals)
  Expected: "25"        Student: "30"         → True (numbers)
```

#### 6.3.2 Category Dictionaries
```python
CITIES = {'tokyo', 'paris', 'london', 'berlin', 'rome', 'madrid', 
          'beijing', 'moscow', 'washington', 'munich', 'vienna', etc.}

COUNTRIES = {'russia', 'china', 'usa', 'america', 'japan', 'india', 
             'brazil', 'germany', 'france', 'italy', etc.}

US_PRESIDENTS = {'washington', 'lincoln', 'jefferson', 'roosevelt', 
                 'kennedy', 'obama', 'trump', 'biden', etc.}

AUTHORS = {'shakespeare', 'dickens', 'austen', 'orwell', 'hemingway', etc.}

ARTISTS = {'picasso', 'vinci', 'leonardo', 'michelangelo', 'monet', etc.}

SCIENTISTS = {'einstein', 'newton', 'darwin', 'curie', 'galileo', etc.}

LANGUAGES = {'english', 'japanese', 'chinese', 'spanish', 'french', etc.}

PLANETS = {'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', etc.}

COLORS = {'red', 'blue', 'green', 'yellow', 'orange', 'purple', etc.}
```

### 6.4 Contradiction Detection

#### 6.4.1 Algorithm: check_contradiction()
```python
Purpose: Detect logical impossibilities

Contradiction Pairs:
  - ('frozen', 'fire'), ('ice', 'fire'), ('cold', 'fire')
  - ('gills', 'lungs'), ('lungs', 'gills')
  - ('green', 'purple'), ('red', 'blue'), ('black', 'white')
  - ('opposite', 'same')

Logic:
  If word1 in expected AND word2 in student → Contradiction

Returns: Boolean (True if contradiction found)

Applied in evaluate():
  if check_contradiction(expected, response):
      final_score = min(final_score, 1)  # Cap at 1/10
```

### 6.5 Containment Check

#### 6.5.1 Algorithm: containment_check()
```python
Purpose: Check if expected answer is contained in student answer

Method 1: Direct String Containment
  if clean_text(expected) in clean_text(student):
      return 1.0

Method 2: Token-based Containment
  1. Tokenize and preprocess both
  2. Remove stopwords
  3. For each expected token:
       Find fuzzy match in student tokens (85% threshold)
       Count contained tokens
  4. Return: contained_count / total_expected

Applied in evaluate():
  if containment >= 0.8:  # 80%+ contained
      final_score = max(final_score, 8)
  if containment == 1.0:  # 100% contained
      final_score = max(final_score, 9)

Example:
  Expected: "A planet"
  Student:  "Earth is the third planet from the sun"
  Containment: 1.0 (100%)
  Boost: Minimum 9/10
```

---

## 7. SCORING ALGORITHM

### 7.1 Complete Evaluation Flow
```python
def evaluate(expected, response, debug=False):
    # PHASE 1: EARLY EXITS
    expected_normalized = clean_text(expected)
    response_normalized = clean_text(response)
    
    if expected_normalized == response_normalized:
        return 10  # Perfect match
    
    if not response or not response.strip():
        return 0  # Empty answer
    
    num_match = numbers_match(expected, response)
    if num_match is True:
        return 9  # Numbers match exactly
    
    # PHASE 2: METRIC CALCULATION
    metrics = [
        exact_match(expected, response),           # 1%
        partial_match(expected, response),          # 5%
        cosine_similarity_score(expected, response),# 2%
        sentiment_analysis(response),               # 0%
        enhanced_sentence_match(expected, response),# 40%
        multinomial_naive_bayes_score(...),         # 2%
        semantic_similarity_score(expected, response),# 40%
        coherence_score(expected, response),        # 2%
        relevance_score(expected, response)         # 8%
    ]
    
    weights = [0.01, 0.05, 0.02, 0.00, 0.40, 0.02, 0.40, 0.02, 0.08]
    
    scaled_scores = [score * 10 for score in metrics]
    final_score = weighted_average(scaled_scores, weights)
    
    # PHASE 3: CALCULATE ENHANCEMENT VALUES
    avg_semantic = (metrics[4] + metrics[6]) / 2
    synonym_value = synonym_boost(expected, response)
    antonym_value = antonym_penalty(expected, response)
    containment = containment_check(expected, response)
    mismatch = check_semantic_mismatch(expected, response)
    contradiction = check_contradiction(expected, response)
    
    # PHASE 4: APPLY BOOSTS (Sequential Max)
    # Boost 0: High Semantic Similarity
    if avg_semantic >= 0.7:
        final_score = max(final_score, 9)
    elif avg_semantic >= 0.55:
        final_score = max(final_score, 8)
    elif avg_semantic >= 0.45:
        final_score = max(final_score, 7)
    
    # Boost 1: Containment
    if containment >= 0.8:
        final_score = max(final_score, 8)
        if containment == 1.0:
            final_score = max(final_score, 9)
    
    # Boost 2: Synonyms
    if synonym_value > 0.2:
        if synonym_value >= 0.5:
            final_score = max(final_score, 7)
            boost = synonym_value * 0.5
        else:
            boost = synonym_value * 0.4
        final_score = min(10, final_score * (1 + boost))
    
    # PHASE 5: APPLY PENALTIES (Sequential Min)
    # Penalty 1: Antonyms
    if antonym_value > 0:
        multiplier = max(0.1, 1 - (antonym_value * 1.8))
        final_score = final_score * multiplier
    
    # Penalty 2: Low Semantic Similarity
    if avg_semantic < 0.30:
        final_score = min(final_score, 3)
    if avg_semantic < 0.20:
        final_score = min(final_score, 1)
    
    # Penalty 3: Semantic Mismatch
    if mismatch:
        final_score = min(final_score, 2)
    
    # Penalty 4: Contradiction
    if contradiction:
        final_score = min(final_score, 1)
    
    # PHASE 6: FINALIZATION
    rounded_score = round(final_score)
    return max(0, min(10, rounded_score))
```

### 7.2 Boost Priority System
```
Priority 1: AI Semantic Similarity (Gates minimum score)
  ├── 70%+ similarity → minimum 9/10
  ├── 55%+ similarity → minimum 8/10
  └── 45%+ similarity → minimum 7/10

Priority 2: Containment (Rewards detail)
  ├── 100% contained → minimum 9/10
  └── 80%+ contained → minimum 8/10

Priority 3: Synonym Usage (Vocabulary variation)
  ├── 50%+ synonyms → minimum 7/10 + 50% boost
  └── 20%+ synonyms → 40% boost

Note: max() operation ensures highest boost wins
```

### 7.3 Penalty Priority System
```
Priority 1: Antonym (Opposite meaning)
  └── Any antonym → Up to 90% score reduction

Priority 2: Low AI Semantic (Wrong answer)
  ├── <20% similarity → cap at 1/10
  └── <30% similarity → cap at 3/10

Priority 3: Semantic Mismatch (Wrong category)
  └── Different city/name/number → cap at 2/10

Priority 4: Contradiction (Logical impossibility)
  └── ice=fire, gills=lungs → cap at 1/10

Note: min() operation ensures lowest cap wins (strictest penalty)
```

---

## 8. API IMPLEMENTATION

### 8.1 Flask Routes

#### 8.1.1 Student Test Taking
```python
@app.route('/student/take_test/<int:test_id>', methods=['GET', 'POST'])
def student_take_test(test_id):
    if 'student_logged_in' not in session:
        return redirect(url_for('student_login'))
    
    if request.method == 'POST':
        # Get all answers from form
        for question_id in request.form:
            student_answer = request.form[question_id]
            
            # Fetch expected answer from database
            cur.execute("SELECT answer FROM questions WHERE id = %s", 
                       (question_id,))
            expected_answer = cur.fetchone()[0]
            
            # EVALUATE ANSWER
            score = evaluate(expected_answer, student_answer)
            
            # Store score in database
            cur.execute("""
                INSERT INTO scores (student_id, test_id, question_id, 
                                   student_answer, score, timestamp)
                VALUES (%s, %s, %s, %s, %s, NOW())
            """, (session['student_id'], test_id, question_id, 
                  student_answer, score))
        
        mysql.connection.commit()
        return redirect(url_for('student_view_score', test_id=test_id))
```

#### 8.1.2 Teacher View Scores
```python
@app.route('/teacher/view_scores/<int:test_id>')
def teacher_view_scores(test_id):
    cur.execute("""
        SELECT s.student_name, q.question, sc.student_answer, 
               q.answer, sc.score
        FROM scores sc
        JOIN students s ON sc.student_id = s.id
        JOIN questions q ON sc.question_id = q.id
        WHERE sc.test_id = %s
        ORDER BY s.student_name, q.id
    """, (test_id,))
    
    results = cur.fetchall()
    return render_template('teacher_view_score.html', scores=results)
```

### 8.2 Database Schema

#### 8.2.1 Key Tables
```sql
-- Students
CREATE TABLE students (
    id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) UNIQUE,
    password VARCHAR(255),
    student_name VARCHAR(100),
    email VARCHAR(100)
);

-- Tests
CREATE TABLE tests (
    id INT PRIMARY KEY AUTO_INCREMENT,
    test_name VARCHAR(100),
    subject VARCHAR(50),
    teacher_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Questions
CREATE TABLE questions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    test_id INT,
    question TEXT,
    answer TEXT,  -- Expected answer
    marks INT DEFAULT 10
);

-- Scores (Evaluation Results)
CREATE TABLE scores (
    id INT PRIMARY KEY AUTO_INCREMENT,
    student_id INT,
    test_id INT,
    question_id INT,
    student_answer TEXT,
    score INT,  -- 0-10 from evaluate()
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (student_id) REFERENCES students(id),
    FOREIGN KEY (test_id) REFERENCES tests(id),
    FOREIGN KEY (question_id) REFERENCES questions(id)
);
```

---

## 9. PERFORMANCE METRICS

### 9.1 Test Results

#### 9.1.1 Original Test Suite (50 Questions)
```
Categories: 10 categories
- Exact matches
- Paraphrases
- Partial answers
- Typos
- Word order changes
- Wrong answers
- Extra information
- Concatenated words
- Synonyms
- Opposites

Results:
  Perfect (0 diff):      15 cases (30%)
  Good (1 diff):         24 cases (48%)
  Acceptable (2 diff):   8 cases (16%)
  Needs Work (3+ diff):  3 cases (6%)
  
  Average Error: 1.08 points
  Accuracy: 94% (within ±2 points)
```

#### 9.1.2 Extended Test Suite (116 Unseen Questions)
```
Domains: 12 categories
- Science (11 questions)
- History (10 questions)
- Geography (10 questions)
- Math (10 questions)
- Literature (10 questions)
- Technology (10 questions)
- General Knowledge (10 questions)
- Wrong Answers (10 questions)
- Paraphrases (10 questions)
- Partial Answers (10 questions)
- Opposites (10 questions)
- Detailed Answers (5 questions)

Results:
  Perfect (0 diff):      33 cases (28.4%)
  Good (1 diff):         58 cases (50.0%)
  Acceptable (2 diff):   9 cases (7.8%)
  Needs Work (3+ diff):  16 cases (13.8%)
  
  Average Error: 1.31 points
  Accuracy: 86.2% (within ±2 points)
```

### 9.2 Performance Benchmarks

#### 9.2.1 Response Time
```
Model Loading:        2-3 seconds (once at startup)
Single Evaluation:    230-400ms average
  - Preprocessing:    20-30ms
  - AI Inference:     100-200ms
  - Metrics:          50-75ms
  - Enhancements:     50-75ms
  - Scoring:          10-20ms

Throughput: ~2.5-4 evaluations per second
```

#### 9.2.2 Resource Usage
```
Memory:
  - Base Flask App:   ~50 MB
  - AI Model Loaded:  ~250 MB
  - Total Runtime:    ~300-350 MB

CPU:
  - Idle:             <1%
  - Per Evaluation:   15-25% (single core)
  - AI Inference:     Up to 80% (during encoding)

Disk:
  - Model File:       90 MB (cached after first download)
  - NLTK Data:        ~50 MB
```

### 9.3 Accuracy Breakdown by Category

```
Category                  Accuracy  Avg Error
──────────────────────────────────────────────
Exact Matches             100%      0.00
Paraphrases               85%       1.40
Synonyms                  95%       0.80
Typos                     90%       1.00
Word Order                100%      0.20
Detailed Answers          100%      0.80
Wrong Answers (detect)    88%       1.50
Opposites (detect)        85%       1.80
Number Conversions        92%       1.20
Partial Answers           75%       2.10
──────────────────────────────────────────────
Overall                   86.2%     1.31
```

---

## 10. CODE STRUCTURE

### 10.1 File Organization
```
mass/
├── admin.py                    # Main Flask application (1475 lines)
│   ├── Imports & Configuration (lines 1-60)
│   ├── Number Processing (lines 68-205)
│   ├── Text Cleaning (lines 207-244)
│   ├── Core Metrics (lines 246-356)
│   ├── Enhancement Systems (lines 358-712)
│   ├── Main Evaluate Function (lines 714-828)
│   └── Flask Routes (lines 830-1475)
│
├── requirements.txt            # Python dependencies
├── create_tables.py            # Database initialization
├── test_evaluation.py          # Original test suite (50 cases)
├── test_evaluation_extended.py # Extended test suite (116 cases)
│
├── templates/                  # HTML templates
│   ├── Homepage.html
│   ├── admin*.html
│   ├── teacher*.html
│   └── student*.html
│
├── static/                     # CSS, JS, Images
│   ├── style.css
│   ├── animate.css
│   └── js/
│
└── docs/                       # Documentation
    ├── METHODOLOGY.md
    ├── DEMO_QUESTIONS.md
    ├── PRODUCTION_GUIDE.md
    └── IMPROVEMENTS_SUMMARY.md
```

### 10.2 Function Dependencies
```
evaluate()
├── clean_text()
├── numbers_match()
│   └── normalize_numbers()
├── preprocess_text()
│   ├── clean_text()
│   └── word_tokenize()
├── exact_match()
├── partial_match()
│   ├── preprocess_text()
│   └── fuzzy_token_match()
├── cosine_similarity_score()
│   └── preprocess_text()
├── enhanced_sentence_match()
│   └── SENTENCE_MODEL.encode()
├── semantic_similarity_score()
│   └── SENTENCE_MODEL.encode()
├── multinomial_naive_bayes_score()
├── coherence_score()
├── relevance_score()
│   └── fuzzy_token_match()
├── synonym_boost()
│   ├── preprocess_text()
│   └── check_synonyms()
├── antonym_penalty()
│   ├── preprocess_text()
│   └── check_antonyms()
├── containment_check()
│   ├── clean_text()
│   ├── preprocess_text()
│   └── fuzzy_token_match()
├── check_semantic_mismatch()
│   └── numbers_match()
├── check_contradiction()
└── weighted_average_score()
```

### 10.3 Key Configuration Constants
```python
# AI Model
SENTENCE_MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Thresholds
FUZZY_MATCH_THRESHOLD = 0.85  # 85% similarity for typos
SYNONYM_BOOST_THRESHOLD = 0.2  # 20% for boost activation
STRONG_SYNONYM_THRESHOLD = 0.5  # 50% for strong boost
CONTAINMENT_THRESHOLD = 0.8    # 80% for containment boost

# Semantic Similarity Gates
HIGH_SEMANTIC = 0.7   # 70%+ → minimum 9/10
GOOD_SEMANTIC = 0.55  # 55%+ → minimum 8/10
MODERATE_SEMANTIC = 0.45  # 45%+ → minimum 7/10
LOW_SEMANTIC = 0.30   # <30% → cap at 3/10
VERY_LOW_SEMANTIC = 0.20  # <20% → cap at 1/10

# Penalty Multipliers
ANTONYM_PENALTY_FACTOR = 1.8  # 90% max reduction
ANTONYM_STRENGTH_MULTIPLIER = 2.0  # Double penalty strength
SYNONYM_BOOST_WEAK = 0.4  # 40% boost
SYNONYM_BOOST_STRONG = 0.5  # 50% boost

# Score Caps
SEMANTIC_MISMATCH_CAP = 2  # Wrong category
CONTRADICTION_CAP = 1  # Logical impossibility
```

---

## APPENDIX A: Example Evaluations

### Example 1: Perfect Paraphrase
```
Question: "What is intelligence?"
Expected: "Ability to learn and think"
Student:  "Capacity to understand and reason"

Metrics:
  Exact Match: 0.0
  Partial Match: 0.0 (no word overlap)
  TF-IDF: 0.15 (minimal keyword match)
  Enhanced Match: 0.76 (HIGH AI semantic)
  Semantic Sim: 0.78 (HIGH AI semantic)
  Coherence: 0.9
  Relevance: 0.0

Avg Semantic: 0.77
Base Score: 7.72

Boosts:
  - High Semantic (≥0.7): min 9/10 → 9.0
  - Synonym Boost: 0.75 (75% synonyms)
    → Strong boost: 9 × 1.375 = 12.375 → capped at 10

Final Score: 10/10
```

### Example 2: Opposite Answer
```
Question: "What is hot?"
Expected: "High temperature"
Student:  "Cold and freezing"

Metrics:
  Exact Match: 0.0
  Enhanced Match: 0.22 (low semantic)
  Semantic Sim: 0.18 (very low)
  
Avg Semantic: 0.20
Base Score: 1.95

Antonym Check:
  - 'hot' ↔ 'cold': TRUE
  - 'high' ↔ 'freezing': TRUE
  - Antonym Ratio: 2/2 = 1.0
  - Penalty Multiplier: max(0.1, 1 - 1.0×1.8×2) = 0.1

Penalties:
  - Antonym: 1.95 × 0.1 = 0.195
  - Very Low Semantic (<0.20): cap at 1

Final Score: 1/10
```

### Example 3: Number Conversion
```
Question: "When did WW2 end?"
Expected: "1945"
Student:  "Nineteen forty five"

Preprocessing:
  normalize_numbers("nineteen forty five")
  → "19 4 5"
  → "1945" (post-processing regex)

Number Match:
  Expected: {1945}
  Student:  {1945}
  → TRUE (exact match)

Early Exit: Return 9/10
(No need to run full evaluation)

Final Score: 9/10
```

---

## APPENDIX B: Troubleshooting

### Common Issues

#### Issue 1: Model Download Fails
```
Error: Unable to download paraphrase-MiniLM-L6-v2
Solution:
  1. Check internet connection
  2. Run: pip install --upgrade sentence-transformers
  3. Pre-download: python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"
```

#### Issue 2: NLTK Data Missing
```
Error: Resource 'wordnet' not found
Solution:
  import nltk
  nltk.download('wordnet')
  nltk.download('punkt')
  nltk.download('vader_lexicon')
  nltk.download('omw-1.4')
```

#### Issue 3: MySQL Connection Error
```
Error: Access denied for user 'root'@'localhost'
Solution:
  1. Check MySQL service is running
  2. Update password in admin.py line 30
  3. Grant privileges: GRANT ALL ON answer_evaluation.* TO 'root'@'localhost';
```

---

## CONCLUSION

This technical documentation provides a complete view of the AI-powered answer quality analyze system's implementation. The system combines state-of-the-art transformer models (80% weight) with classical NLP techniques and intelligent rule-based enhancements to achieve **86% accuracy on unseen questions** while maintaining fast response times (~300ms average) and production-ready reliability.

**Key Technical Achievements**:
- Hybrid AI + Rule-based approach
- 9-metric weighted evaluation engine
- Intelligent boost/penalty system (synonyms, antonyms, contradictions)
- Number normalization with word↔digit conversion
- Robust preprocessing pipeline
- High accuracy with low computational cost

**Production Status**: ✅ Ready for deployment in educational institutions.

---

**Last Updated**: February 2026  
**Version**: 1.0  
**Project Team**: Adarsh S, Freddy Das Felix, Durga V S, Meenakshi S
