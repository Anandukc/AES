# src/evaluation.py
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
import re

from .extensions import SENTENCE_MODEL
from .utils import (
    clean_text, normalize_numbers, numbers_match, fuzzy_token_match,
    check_antonyms, check_synonyms, check_contradiction
)

# Download required NLTK data (only needed once)
for resource in ['punkt', 'wordnet', 'vader_lexicon', 'omw-1.4', 'stopwords']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

EN_STOPWORDS = set(stopwords.words("english"))

def preprocess_text(text):
    """Tokenize, lemmatize, and clean text"""
    if not text:
        return []
    text = clean_text(text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token.isalnum()]
    return lemmatized_tokens

def exact_match(expected_answer, student_answer):
    expected_normalized = clean_text(expected_answer)
    student_normalized = clean_text(student_answer)
    return int(expected_normalized == student_normalized)

def partial_match(expected_answer, student_answer):
    expected_tokens = preprocess_text(expected_answer)
    student_tokens = preprocess_text(student_answer)
    stopwords_set = set(['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'])
    expected_tokens = [t for t in expected_tokens if t not in stopwords_set]
    student_tokens = [t for t in student_tokens if t not in stopwords_set]
    
    if len(expected_tokens) == 0:
        return 1.0
    if len(student_tokens) == 0:
        return 0.0
    
    matched_count = 0
    for exp_token in expected_tokens:
        for stu_token in student_tokens:
            if fuzzy_token_match(exp_token, stu_token, threshold=0.85):
                matched_count += 1
                break
    return matched_count / len(expected_tokens)

def cosine_similarity_score(expected_answer, student_answer):
    vectorizer = TfidfVectorizer(
        tokenizer=preprocess_text,
        lowercase=True,
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform([expected_answer, student_answer])
    cosine_sim = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return cosine_sim

def sentiment_analysis(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return (sentiment_score + 1) / 2  # Normalize to [0,1]

def enhanced_sentence_match(expected_answer, student_answer):
    expected_normalized = clean_text(expected_answer)
    student_normalized = clean_text(student_answer)
    embeddings_expected = SENTENCE_MODEL.encode([expected_normalized])
    embeddings_student = SENTENCE_MODEL.encode([student_normalized])
    similarity = cosine_similarity([embeddings_expected.flatten()], [embeddings_student.flatten()])[0][0]
    return similarity

def multinomial_naive_bayes_score(expected_answer, student_answer):
    answers = [expected_answer.lower(), student_answer.lower()]
    vectorizer = CountVectorizer(
        tokenizer=preprocess_text,
        lowercase=True,
        stop_words='english'
    )
    X = vectorizer.fit_transform(answers)
    y = [0, 1]
    clf = MultinomialNB()
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    return probs[1][1]

def weighted_average_score(scores, weights):
    weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
    total_weight = sum(weights)
    return weighted_sum / total_weight

def semantic_similarity_score(expected_answer, student_answer):
    expected_normalized = clean_text(expected_answer)
    student_normalized = clean_text(student_answer)
    embeddings_expected = SENTENCE_MODEL.encode([expected_normalized])
    embeddings_student = SENTENCE_MODEL.encode([student_normalized])
    similarity = cosine_similarity([embeddings_expected.flatten()], [embeddings_student.flatten()])[0][0]
    return similarity

def coherence_score(expected_answer, student_answer):
    len_expected = len(word_tokenize(clean_text(expected_answer)))
    len_student = len(word_tokenize(clean_text(student_answer)))
    
    if len_expected == 0 or len_student == 0:
        return 0.5
    if len_student >= len_expected:
        return 0.9
    else:
        return len_student / len_expected

def relevance_score(expected_answer, student_answer):
    expected_tokens = set(word_tokenize(clean_text(expected_answer)))
    student_tokens = set(word_tokenize(clean_text(student_answer)))
    
    stopwords_set = set(['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'])
    expected_tokens = expected_tokens - stopwords_set
    student_tokens = student_tokens - stopwords_set
    
    if len(expected_tokens) == 0:
        return 1.0
    
    matched = 0
    for exp_token in expected_tokens:
        for stu_token in student_tokens:
            if fuzzy_token_match(exp_token, stu_token, threshold=0.85):
                matched += 1
                break
    return matched / len(expected_tokens)

def check_semantic_mismatch(expected_answer, student_answer):
    expected_lower = expected_answer.lower()
    student_lower = student_answer.lower()
    expected_tokens = set(expected_lower.split())
    student_tokens = set(student_lower.split())
    
    num_match_result = numbers_match(expected_answer, student_answer)
    if num_match_result is False:
        return True
    
    mammals = {'cat', 'dog', 'lion', 'tiger', 'elephant', 'horse', 'cow', 'bear', 'wolf', 'whale', 'dolphin'}
    birds = {'bird', 'eagle', 'sparrow', 'crow', 'parrot', 'penguin', 'owl', 'hawk'}
    fish = {'fish', 'shark', 'salmon', 'tuna', 'goldfish'}
    reptiles = {'snake', 'lizard', 'crocodile', 'turtle', 'alligator'}
    insects = {'insect', 'ant', 'bee', 'butterfly', 'spider', 'mosquito'}
    
    def get_animal_category(tokens):
        if tokens & mammals: return 'mammal'
        if tokens & birds: return 'bird'
        if tokens & fish: return 'fish'
        if tokens & reptiles: return 'reptile'
        if tokens & insects: return 'insect'
        return None
    
    exp_category = get_animal_category(expected_tokens)
    stu_category = get_animal_category(student_tokens)
    if exp_category and stu_category and exp_category != stu_category:
        return True
    
    cities = {'tokyo', 'paris', 'london', 'berlin', 'rome', 'madrid', 'beijing', 'moscow',
              'washington', 'munich', 'vienna', 'prague', 'cairo', 'delhi', 'sydney',
              'toronto', 'vancouver', 'seoul', 'bangkok', 'dubai', 'amsterdam'}
    if (expected_tokens & cities) and (student_tokens & cities) and (expected_tokens & cities) != (student_tokens & cities):
        return True
    
    countries = {'russia', 'china', 'usa', 'america', 'japan', 'india', 'brazil', 'germany',
                 'france', 'italy', 'spain', 'canada', 'australia', 'mexico', 'korea',
                 'england', 'britain', 'egypt', 'greece', 'turkey', 'iran', 'iraq'}
    if (expected_tokens & countries) and (student_tokens & countries) and (expected_tokens & countries) != (student_tokens & countries):
        return True
    
    us_presidents = {'washington', 'lincoln', 'jefferson', 'roosevelt', 'kennedy', 'obama', 
                     'trump', 'biden', 'clinton', 'bush', 'nixon', 'reagan', 'adams', 'wilson'}
    authors = {'shakespeare', 'dickens', 'austen', 'orwell', 'hemingway', 'twain', 'tolkien',
               'rowling', 'christie', 'king', 'poe', 'shelley', 'wilde', 'joyce'}
    artists = {'picasso', 'vinci', 'leonardo', 'michelangelo', 'monet', 'gogh', 'rembrandt',
               'dali', 'warhol', 'raphael', 'botticelli', 'caravaggio'}
    scientists = {'einstein', 'newton', 'darwin', 'curie', 'galileo', 'hawking', 'tesla',
                  'edison', 'pasteur', 'faraday', 'bohr', 'planck', 'turing'}
    explorers = {'columbus', 'magellan', 'polo', 'cook', 'armstrong', 'aldrin', 'gagarin'}
    
    for category in [us_presidents, authors, artists, scientists, explorers]:
        exp_people = expected_tokens & category
        stu_people = student_tokens & category
        if exp_people and stu_people and exp_people != stu_people:
            return True
    
    languages = {'english', 'japanese', 'chinese', 'spanish', 'french', 'german', 'italian',
                 'portuguese', 'russian', 'arabic', 'hindi', 'korean', 'dutch', 'greek'}
    if (expected_tokens & languages) and (student_tokens & languages) and (expected_tokens & languages) != (student_tokens & languages):
        return True
    
    planets = {'mercury', 'venus', 'earth', 'mars', 'jupiter', 'saturn', 'uranus', 'neptune', 'pluto'}
    if (expected_tokens & planets) and (student_tokens & planets) and (expected_tokens & planets) != (student_tokens & planets):
        return True
    
    basic_colors = {'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'black', 'white', 'pink', 'brown', 'gray', 'grey'}
    exp_colors = expected_tokens & basic_colors
    stu_colors = student_tokens & basic_colors
    if exp_colors and stu_colors and exp_colors != stu_colors:
        if len(expected_tokens) <= 3 and len(student_tokens) <= 3:
            return True
    
    return False

def containment_check(expected_answer, student_answer):
    expected_clean = clean_text(expected_answer)
    student_clean = clean_text(student_answer)
    
    if expected_clean in student_clean:
        return 1.0
    
    expected_tokens = set(preprocess_text(expected_answer))
    student_tokens = set(preprocess_text(student_answer))
    
    stopwords_set = set(['a', 'an', 'the', 'is', 'are', 'am', 'was', 'were', 'be', 'been', 'being', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by'])
    expected_tokens = expected_tokens - stopwords_set
    student_tokens = student_tokens - stopwords_set
    
    if len(expected_tokens) == 0:
        return 0.5
    
    contained = 0
    for exp_token in expected_tokens:
        for stu_token in student_tokens:
            if fuzzy_token_match(exp_token, stu_token, threshold=0.85):
                contained += 1
                break
    return contained / len(expected_tokens)

def synonym_boost(expected_answer, student_answer):
    expected_tokens = [t for t in preprocess_text(expected_answer) if len(t) > 2]
    student_tokens = [t for t in preprocess_text(student_answer) if len(t) > 2]
    
    synonym_count = 0
    for exp_token in expected_tokens:
        for stu_token in student_tokens:
            if exp_token != stu_token and check_synonyms(exp_token, stu_token):
                synonym_count += 1
                break
    if len(expected_tokens) == 0:
        return 0.0
    return synonym_count / len(expected_tokens)

def antonym_penalty(expected_answer, student_answer):
    expected_tokens = [t.lower() for t in preprocess_text(expected_answer)]
    student_tokens = [t.lower() for t in preprocess_text(student_answer)]
    expected_words = expected_answer.lower().split()
    student_words = student_answer.lower().split()
    
    all_expected = set(expected_tokens) | set(expected_words)
    all_student = set(student_tokens) | set(student_words)
    
    antonym_count = 0
    for exp_token in all_expected:
        for stu_token in all_student:
            if exp_token != stu_token and check_antonyms(exp_token, stu_token):
                antonym_count += 1
                break
    if len(all_expected) == 0:
        return 0.0
    penalty = antonym_count / len(all_expected)
    if antonym_count >= 1:
        penalty = min(1.0, penalty * 2.0)
    return penalty

def evaluate(expected, response, debug=False):
    """Main evaluation function with comprehensive scoring"""
    expected_normalized = clean_text(expected)
    response_normalized = clean_text(response)
    
    if expected_normalized == response_normalized:
        return 10
    elif not response or not response.strip():
        return 0
    
    num_match = numbers_match(expected, response)
    if num_match is True:
        return 9

    exact_match_score = exact_match(expected, response)
    partial_match_score = partial_match(expected, response)
    cosine_similarity_score_value = cosine_similarity_score(expected, response)
    sentiment_score = sentiment_analysis(response)
    enhanced_sentence_match_score = enhanced_sentence_match(expected, response)
    multinomial_naive_bayes_score_value = multinomial_naive_bayes_score(expected, response)
    semantic_similarity_value = semantic_similarity_score(expected, response)
    coherence_value = coherence_score(expected, response)
    relevance_value = relevance_score(expected, response)
    
    antonym_penalty_value = antonym_penalty(expected, response)
    synonym_boost_value = synonym_boost(expected, response)
    containment_value = containment_check(expected, response)
    semantic_mismatch = check_semantic_mismatch(expected, response)

    scores = [exact_match_score, partial_match_score, cosine_similarity_score_value,
              sentiment_score, enhanced_sentence_match_score, multinomial_naive_bayes_score_value,
              semantic_similarity_value, coherence_value, relevance_value]
    
    weights = [
        0.01, 0.05, 0.02, 0.00, 0.40, 0.02, 0.40, 0.02, 0.08
    ]
    
    scaled_scores = [score * 10 for score in scores]
    final_score = weighted_average_score(scaled_scores, weights)
    
    avg_semantic = (enhanced_sentence_match_score + semantic_similarity_value) / 2
    
    if avg_semantic >= 0.7:
        final_score = max(final_score, 9)
    elif avg_semantic >= 0.55:
        final_score = max(final_score, 8)
    elif avg_semantic >= 0.45:
        final_score = max(final_score, 7)
    
    if containment_value >= 0.8:
        final_score = max(final_score, 8)
        if containment_value == 1.0:
            final_score = max(final_score, 9)
    
    if synonym_boost_value > 0.2:
        if synonym_boost_value >= 0.5:
            final_score = max(final_score, 7)
            boost_amount = synonym_boost_value * 0.5
            final_score = min(10, final_score * (1 + boost_amount))
        else:
            boost_amount = synonym_boost_value * 0.4
            final_score = min(10, final_score * (1 + boost_amount))
    
    if antonym_penalty_value > 0:
        penalty_multiplier = max(0.1, 1 - (antonym_penalty_value * 1.8))
        final_score = final_score * penalty_multiplier
    
    if avg_semantic < 0.30:
        final_score = min(final_score, 3)
    if avg_semantic < 0.20:
        final_score = min(final_score, 1)
    
    if semantic_mismatch:
        final_score = min(final_score, 2)
    
    if check_contradiction(expected, response):
        final_score = min(final_score, 1)
    
    rounded_score = round(final_score)
    rounded_score = max(0, min(10, rounded_score))

    if debug:
        print(f"\n--- Evaluating: '{expected[:30]}...' vs '{response[:30]}...' ---")
        print(f"Exact Match: {exact_match_score:.2f} | Partial Match: {partial_match_score:.2f}")
        print(f"Cosine Sim: {cosine_similarity_score_value:.2f} | Sentiment: {sentiment_score:.2f}")
        print(f"Enhanced Match: {enhanced_sentence_match_score:.2f} | Naive Bayes: {multinomial_naive_bayes_score_value:.2f}")
        print(f"Semantic Sim: {semantic_similarity_value:.2f} | Coherence: {coherence_value:.2f} | Relevance: {relevance_value:.2f}")
        print(f"Containment: {containment_value:.2f} | Synonym Boost: {synonym_boost_value:.2f} | Antonym Penalty: {antonym_penalty_value:.2f}")
        print(f"Avg Semantic: {avg_semantic:.2f} | Semantic Mismatch: {semantic_mismatch} | Final Score: {rounded_score}/10\n")

    return rounded_score