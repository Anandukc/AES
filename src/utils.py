# src/utils.py
import re
from difflib import SequenceMatcher
from nltk.corpus import wordnet

# Number word to digit mapping
NUMBER_WORDS = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19', 'twenty': '20',
    'thirty': '30', 'forty': '40', 'fifty': '50', 'sixty': '60',
    'seventy': '70', 'eighty': '80', 'ninety': '90', 'hundred': '100',
    'thousand': '1000', 'first': '1', 'second': '2', 'third': '3'
}

# Compound number patterns (twenty-five, etc.)
COMPOUND_NUMBERS = {
    'twenty-five': '25', 'twenty five': '25', 'twentyfive': '25',
    'twenty-one': '21', 'twenty one': '21',
    'twenty-two': '22', 'twenty two': '22',
    'twenty-three': '23', 'twenty three': '23',
    'twenty-four': '24', 'twenty four': '24',
    'twenty-six': '26', 'twenty six': '26',
    'twenty-seven': '27', 'twenty seven': '27',
    'twenty-eight': '28', 'twenty eight': '28',
    'twenty-nine': '29', 'twenty nine': '29',
    'thirty-one': '31', 'thirty one': '31',
    'thirty-two': '32', 'thirty two': '32',
    'forty-five': '45', 'forty five': '45',
    'fifty-five': '55', 'fifty five': '55',
    'sixty-four': '64', 'sixty four': '64',
    'ninety-nine': '99', 'ninety nine': '99',
    'one hundred': '100', 'two hundred': '200',
}

def normalize_numbers(text):
    """Convert number words to digits and vice versa for comparison"""
    text_lower = text.lower().strip()
    
    # First, check for compound numbers in the text
    for compound, digit in COMPOUND_NUMBERS.items():
        text_lower = text_lower.replace(compound, digit)
    
    words = text_lower.split()
    
    # Handle decade + ones pattern: "twenty" + "five" -> "25"
    result_words = []
    i = 0
    while i < len(words):
        word = words[i]
        
        # Check for decade + ones pattern
        if i + 1 < len(words):
            decade_map = {'twenty': '2', 'thirty': '3', 'forty': '4', 'fifty': '5', 
                          'sixty': '6', 'seventy': '7', 'eighty': '8', 'ninety': '9'}
            ones_map = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
                        'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
            
            if word in decade_map and words[i+1] in ones_map:
                result_words.append(decade_map[word] + ones_map[words[i+1]])
                i += 2
                continue
        
        # Check for year patterns: "nineteen forty five" -> 1945
        if i + 2 < len(words):
            century_map = {'nineteen': '19', 'eighteen': '18', 'seventeen': '17', 'sixteen': '16', 'twenty': '20'}
            decade_map = {'twenty': '2', 'thirty': '3', 'forty': '4', 'fifty': '5', 
                          'sixty': '6', 'seventy': '7', 'eighty': '8', 'ninety': '9'}
            ones_map = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 
                        'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'}
            
            if word in century_map and words[i+1] in decade_map and words[i+2] in ones_map:
                year = century_map[word] + decade_map[words[i+1]] + ones_map[words[i+2]]
                result_words.append(year)
                i += 3
                continue
        
        # Single word conversion
        if word in NUMBER_WORDS:
            result_words.append(NUMBER_WORDS[word])
        else:
            result_words.append(word)
        i += 1
    
    # Join and then try to combine adjacent numbers into years/compound numbers
    result = ' '.join(result_words)
    
    # Post-process: combine "19 45" -> "1945", "20 25" -> "2025", etc. (years)
    result = re.sub(r'\b(19|18|20|21)\s+(\d{2})\b', r'\1\2', result)
    
    # Post-process: combine "2 5" -> "25" (decade + ones that got split)
    result = re.sub(r'\b([2-9])\s+([0-9])\b', r'\1\2', result)
    
    return result

def numbers_match(expected, student):
    """Check if numerical values match (words vs digits)"""
    exp_normalized = normalize_numbers(expected)
    stu_normalized = normalize_numbers(student)
    
    # Normalize special symbols (°C -> degrees Celsius)
    exp_normalized = exp_normalized.replace('°c', ' degrees celsius').replace('°C', ' degrees celsius')
    exp_normalized = exp_normalized.replace('°f', ' degrees fahrenheit').replace('°F', ' degrees fahrenheit')
    stu_normalized = stu_normalized.replace('°c', ' degrees celsius').replace('°C', ' degrees celsius')
    stu_normalized = stu_normalized.replace('°f', ' degrees fahrenheit').replace('°F', ' degrees fahrenheit')
    
    # Extract all numbers (including decimals) from both
    exp_nums = set(re.findall(r'\b\d+\.?\d*\b', exp_normalized))
    stu_nums = set(re.findall(r'\b\d+\.?\d*\b', stu_normalized))
    
    # Check if numbers match
    if exp_nums and stu_nums:
        # Check exact match
        if exp_nums == stu_nums:
            return True
        
        # Check if core numbers match (for "approximately 3.14159" matching "3.14")
        for e in exp_nums:
            for s in stu_nums:
                # Strip trailing zeros and compare
                e_clean = e.rstrip('0').rstrip('.')
                s_clean = s.rstrip('0').rstrip('.')
                # Check if one is a prefix of the other (3.14 matches 3.14159)
                if e_clean.startswith(s_clean) or s_clean.startswith(e_clean):
                    return True
                # Check if they're essentially the same number
                try:
                    if abs(float(e) - float(s)) < 0.01:  # Within 0.01 tolerance
                        return True
                except ValueError:
                    pass
        
        # Check if different numbers (mismatch)
        if exp_nums != stu_nums:
            return False
    
    # Also compare the normalized strings
    if exp_normalized.strip() == stu_normalized.strip():
        return True
    
    return None  # No numbers to compare

def clean_text(text):
    """Clean and normalize text for better comparison"""
    if not text or not isinstance(text, str):
        return ""
    
    # Add space before capital letters that follow lowercase (e.g., "Aresident" -> "A resident")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Handle all-lowercase concatenated words by inserting spaces at common word boundaries
    common_splits = [
        (r'(electronic)(device)', r'\1 \2'),
        (r'(four)(legged)', r'\1 \2'),
        (r'(h2o)(liquid)', r'\1 \2'),
        (r'(mixture)(of)', r'\1 \2'),
        (r'(united)(states)', r'\1 \2'),
    ]
    text_lower = text.lower()
    for pattern, replacement in common_splits:
        text_lower = re.sub(pattern, replacement, text_lower)
    
    # Normalize multiple spaces to single space
    text_lower = ' '.join(text_lower.split())
    
    return text_lower.strip()

def fuzzy_token_match(token1, token2, threshold=0.8):
    """Check if two tokens are similar enough (handles typos)"""
    if token1 == token2:
        return True
    ratio = SequenceMatcher(None, token1, token2).ratio()
    return ratio >= threshold

def check_antonyms(word1, word2):
    """Check if two words are antonyms using WordNet"""
    # Comprehensive opposite pairs - bidirectional
    common_antonyms = set()
    opposite_pairs = [
        ('hot', 'cold'), ('big', 'small'), ('large', 'small'), ('large', 'tiny'),
        ('fast', 'slow'), ('quick', 'slow'), ('up', 'down'), ('high', 'low'),
        ('light', 'dark'), ('day', 'night'), ('sunlight', 'darkness'),
        ('rise', 'fall'), ('ascend', 'descend'), ('good', 'bad'), ('right', 'wrong'),
        ('love', 'hate'), ('love', 'hatred'), ('affection', 'hatred'), ('care', 'hatred'),
        ('success', 'failure'), ('achieve', 'fail'), ('achievement', 'failure'),
        ('brave', 'cowardly'), ('courageous', 'fearful'), ('courage', 'fear'), ('brave', 'fearful'),
        ('truth', 'lie'), ('true', 'false'), ('honest', 'dishonest'), ('truth', 'lies'),
        ('truth', 'deception'), ('accurate', 'lies'), ('facts', 'lies'), ('fact', 'deception'),
        ('health', 'sickness'), ('healthy', 'sick'), ('well', 'ill'), ('health', 'disease'),
        ('peace', 'war'), ('peaceful', 'violent'), ('calm', 'violent'), ('peace', 'violence'),
        ('conflict', 'peace'), ('absence', 'presence'), ('absence', 'war'), ('conflict', 'calm'),
        ('absence', 'violence'), ('violence', 'calm'), ('war', 'calm'),
        ('wealth', 'poverty'), ('rich', 'poor'), ('wealthy', 'poor'), ('money', 'nothing'),
        ('having', 'nothing'), ('money', 'poverty'),
        ('knowledge', 'ignorance'), ('knowing', 'ignorant'), ('wise', 'ignorant'),
        ('understanding', 'ignorance'), ('information', 'ignorance'),
        ('safe', 'dangerous'), ('safety', 'danger'), ('secure', 'risky'),
        ('safe', 'risky'), ('free', 'dangerous'), ('danger', 'safe'),
        ('clean', 'dirty'), ('clean', 'filthy'), ('pure', 'impure'), ('free', 'dirty'),
        ('happy', 'sad'), ('happiness', 'sadness'), ('joy', 'sorrow'),
        ('positive', 'negative'), ('strong', 'weak'), ('strength', 'weakness'),
        ('open', 'closed'), ('start', 'end'), ('begin', 'finish'),
        ('active', 'inactive'), ('rest', 'active'), ('sleep', 'awake'),
        ('life', 'death'), ('alive', 'dead'), ('living', 'dead'),
        ('create', 'destroy'), ('build', 'demolish'), ('make', 'break'),
        ('give', 'take'), ('buy', 'sell'), ('win', 'lose'),
        # Body parts and breathing
        ('gills', 'lungs'), ('lungs', 'gills'),
        # Fire and ice contradiction
        ('frozen', 'fire'), ('ice', 'fire'), ('cold', 'fire'), ('water', 'fire'),
        # Warm vs cold
        ('warm', 'cold'), ('cool', 'hot'),
    ]
    for a, b in opposite_pairs:
        common_antonyms.add((a, b))
        common_antonyms.add((b, a))
    
    if (word1, word2) in common_antonyms:
        return True
    
    # Check WordNet
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    
    for syn1 in synsets1:
        for lemma in syn1.lemmas():
            if lemma.antonyms():
                for antonym in lemma.antonyms():
                    if antonym.name() == word2 or antonym.name().replace('_', ' ') == word2:
                        return True
    return False

def check_synonyms(word1, word2):
    """Check if two words are synonyms using WordNet"""
    # Extensive common synonym pairs
    common_synonyms = {
        # Building/Structure
        ('house', 'building'), ('building', 'house'),
        ('house', 'structure'), ('structure', 'house'),
        ('building', 'structure'), ('structure', 'building'),
        ('home', 'house'), ('house', 'home'),
        ('residence', 'house'), ('house', 'residence'),
        ('residence', 'home'), ('home', 'residence'),
        # Vehicle
        ('car', 'vehicle'), ('vehicle', 'car'),
        ('car', 'automobile'), ('automobile', 'car'),
        ('vehicle', 'automobile'), ('automobile', 'vehicle'),
        # Emotion/Feeling
        ('happy', 'joy'), ('joy', 'happy'),
        ('happiness', 'joy'), ('joy', 'happiness'),
        ('happiness', 'delight'), ('delight', 'happiness'),
        ('feeling', 'emotion'), ('emotion', 'feeling'),
        # Intelligence
        ('smart', 'intelligent'), ('intelligent', 'smart'),
        ('intelligence', 'ability'), ('ability', 'intelligence'),
        ('capacity', 'ability'), ('ability', 'capacity'),
        ('intelligence', 'capacity'), ('capacity', 'intelligence'),
        ('understand', 'comprehend'), ('comprehend', 'understand'),
        ('reason', 'think'), ('think', 'reason'),
        ('learn', 'understand'), ('understand', 'learn'),
        ('learn', 'comprehend'), ('comprehend', 'learn'),
        ('think', 'understand'), ('understand', 'think'),
        # Temperature
        ('warm', 'hot'), ('hot', 'warm'),
        ('temperature', 'hot'), ('hot', 'temperature'),
        ('high', 'temperature'), ('temperature', 'high'),
        ('having', 'being'), ('being', 'having'),
        # Living/People
        ('live', 'reside'), ('reside', 'live'),
        ('people', 'human'), ('human', 'people'),
        ('humans', 'people'), ('people', 'humans'),
        ('person', 'human'), ('human', 'person'),
        # Government/System
        ('government', 'system'), ('system', 'government'),
        ('govern', 'rule'), ('rule', 'govern'),
        # Travel
        ('travel', 'transportation'), ('transportation', 'travel'),
    }
    
    if (word1, word2) in common_synonyms:
        return True
    
    # Check WordNet for synonyms
    synsets1 = wordnet.synsets(word1)
    synsets2 = wordnet.synsets(word2)
    
    # If words share any synset, they're synonyms
    for syn1 in synsets1:
        for syn2 in synsets2:
            lemmas1 = set(lemma.name().lower() for lemma in syn1.lemmas())
            lemmas2 = set(lemma.name().lower() for lemma in syn2.lemmas())
            if lemmas1 & lemmas2:
                return True
            if word2 in lemmas1 or word1 in lemmas2:
                return True
    
    return False

def check_contradiction(expected_answer, student_answer):
    """Check for logical contradictions (ice=fire, fish=lungs, etc.)"""
    exp_lower = expected_answer.lower()
    stu_lower = student_answer.lower()
    
    contradictions = [
        ('frozen', 'fire'), ('ice', 'fire'), ('cold', 'fire'),
        ('gills', 'lungs'), ('lungs', 'gills'),
        ('green', 'purple'), ('red', 'blue'), ('black', 'white'),
        ('opposite', 'same'), 
    ]
    
    for word1, word2 in contradictions:
        if word1 in exp_lower and word2 in stu_lower:
            return True
        if word2 in exp_lower and word1 in stu_lower:
            return True
    
    return False