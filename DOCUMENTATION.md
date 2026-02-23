# AI-Powered Answer Evaluation System Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [User Roles](#user-roles)
4. [Database Schema](#database-schema)
5. [Application Flow](#application-flow)
6. [AI Evaluation Algorithm](#ai-evaluation-algorithm)
7. [Route Documentation](#route-documentation)
8. [Setup Instructions](#setup-instructions)
9. [Security Considerations](#security-considerations)

---

## System Overview

This is an **AI-Powered Answer Evaluation System** built with Flask that automates the grading of descriptive answers using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The system supports three user roles: Admin, Teacher, and Student.

### Key Features
- Multi-role authentication system
- Test creation and management
- Automated answer evaluation using 9 different NLP/ML metrics
- Real-time score calculation and feedback
- Comprehensive admin panel for user management

### Technology Stack
- **Backend**: Flask (Python)
- **Database**: MySQL
- **NLP Libraries**: NLTK, Sentence Transformers, scikit-learn
- **ML Models**: SentenceTransformer (paraphrase-MiniLM-L6-v2)
- **Frontend**: HTML Templates (Jinja2)

---

## Architecture

### Three-Tier Architecture
```
┌─────────────────┐
│  Presentation   │  → HTML Templates (Jinja2)
│      Layer      │
└────────┬────────┘
         │
┌────────▼────────┐
│   Application   │  → Flask Routes & Business Logic
│      Layer      │
└────────┬────────┘
         │
┌────────▼────────┐
│   Data Layer    │  → MySQL Database
└─────────────────┘
```

---

## User Roles

### 1. Admin
**Responsibilities:**
- Manage student accounts (CRUD operations)
- Manage teacher accounts (CRUD operations)
- Monitor student performance
- View all tests and questions
- Delete student scores

**Entry Point:** `/admin/login`

### 2. Teacher
**Responsibilities:**
- Create and manage tests
- Add questions and expected answers
- View student submissions
- Monitor student performance on their tests

**Entry Point:** `/teacher_login`

### 3. Student
**Responsibilities:**
- Take available tests
- View their scores and feedback
- See expected answers vs their answers

**Entry Point:** `/student_login`

---

## Database Schema

### Tables

#### 1. Admins
```sql
- admin_id (Primary Key)
- username
- password
```

#### 2. Teachers
```sql
- teacher_id (Primary Key)
- username
- password
```

#### 3. Students
```sql
- student_id (Primary Key)
- username
- password
```

#### 4. Tests
```sql
- test_id (Primary Key)
- test_name
- teacher_id (Foreign Key → Teachers)
```

#### 5. Questions
```sql
- question_id (Primary Key)
- question_text
- test_id (Foreign Key → Tests)
```

#### 6. ExpectedAnswers
```sql
- answer_id (Primary Key)
- answer_text
- question_id (Foreign Key → Questions)
```

#### 7. StudentAnswers
```sql
- answer_id (Primary Key)
- student_id (Foreign Key → Students)
- test_id (Foreign Key → Tests)
- question_id (Foreign Key → Questions)
- answer_text
- score
```

#### 8. TeacherStudentRelationship
```sql
- teacher_id (Foreign Key → Teachers)
- student_id (Foreign Key → Students)
```

### Entity Relationship Diagram
```
Admins                    Teachers ──┐
                              │      │
                              │      │ Creates
                              │      ▼
Students ──┐                  │    Tests
           │                  │      │
           │ Takes             │      │ Contains
           │                  │      ▼
           └───────────────►  │  Questions
                             │      │
                             │      │ Has
                             │      ▼
                             └──► ExpectedAnswers
                                    │
                                    │ Compared with
                                    ▼
                              StudentAnswers
```

---

## Application Flow

### Admin Workflow

```
1. Admin Login (/admin/login)
   ↓
2. Admin Dashboard (/admin/home)
   ↓
3. Choose Action:
   ├─→ Manage Students (/admin/students)
   │   ├─→ Add Student (POST /admin/add_student)
   │   ├─→ Update Student (POST /admin/update_student/<id>)
   │   ├─→ Delete Student (POST /admin/delete_student/<id>)
   │   └─→ View Student Scores (/admin/view_student_scores/<id>)
   │
   └─→ Manage Teachers (/admin/teachers)
       ├─→ Add Teacher (/admin/add_teacher)
       ├─→ Update Teacher (/admin/update_teacher/<id>)
       ├─→ Delete Teacher (POST /admin/delete_teacher/<id>)
       └─→ View Teacher Tests (/admin/view_teacher_tests/<id>)
           └─→ View Test Questions (/admin/view_test_questions/<id>)
```

### Teacher Workflow

```
1. Teacher Login (/teacher_login)
   ↓
2. Teacher Home (/teacher_home)
   ↓
3. Choose Action:
   ├─→ Create New Test (POST with 'add_test_name')
   ├─→ Update Test Name (POST with 'update_test_name')
   ├─→ Delete Test (POST with 'delete_test_name')
   │   └─→ [Cascade Delete: Questions → ExpectedAnswers → StudentAnswers]
   │
   ├─→ Manage Questions (/teacher/view_test_questions/<test_id>)
   │   ├─→ Add Question (POST with 'add_question')
   │   │   └─→ Add Expected Answers
   │   └─→ Delete Question (POST with 'delete_question')
   │
   └─→ View Student Scores (/teacher_view_score)
       └─→ See all submissions grouped by student and test
```

### Student Workflow

```
1. Student Login (/student_login)
   ↓
2. Student Home (/student_home)
   ↓
3. Choose Action:
   ├─→ Take Test (/student_take_test)
   │   ├─→ View Available Tests (only untaken)
   │   ├─→ Select Test
   │   └─→ Answer Questions (/student_take_test/<test_id>)
   │       └─→ Submit Answers (POST)
   │           └─→ Store in StudentAnswers table
   │
   └─→ View Scores (/student_view_score)
       └─→ AI Evaluation Triggered Here
           ├─→ Calculate scores using evaluate()
           ├─→ Update scores in database
           └─→ Display results with feedback
```

---

## AI Evaluation Algorithm

### Overview
The `evaluate()` function uses **9 different NLP/ML metrics** to assess answer quality, combining them with weighted averaging to produce a final score (0-10).

### Evaluation Metrics

#### 1. Exact Match Score (Weight: 15%)
```python
def exact_match(expected_answer, student_answer)
```
- **Purpose**: Check if answers are identical
- **Method**: String comparison
- **Output**: 1 (match) or 0 (no match)

#### 2. Partial Match Score (Weight: 10%)
```python
def partial_match(expected_answer, student_answer)
```
- **Purpose**: Measure token overlap
- **Method**: 
  - Tokenize and lemmatize both answers
  - Calculate intersection of tokens
  - Return ratio of common tokens
- **Libraries**: NLTK (WordNetLemmatizer)

#### 3. Cosine Similarity Score (Weight: 10%)
```python
def cosine_similarity_score(expected_answer, student_answer)
```
- **Purpose**: Measure text similarity using TF-IDF
- **Method**:
  - Convert texts to TF-IDF vectors
  - Calculate cosine similarity
- **Libraries**: scikit-learn (TfidfVectorizer)

#### 4. Sentiment Analysis (Weight: 5%)
```python
def sentiment_analysis(text)
```
- **Purpose**: Analyze answer sentiment
- **Method**: VADER sentiment analysis
- **Output**: Normalized polarity score [0, 1]
- **Libraries**: NLTK (SentimentIntensityAnalyzer)

#### 5. Enhanced Sentence Match (Weight: 10%)
```python
def enhanced_sentence_match(expected_answer, student_answer)
```
- **Purpose**: Semantic similarity using embeddings
- **Method**:
  - Generate sentence embeddings
  - Calculate cosine similarity
- **Model**: SentenceTransformer ('paraphrase-MiniLM-L6-v2')

#### 6. Multinomial Naive Bayes Score (Weight: 10%)
```python
def multinomial_naive_bayes_score(expected_answer, student_answer)
```
- **Purpose**: Classification-based evaluation
- **Method**:
  - Convert to count vectors
  - Train Naive Bayes classifier
  - Return probability score
- **Libraries**: scikit-learn (MultinomialNB, CountVectorizer)

#### 7. Semantic Similarity Score (Weight: 10%)
```python
def semantic_similarity_score(expected_answer, student_answer)
```
- **Purpose**: Deep semantic comparison
- **Method**: Identical to Enhanced Sentence Match
- **Model**: SentenceTransformer

#### 8. Coherence Score (Weight: 10%)
```python
def coherence_score(expected_answer, student_answer)
```
- **Purpose**: Measure length appropriateness
- **Method**: 
  - Compare word counts
  - Return ratio of shorter/longer
- **Output**: Value closer to 1 indicates similar lengths

#### 9. Relevance Score (Weight: 10%)
```python
def relevance_score(expected_answer, student_answer)
```
- **Purpose**: Measure topic relevance
- **Method**:
  - Tokenize and lowercase both texts
  - Calculate token intersection
  - Return ratio of common tokens

### Scoring Algorithm

```python
def evaluate(expected, response):
    # Special cases
    if expected == response:
        return 10  # Perfect match
    if not response:
        return 0   # Empty answer
    
    # Calculate all 9 metrics (values: 0-1)
    scores = [
        exact_match_score,
        partial_match_score,
        cosine_similarity_score_value,
        sentiment_score,
        enhanced_sentence_match_score,
        multinomial_naive_bayes_score_value,
        semantic_similarity_value,
        coherence_value,
        relevance_value
    ]
    
    # Define weights (sum = 0.9)
    weights = [0.15, 0.1, 0.1, 0.05, 0.1, 0.1, 0.1, 0.1, 0.1]
    
    # Scale to 0-10 range
    scaled_scores = [score * 10 for score in scores]
    
    # Calculate weighted average
    final_score = sum(score * weight for score, weight in zip(scaled_scores, weights))
    
    # Round to nearest integer
    return round(final_score)
```

### Preprocessing Pipeline

```python
def preprocess_text(text):
    1. Tokenization (word_tokenize)
    2. Lemmatization (WordNetLemmatizer)
    3. Lowercase conversion
    4. Return processed tokens
```

### Model Loading
- **SentenceTransformer Model**: `paraphrase-MiniLM-L6-v2`
- **Loaded**: On first use (cached by library)
- **Purpose**: Generate 384-dimensional sentence embeddings

---

## Route Documentation

### Public Routes

#### `GET /`
- **Purpose**: Homepage
- **Template**: `Homepage.html`
- **Authentication**: None

---

### Admin Routes

#### `GET/POST /admin/login`
- **Purpose**: Admin authentication
- **Method**: 
  - GET: Display login form
  - POST: Validate credentials
- **Success**: Redirect to `/admin/home`
- **Failure**: Show error message

#### `GET /admin/home`
- **Purpose**: Admin dashboard
- **Authentication**: Required (`admin_logged_in` session)
- **Template**: `adminhome.html`

#### `GET /admin/students`
- **Purpose**: List all students
- **Query**: `SELECT * FROM Students`
- **Template**: `admin_students.html`

#### `POST /admin/add_student`
- **Purpose**: Create new student
- **Parameters**: `username`, `password`
- **Query**: `INSERT INTO Students`

#### `POST /admin/update_student/<int:student_id>`
- **Purpose**: Update student credentials
- **Parameters**: `username`, `password`

#### `POST /admin/delete_student/<int:student_id>`
- **Purpose**: Delete student and related data

#### `GET /admin/view_student_scores/<int:student_id>`
- **Purpose**: View all scores for a student
- **Query**: Complex JOIN across multiple tables
- **Template**: `student_scores.html`

#### `POST /admin/delete_student_score/<int:answer_id>`
- **Purpose**: Delete specific student answer

#### `GET /admin/teachers`
- **Purpose**: List all teachers
- **Template**: `admin_teachers.html`

#### `GET/POST /admin/add_teacher`
- **Purpose**: Create new teacher account

#### `GET/POST /admin/update_teacher/<int:teacher_id>`
- **Purpose**: Update teacher credentials
- **Template**: `update_teacher.html`

#### `POST /admin/delete_teacher/<int:teacher_id>`
- **Purpose**: Delete teacher
- **Cascade**: Deletes TeacherStudentRelationship records

#### `GET /admin/view_teacher_tests/<int:teacher_id>`
- **Purpose**: View tests created by teacher
- **Template**: `view_teacher_tests.html`

#### `GET /admin/view_test_questions/<int:test_id>`
- **Purpose**: View questions for a test
- **Template**: `view_test_questions.html`

#### `GET /admin/view_question_answers/<int:question_id>`
- **Purpose**: View expected answers
- **Template**: `view_question_answers.html`

#### `GET /admin/logout`
- **Purpose**: End admin session

---

### Teacher Routes

#### `GET/POST /teacher_login`
- **Purpose**: Teacher authentication
- **Session Data**: `teacher_logged_in`, `teacher_id`

#### `GET/POST /teacher_home`
- **Purpose**: Manage tests
- **Actions**:
  - Add test name
  - Update test name
  - Delete test (with cascade)
- **Template**: `teacher_home.html`

#### `GET/POST /teacher/view_test_questions/<int:test_id>`
- **Purpose**: Manage questions
- **Actions**:
  - Add question with expected answers
  - Delete question
- **Template**: `view_teacher_test_questions.html`

#### `GET /teacher_view_score`
- **Purpose**: View all student scores
- **Data Structure**: Grouped by student and test
- **Template**: `teacher_view_score.html`

#### `GET /teacher_logout`
- **Purpose**: End teacher session

---

### Student Routes

#### `GET/POST /student_login`
- **Purpose**: Student authentication
- **Session Data**: `student_logged_in`, `student_id`

#### `GET /student_home`
- **Purpose**: Student dashboard
- **Template**: `student_home.html`

#### `GET/POST /student_take_test`
- **Purpose**: List available tests
- **Filter**: Only shows tests not yet taken
- **Template**: `student_take_test.html`

#### `GET/POST /student_take_test/<int:test_id>`
- **Purpose**: Answer test questions
- **Method**:
  - GET: Display questions
  - POST: Store answers in StudentAnswers
- **Template**: `student_take_test_questions.html`

#### `GET /student_view_score`
- **Purpose**: View evaluated scores
- **Process**:
  1. Fetch student answers and expected answers
  2. Call `evaluate()` for each question
  3. Update scores in database
  4. Display results grouped by test
- **Template**: `student_view_score.html`

#### `GET /student_logout`
- **Purpose**: End student session

---

## Setup Instructions

### Prerequisites
- **Python 3.8+** (with pip package manager)
- **MySQL Server** (running and accessible)
- **Internet Connection** (for downloading models and data)
- **Minimum 4GB RAM** (for ML models)
- **2GB Free Disk Space** (for NLP models)

### Step 1: Clone/Download Project
```bash
cd c:\Anandukc\mass
```

### Step 2: Install Python Libraries
```bash
pip install -r requirements.txt
```

This installs:
- **Flask** (3.1.2) - Web framework
- **Flask-MySQLdb** (2.0.0) - MySQL connector
- **pandas** (3.0.0) - Data manipulation
- **nltk** (3.9.2) - Natural Language Toolkit
- **sentence-transformers** (5.2.3) - Semantic text embeddings
- **scikit-learn** (1.8.0) - Machine learning algorithms
- **language-tool-python** (3.2.2) - Grammar checking
- **mysql-connector-python** (9.6.0) - MySQL driver

### Step 3: Download NLTK Data Packages ⚠️ IMPORTANT

After installing NLTK library, you MUST download required data packages:

**Option 1: Python Script (Recommended)**
```python
import nltk

# Download required NLTK data
nltk.download('stopwords')      # English stopwords list
nltk.download('punkt')          # Sentence/word tokenizer
nltk.download('wordnet')        # WordNet lemmatizer
nltk.download('vader_lexicon')  # Sentiment analysis lexicon
```

**Option 2: Interactive Downloader**
```python
import nltk
nltk.download()  # Opens GUI to select packages
```

**Option 3: Command Line**
```bash
python -m nltk.downloader stopwords punkt wordnet vader_lexicon
```

**Verification:**
```python
# Test if NLTK data is properly installed
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

# This should work without errors
print(stopwords.words('english')[:5])
print(word_tokenize("This is a test"))
```

### Step 4: Download Sentence Transformer Model ⚠️ IMPORTANT

The first time you run the application, it will automatically download the pre-trained model:

**Model:** `paraphrase-MiniLM-L6-v2`  
**Size:** ~90MB  
**Location:** `~/.cache/torch/sentence_transformers/`

**Manual Download (if needed):**
```python
from sentence_transformers import SentenceTransformer

# This will download the model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
print("Model downloaded successfully!")
```

### Step 5: Setup MySQL Database
```sql
CREATE DATABASE answer_evaluation;
```

Run the database creation script:
```bash
python create_tables.py
```

### Step 6: Configure Database Connection
Edit `admin.py`:
```python
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'your_password'  # Update this
app.config['MYSQL_DB'] = 'answer_evaluation'
```

### Step 7: Run Application
```bash
python admin.py
```

Access at: `http://localhost:5000`

---

## Configuration

### Flask Settings
```python
app.secret_key = 'your_secret_key'  # Change in production
app.template_folder = 'templates'
```

### MySQL Configuration
```python
MYSQL_HOST = 'localhost'
MYSQL_USER = 'root'
MYSQL_PASSWORD = 'akc2sysit25$'
MYSQL_DB = 'answer_evaluation'
```

### NLP Models
- **SentenceTransformer**: `paraphrase-MiniLM-L6-v2`
- **NLTK Stopwords**: English
- **Lemmatizer**: WordNet

---

## Security Considerations

### Current Security Issues ⚠️

1. **Plain Text Passwords**
   - Passwords stored without hashing
   - **Risk**: Database breach exposes all credentials
   - **Solution**: Use `werkzeug.security` or `bcrypt`
   ```python
   from werkzeug.security import generate_password_hash, check_password_hash
   ```

2. **Hardcoded Secret Key**
   - Secret key visible in source code
   - **Risk**: Session hijacking
   - **Solution**: Use environment variables
   ```python
   app.secret_key = os.environ.get('SECRET_KEY')
   ```

3. **SQL Injection Risk**
   - Using string formatting with user input
   - **Current Status**: Mostly mitigated with parameterized queries
   - **Recommendation**: Always use `%s` placeholders

4. **No CSRF Protection**
   - Forms lack CSRF tokens
   - **Risk**: Cross-site request forgery
   - **Solution**: Use Flask-WTF

5. **No Input Validation**
   - User inputs not sanitized
   - **Risk**: XSS attacks
   - **Solution**: Implement validation and sanitization

6. **Session Management**
   - No session timeout
   - No "Remember Me" option
   - **Solution**: Implement session expiration

### Recommended Security Improvements

```python
# 1. Password Hashing
from werkzeug.security import generate_password_hash, check_password_hash

# On registration
hashed = generate_password_hash(password, method='pbkdf2:sha256')

# On login
if check_password_hash(stored_hash, password):
    # Login successful

# 2. CSRF Protection
from flask_wtf.csrf import CSRFProtect
csrf = CSRFProtect(app)

# 3. Input Validation
from wtforms import StringField, PasswordField
from wtforms.validators import DataRequired, Length

# 4. Environment Variables
import os
from dotenv import load_dotenv

load_dotenv()
app.config['MYSQL_PASSWORD'] = os.getenv('DB_PASSWORD')
app.secret_key = os.getenv('SECRET_KEY')
```

---

## Performance Optimization

### Current Bottlenecks

1. **ML Model Loading**
   - SentenceTransformer loaded multiple times
   - **Solution**: Load once at startup
   ```python
   # At module level
   MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')
   ```

2. **Database Queries**
   - Multiple queries in loops
   - **Solution**: Use JOIN queries, batch operations

3. **Score Calculation**
   - `evaluate()` called in render loop
   - **Solution**: Calculate during submission, store in DB

### Recommended Optimizations

```python
# 1. Cache ML Model
@app.before_first_request
def initialize():
    global MODEL
    MODEL = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 2. Batch Processing
def evaluate_all_answers(student_id, test_id):
    # Fetch all at once
    # Process in batch
    # Update in single transaction

# 3. Database Indexing
CREATE INDEX idx_student_test ON StudentAnswers(student_id, test_id);
CREATE INDEX idx_test_teacher ON Tests(teacher_id);
```

---

## Error Handling

### Current State
- Minimal error handling
- Some try-except blocks for deletions
- Print statements for debugging

### Recommended Improvements

```python
import logging

# Setup logging
logging.basicConfig(
    filename='app.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s'
)

# In routes
try:
    # Database operation
    pass
except Exception as e:
    logging.error(f"Error in route: {e}")
    flash("An error occurred. Please try again.")
    return redirect(url_for('error_page'))
```

---

## Testing

### Manual Testing Checklist

**Admin:**
- [ ] Login with valid/invalid credentials
- [ ] Add/Update/Delete student
- [ ] Add/Update/Delete teacher
- [ ] View student scores
- [ ] View teacher tests

**Teacher:**
- [ ] Login with valid/invalid credentials
- [ ] Create/Update/Delete test
- [ ] Add/Delete questions
- [ ] Add expected answers
- [ ] View student scores

**Student:**
- [ ] Login with valid/invalid credentials
- [ ] View available tests
- [ ] Take test (answer questions)
- [ ] Submit answers
- [ ] View scores

**AI Evaluation:**
- [ ] Exact match returns 10
- [ ] Empty answer returns 0
- [ ] Similar answers get high scores
- [ ] Dissimilar answers get low scores

### Unit Test Example

```python
import unittest

class TestEvaluate(unittest.TestCase):
    def test_exact_match(self):
        score = evaluate("Python is a language", "Python is a language")
        self.assertEqual(score, 10)
    
    def test_empty_answer(self):
        score = evaluate("Expected", "")
        self.assertEqual(score, 0)
    
    def test_similar_answer(self):
        score = evaluate(
            "Python is a programming language",
            "Python is a language for programming"
        )
        self.assertGreater(score, 5)
```

---

## Troubleshooting

### Common Issues

#### 1. NLTK Data Missing ⚠️ MOST COMMON
```
LookupError: Resource stopwords not found
LookupError: Resource punkt not found
LookupError: Resource wordnet not found
LookupError: Resource vader_lexicon not found
```
**Cause:** NLTK library installed but data packages not downloaded

**Solution:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('omw-1.4')  # Optional, for wordnet
```

**Or download all at once:**
```python
import nltk
nltk.download('all')  # Warning: ~3.5GB download
```

**Check download location:**
```python
import nltk
print(nltk.data.path)
# Default Windows: C:\Users\YourName\AppData\Roaming\nltk_data
# Default Linux/Mac: ~/nltk_data
```

#### 2. Sentence Transformer Model Issues
```
OSError: Can't load model paraphrase-MiniLM-L6-v2
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]>
```
**Cause:** Model not downloaded or network issues

**Solution 1 - Manual Download:**
```python
from sentence_transformers import SentenceTransformer
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
```

**Solution 2 - Check Cache:**
```python
import os
cache_dir = os.path.expanduser('~/.cache/torch/sentence_transformers/')
print(f"Model cache: {cache_dir}")
# If empty, model needs to be downloaded
```

**Solution 3 - Use Proxy (if behind firewall):**
```python
import os
os.environ['HTTP_PROXY'] = 'http://proxy.example.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.example.com:8080'
```

#### 3. Import Errors
```
ModuleNotFoundError: No module named 'nltk'
ModuleNotFoundError: No module named 'sentence_transformers'
```
**Cause:** Libraries not installed

**Solution:**
```bash
pip install nltk==3.9.2
pip install sentence-transformers==5.2.3
pip install scikit-learn==1.8.0
# Or install all:
pip install -r requirements.txt
```

**Check if installed:**
```bash
pip list | grep nltk
pip list | grep sentence-transformers
```

#### 4. Memory Error During Evaluation
```
MemoryError: Unable to allocate array
RuntimeError: CUDA out of memory
```
**Cause:** Insufficient RAM for ML models

**Solution:**
- Close other applications
- Process answers in smaller batches
- Use CPU instead of GPU:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU usage
```

#### 5. Database Connection Error
```
Error: Can't connect to MySQL server
```
**Solution:**
- Verify MySQL is running
- Check credentials in `admin.py`
- Ensure database exists

#### 6. NLTK Data Missing
```
LookupError: Resource stopwords not found
```
**Solution:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('vader_lexicon')
```

#### 7. Model Download Issues
```
Error downloading sentence-transformers model
```
**Solution:**
- Check internet connection
- Manually download model
- Set HuggingFace cache directory

#### 8. Port Already in Use
```
OSError: [Errno 48] Address already in use
```
**Solution:**
```bash
# Change port in admin.py
app.run(debug=True, port=5001)
```

#### 9. Template Not Found
```
jinja2.exceptions.TemplateNotFound
```
**Solution:**
- Verify templates/ directory exists
- Check file names match route references
- Ensure correct case sensitivity

### NLP-Specific Troubleshooting

#### Slow First Run
**Issue:** First evaluation takes 30-60 seconds  
**Cause:** Model loading and NLTK initialization  
**Normal Behavior:** Subsequent evaluations are much faster

#### Inconsistent Scores
**Issue:** Same answer gets different scores  
**Cause:** Model uses floating-point calculations  
**Solution:** Scores rounded to integers for consistency

#### Low Scores for Good Answers
**Issue:** Similar answer gets low score  
**Debugging:**
```python
# Add this to evaluate() function to see individual scores:
print("Exact Match Score:", exact_match_score)
print("Partial Match Score:", partial_match_score)
print("Semantic Similarity Score:", semantic_similarity_value)
# Check which metric is causing low score
```

---

## Future Enhancements

### Planned Features

1. **Advanced Analytics**
   - Performance trends over time
   - Comparative analysis
   - Difficulty rating for questions

2. **Enhanced AI**
   - Grammar checking using language-tool-python
   - Plagiarism detection
   - Answer completeness scoring

3. **User Experience**
   - Real-time answer preview
   - Auto-save functionality
   - Mobile-responsive design

4. **Administration**
   - Bulk user import (CSV)
   - Email notifications
   - Report generation (PDF)

5. **Question Types**
   - Multiple choice
   - True/False
   - Fill in the blanks
   - Mixed question types

6. **Advanced Features**
   - Time-limited tests
   - Randomized questions
   - Question pools
   - Partial credit system

---

## API Documentation (Future)

If converting to REST API:

```
POST /api/v1/auth/login
GET  /api/v1/tests
POST /api/v1/tests
GET  /api/v1/tests/{id}
POST /api/v1/tests/{id}/submit
GET  /api/v1/scores
```

---

## Contributing

For team members working on this project:

1. Follow PEP 8 style guidelines
2. Add docstrings to all functions
3. Write unit tests for new features
4. Update this documentation
5. Use meaningful commit messages

---

## License

[Specify your license here]

---

## Contact & Support

**Developer**: [Your Name]
**Email**: [Your Email]
**Repository**: [GitHub URL]

---

## Changelog

### Version 1.0.0 (Current)
- Initial release
- Three-role authentication system
- AI-powered answer evaluation
- Admin, Teacher, Student modules
- 9 NLP/ML evaluation metrics

### Planned for Version 1.1.0
- Password hashing
- CSRF protection
- Enhanced error handling
- Performance optimizations
- Grammar checking integration

---

## Quick Start Guide for NLP Setup

### Complete Setup Commands (Copy-Paste Ready)

```bash
# 1. Install Python packages
pip install -r requirements.txt

# 2. Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('vader_lexicon')"

# 3. Download Sentence Transformer model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"

# 4. Verify installation
python -c "import nltk; from sentence_transformers import SentenceTransformer; from sklearn.feature_extraction.text import TfidfVectorizer; print('All NLP libraries installed successfully!')"
```

### What Gets Installed?

| Component | Size | Purpose | Installation Time |
|-----------|------|---------|------------------|
| **NLTK Library** | ~10MB | Text processing | 10-30 seconds |
| **NLTK Data** | ~50MB | Stopwords, tokenizers | 30-60 seconds |
| **Sentence Transformers** | ~200MB | Deep learning library | 1-3 minutes |
| **Pre-trained Model** | ~90MB | Semantic embeddings | 1-2 minutes |
| **Scikit-learn** | ~30MB | ML algorithms | 30-60 seconds |
| **PyTorch** | ~750MB | Deep learning backend | 3-5 minutes |
| **Total** | **~1.13GB** | | **6-12 minutes** |

### Verification Script

Save as `test_nlp_setup.py`:

```python
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

def test_evaluation():
    try:
        # Test a simple evaluation
        from admin import evaluate
        score = evaluate("Python is a programming language", 
                        "Python is a language for programming")
        print(f"✓ Evaluation function working (test score: {score}/10)")
        return True
    except Exception as e:
        print(f"✗ Evaluation Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("NLP Setup Verification")
    print("=" * 50)
    
    results = {
        "NLTK": test_nltk(),
        "Sentence Transformers": test_sentence_transformers(),
        "Scikit-learn": test_sklearn(),
        "Evaluation Function": test_evaluation()
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
        print("\nRun: pip install -r requirements.txt")
        print("Then: python -c \"import nltk; nltk.download('all')\"")
    print("=" * 50)
```

Run it:
```bash
python test_nlp_setup.py
```

---

## Appendix

### Evaluation Weights Summary
| Metric | Weight | Purpose |
|--------|--------|---------|
| Exact Match | 15% | Identical answers |
| Partial Match | 10% | Token overlap |
| Cosine Similarity | 10% | TF-IDF similarity |
| Sentiment Analysis | 5% | Answer tone |
| Enhanced Sentence Match | 10% | Semantic similarity |
| Multinomial Naive Bayes | 10% | Classification |
| Semantic Similarity | 10% | Deep semantics |
| Coherence | 10% | Length appropriateness |
| Relevance | 10% | Topic relevance |
| **Total** | **90%** | |

### Database Commands

```sql
-- View all students
SELECT * FROM Students;

-- View student scores
SELECT s.username, t.test_name, SUM(sa.score) as total_score
FROM Students s
JOIN StudentAnswers sa ON s.student_id = sa.student_id
JOIN Tests t ON sa.test_id = t.test_id
GROUP BY s.student_id, t.test_id;

-- View test statistics
SELECT t.test_name, COUNT(DISTINCT sa.student_id) as students_taken,
       AVG(sa.score) as average_score
FROM Tests t
LEFT JOIN StudentAnswers sa ON t.test_id = sa.test_id
GROUP BY t.test_id;
```

---

**Document Version**: 1.0  
**Last Updated**: February 17, 2026  
**Status**: Current
