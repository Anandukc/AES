# Answer Evaluation System

A web-based application that uses Machine Learning and Natural Language Processing techniques to evaluate student answers against expected answers, achieving up to 89% accuracy.

## Features

- Text preprocessing with tokenization and lemmatization
- Multiple evaluation metrics: exact match, partial match, cosine similarity, sentiment analysis
- Semantic similarity using pre-trained models
- Probabilistic analysis with Multinomial Naive Bayes
- Coherence and relevance scoring
- Weighted average scoring system
- Web interface for easy evaluation

## Technologies Used

- **Flask**: Web framework
- **Python**: Programming language
- **Jupyter Notebook**: For testing and development
- **HTML/CSS/Bootstrap**: Frontend
- **SQL**: Database
- **NLP Libraries**: NLTK, Sentence Transformers, etc.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/answer-evaluation-system.git
   cd answer-evaluation-system
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows: venv\Scripts\activate
   # On macOS/Linux: source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   python admin.py
   ```

## Usage

1. Open your browser and go to `http://127.0.0.1:5000`
2. Enter the expected answer and student's answer
3. Click "Evaluate" to get the score

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests.

## License

This project is licensed under the MIT License.
