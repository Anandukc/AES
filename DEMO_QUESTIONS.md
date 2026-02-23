# AI Answer Evaluation System - Demo Questions

## 10 Best Demo Questions (Least Error, Most Reliable)

Use these questions during your project presentation. Each question demonstrates a specific strength of the algorithm.

---

### 1. Exact Match
**Question:** What is Python?  
**Expected Answer:** A programming language  
**Student Answer:** A programming language  
**Expected Score:** 10/10  
**Feature:** Perfect match detection

---

### 2. Synonym Recognition
**Question:** What is a house?  
**Expected Answer:** A building where people live  
**Student Answer:** A structure where humans reside  
**Expected Score:** 9-10/10  
**Feature:** Detects synonyms (building=structure, people=humans, live=reside)

---

### 3. Typo Tolerance
**Question:** What is the largest ocean?  
**Expected Answer:** Pacific Ocean  
**Student Answer:** Pacifc Ocean  
**Expected Score:** 9/10  
**Feature:** Handles spelling mistakes with fuzzy matching

---

### 4. Detailed Answer Support
**Question:** What is Earth?  
**Expected Answer:** A planet  
**Student Answer:** Earth is the third planet from the sun where humans live  
**Expected Score:** 9-10/10  
**Feature:** Longer, more detailed answers are not penalized

---

### 5. Opposite Detection (Wrong Answer)
**Question:** What is hot?  
**Expected Answer:** High temperature  
**Student Answer:** Cold and freezing  
**Expected Score:** 1-2/10  
**Feature:** Detects antonyms and marks them as incorrect

---

### 6. Wrong City Detection
**Question:** What is the capital of Japan?  
**Expected Answer:** Tokyo  
**Student Answer:** Beijing  
**Expected Score:** 1-2/10  
**Feature:** Different cities in same category = wrong answer

---

### 7. Word Order Flexibility
**Question:** Where does the sun rise?  
**Expected Answer:** In the east  
**Student Answer:** The east is where sun rises  
**Expected Score:** 10/10  
**Feature:** Different word order, same meaning = correct

---

### 8. AI Semantic Understanding
**Question:** What is gravity?  
**Expected Answer:** Force that pulls objects toward Earth  
**Student Answer:** Attraction between objects with mass  
**Expected Score:** 9-10/10  
**Feature:** AI understands semantic meaning, not just keywords

---

### 9. Intelligence/Synonym Paraphrase
**Question:** Define intelligence  
**Expected Answer:** Ability to learn and think  
**Student Answer:** Capacity to understand and reason  
**Expected Score:** 9-10/10  
**Feature:** Complex synonym detection with WordNet

---

### 10. Wrong Number Detection
**Question:** What is 5 times 5?  
**Expected Answer:** 25  
**Student Answer:** 30  
**Expected Score:** 1-2/10  
**Feature:** Different numbers = wrong answer

---

## Quick Reference Table

| # | Question | Feature Demonstrated |
|---|----------|---------------------|
| 1 | What is Python? | Exact Match |
| 2 | What is a house? | Synonym Detection |
| 3 | Largest ocean? | Typo Tolerance |
| 4 | What is Earth? | Detailed Answers OK |
| 5 | What is hot? (opposite) | Antonym Penalty |
| 6 | Capital of Japan? (wrong) | City Mismatch |
| 7 | Sun rises where? | Word Order |
| 8 | What is gravity? | AI Semantic |
| 9 | Define intelligence | Synonym Paraphrase |
| 10 | 5 times 5? (wrong) | Number Mismatch |

---

## Algorithm Highlights to Mention

1. **SentenceTransformer AI Model** - Uses `paraphrase-MiniLM-L6-v2` for semantic understanding
2. **WordNet Integration** - Synonym and antonym detection from linguistic database
3. **Fuzzy Matching** - 85% similarity threshold for typo tolerance
4. **9 Evaluation Metrics** - Weighted combination for accurate scoring
5. **80% AI Weight** - Semantic similarity is the primary factor
6. **Smart Penalties** - Detects wrong cities, numbers, names, opposites

---

## Test Accuracy Achieved

- **Original Test Suite:** 94% accuracy (50 questions)
- **Extended Unseen Questions:** 86% accuracy (116 questions)
- **Average Error:** 1.31 points

---

*Good luck with your final year project presentation!*
