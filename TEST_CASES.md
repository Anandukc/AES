# AI Answer Evaluation System - Test Cases

## Instructions
Use these test cases to evaluate the scoring system. Copy questions and answers into your system, then record the actual scores you receive.

---

## Test Case Set 1: Exact & Near-Exact Matches
**Expected Score: 9-10/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 1.1 | What is Python? | Python is a programming language | python is a programming language | 10 | | Case difference only |
| 1.2 | Who invented the telephone? | Alexander Graham Bell | alexander graham bell | 10 | | Case difference only |
| 1.3 | What is the capital of France? | Paris | Paris | 10 | | Exact match |
| 1.4 | What color is the sky? | The sky is blue | sky is blue | 10 | | Article difference |
| 1.5 | Name a fruit | Apple | apple | 10 | | Case only |

---

## Test Case Set 2: Paraphrased Answers (Same Meaning)
**Expected Score: 8-10/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 2.1 | What does HTML stand for? | HyperText Markup Language | Markup language for hypertext | 9 | | Reordered words |
| 2.2 | What is photosynthesis? | Plants make food using sunlight | Process where plants use sunlight to create food | 9 | | More detailed |
| 2.3 | Who is the president of USA? | Joe Biden is the president | President of USA is Joe Biden | 10 | | Word order change |
| 2.4 | What is gravity? | Force that pulls objects down | A force pulling things toward Earth | 9 | | Similar meaning |
| 2.5 | Define democracy | Government by the people | System where people govern themselves | 9 | | Paraphrased |

---

## Test Case Set 3: Partially Correct Answers
**Expected Score: 5-7/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 3.1 | List three colors | Red, blue, and green | Red and blue | 6 | | Missing one item |
| 3.2 | What causes rain? | Water evaporates and condenses in clouds | Water comes from clouds | 6 | | Incomplete |
| 3.3 | Who wrote Romeo and Juliet? | William Shakespeare wrote it | Shakespeare | 7 | | Missing first name |
| 3.4 | What is the solar system? | Sun and planets orbiting it | Planets around the sun | 7 | | Simplified |
| 3.5 | Explain DNA | Genetic material in living things | Something in our body | 5 | | Very vague |

---

## Test Case Set 4: Minor Errors (Typos, Small Mistakes)
**Expected Score: 7-9/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 4.1 | What is a computer? | Electronic device for processing data | Electronik device for processing data | 8 | | Typo: electronik |
| 4.2 | Name the largest ocean | Pacific Ocean | Pasific Ocean | 9 | | Typo: Pasific |
| 4.3 | What is water made of? | Hydrogen and oxygen | Hidrogen and oxygen | 9 | | Typo: Hidrogen |
| 4.4 | Who painted Mona Lisa? | Leonardo da Vinci | Lionardo da Vinci | 8 | | Typo: Lionardo |
| 4.5 | What is the speed of light? | 300000 kilometers per second | 300000 kilometer per second | 10 | | Singular vs plural |

---

## Test Case Set 5: Word Order Changes (Same Words, Different Order)
**Expected Score: 9-10/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 5.1 | What color is grass? | Grass is green in color | Green is the color of grass | 9 | | Completely reordered |
| 5.2 | Where does the sun rise? | The sun rises in the east | In the east the sun rises | 10 | | Different order |
| 5.3 | What is milk? | White liquid from cows | Liquid white from cows | 9 | | Adjective reordered |
| 5.4 | Describe a cat | Small furry animal | Furry small animal | 10 | | Word swap |
| 5.5 | What is music? | Art form using sound | Sound using art form | 8 | | Changes meaning slightly |

---

## Test Case Set 6: Wrong or Irrelevant Answers
**Expected Score: 0-3/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 6.1 | What is the capital of Japan? | Tokyo | Paris | 0 | | Completely wrong |
| 6.2 | What is 2+2? | Four | Seven | 0 | | Wrong answer |
| 6.3 | Who invented the light bulb? | Thomas Edison | Albert Einstein | 1 | | Wrong person |
| 6.4 | What is a lion? | Large carnivorous cat | A type of bird | 0 | | Completely wrong |
| 6.5 | Name a vegetable | Carrot | Strawberry | 0 | | Wrong category |

---

## Test Case Set 7: Extra Information (More Than Expected)
**Expected Score: 8-10/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 7.1 | What is Earth? | A planet | Earth is a planet in the solar system that supports life | 9 | | Extra detail |
| 7.2 | Define a tree | Plant with trunk and branches | A tree is a tall plant with a woody trunk and branches that produces oxygen | 9 | | Much more detail |
| 7.3 | What is snow? | Frozen water | Snow is frozen water that falls from clouds in winter | 9 | | Added context |
| 7.4 | Who is Einstein? | Famous physicist | Einstein was a famous physicist who developed the theory of relativity | 10 | | Extra accurate info |
| 7.5 | What is reading? | Looking at text | Reading is the process of looking at written text and understanding its meaning | 9 | | Elaborated |

---

## Test Case Set 8: Concatenated Words & Formatting Issues
**Expected Score: 9-10/10** (Testing clean_text function)

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 8.1 | What is a computer? | Electronic device | Electronicdevice | 9 | | No space |
| 8.2 | Name a country | United States | UnitedStates | 9 | | Concatenated |
| 8.3 | What is water? | H2O liquid | H2Oliquid | 9 | | No space |
| 8.4 | Describe a dog | Fourlegged animal | Four legged animal | 10 | | Space added |
| 8.5 | What is air? | Mixture of gases | Mixtureof gases | 9 | | Missing space |

---

## Test Case Set 9: Synonym Usage
**Expected Score: 8-10/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 9.1 | What is a house? | Building where people live | Structure where humans reside | 9 | | Synonyms |
| 9.2 | Describe happiness | Feeling of joy | Emotion of delight | 9 | | Synonyms |
| 9.3 | What is a car? | Vehicle for transportation | Automobile for travel | 9 | | Synonyms |
| 9.4 | Define intelligence | Ability to learn and think | Capacity to understand and reason | 9 | | Synonyms |
| 9.5 | What is warm? | Having high temperature | Being hot | 8 | | Similar concept |

---

## Test Case Set 10: Opposite but Related
**Expected Score: 3-5/10**

| # | Question | Expected Answer | Student Answer | Expected Score | Actual Score | Notes |
|---|----------|----------------|----------------|----------------|--------------|-------|
| 10.1 | What is hot? | High temperature | Low temperature | 2 | | Opposite |
| 10.2 | Define day | Period of sunlight | Period of darkness | 2 | | Opposite (night) |
| 10.3 | What goes up? | Rises or ascends | Falls or descends | 2 | | Opposite |
| 10.4 | Describe fast | Moving quickly | Moving slowly | 2 | | Opposite |
| 10.5 | What is big? | Large in size | Small in size | 2 | | Opposite |

---

## Quick Test Cases (for rapid testing)

### Copy-Paste Format:

**Question:** What is Python?  
**Expected:** Python is a programming language  
**Student:** python is a programming language  
**Expected Score:** 10

---

**Question:** What is photosynthesis?  
**Expected:** Plants make food using sunlight  
**Student:** Process where plants use sunlight to create food  
**Expected Score:** 9

---

**Question:** Who is Einstein?  
**Expected:** Famous physicist  
**Student:** Einstein was a scientist  
**Expected Score:** 7-8

---

**Question:** What is 2+2?  
**Expected:** Four  
**Student:** Seven  
**Expected Score:** 0

---

**Question:** What is the capital of Japan?  
**Expected:** Tokyo  
**Student:** Tokyo is the capital  
**Expected Score:** 10

---

## Scoring Interpretation Guide

| Score Range | Interpretation |
|-------------|----------------|
| 10 | Perfect match (meaning-wise) |
| 8-9 | Very good, minor differences |
| 6-7 | Partially correct, missing info |
| 4-5 | Some relevance, mostly wrong |
| 2-3 | Wrong but related concept |
| 0-1 | Completely wrong |

---

## How to Use This Document

1. **Create a test** in your teacher account with questions from one category
2. **Add expected answers** from the "Expected Answer" column
3. **Take the test** as a student using answers from "Student Answer" column
4. **View scores** and record them in the "Actual Score" column
5. **Compare** actual scores with expected scores
6. **Report findings** - which categories work well, which need improvement

---

## Feedback Template

When reporting results, use this format:

```
Test Set: [Number and Name]
Cases Tested: [X out of Y]

Results:
- Case X.Y: Expected [score], Got [score] - [OK/Too High/Too Low]
- Case X.Y: Expected [score], Got [score] - [OK/Too High/Too Low]

Overall: [Good/Needs Improvement]
Specific Issues: [Describe any patterns or problems]
```

---

## Priority Test Cases (Start Here!)

If you have limited time, test these critical cases first:

1. **Case 1.1** - Case sensitivity test
2. **Case 2.3** - Word order test  
3. **Case 3.2** - Partial correctness test
4. **Case 4.2** - Typo tolerance test
5. **Case 6.1** - Wrong answer detection
6. **Case 7.1** - Extra information handling
7. **Case 8.2** - Concatenated words test
8. **Case 9.1** - Synonym recognition

---

**Last Updated:** February 17, 2026
