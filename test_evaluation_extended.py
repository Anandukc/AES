#!/usr/bin/env python3
"""
EXTENDED Test Script - 100+ UNSEEN Questions
Tests generalization of the evaluation algorithm on new data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from admin import evaluate

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

# NEW TEST CASES - NOT USED IN TUNING
extended_test_cases = {
    "1. Science Questions": [
        # Biology
        ("What is a cell?", "Basic unit of life", "basic unit of life", 10),
        ("What is a cell?", "Basic unit of life", "fundamental building block of all living organisms", 9),
        ("What is a cell?", "Basic unit of life", "something in the body", 5),
        ("What is DNA?", "Genetic material", "deoxyribonucleic acid carries genetic information", 9),
        ("What is DNA?", "Genetic material", "it stores hereditary info", 8),
        # Chemistry
        ("What is H2O?", "Water", "water molecule", 10),
        ("What is the atomic number of carbon?", "Six", "6", 9),
        ("What is an atom?", "Smallest unit of matter", "tiny particle that makes up everything", 8),
        # Physics
        ("What is velocity?", "Speed with direction", "rate of change of position", 8),
        ("What is energy?", "Ability to do work", "capacity to perform work", 9),
        ("What is friction?", "Force resisting motion", "resistance between surfaces", 8),
    ],
    
    "2. History Questions": [
        ("Who was the first US president?", "George Washington", "washington", 9),
        ("Who was the first US president?", "George Washington", "Abraham Lincoln", 1),
        ("When did WW2 end?", "1945", "nineteen forty five", 9),
        ("When did WW2 end?", "1945", "1944", 1),
        ("Who discovered America?", "Christopher Columbus", "columbus discovered it in 1492", 9),
        ("What was the Renaissance?", "Cultural rebirth in Europe", "period of artistic and intellectual revival", 9),
        ("Who built the pyramids?", "Ancient Egyptians", "egyptians thousands of years ago", 9),
        ("What caused WW1?", "Assassination of Archduke Franz Ferdinand", "the killing of archduke started it", 8),
        ("Who was Napoleon?", "French military leader", "emperor of france and military commander", 9),
        ("What was the Industrial Revolution?", "Shift to machine manufacturing", "change from hand production to machines", 9),
    ],
    
    "3. Geography Questions": [
        ("What is the largest country?", "Russia", "russian federation", 9),
        ("What is the largest country?", "Russia", "China", 1),
        ("What is the longest river?", "Nile River", "the nile in africa", 9),
        ("What is the longest river?", "Nile River", "Amazon", 2),
        ("What is the highest mountain?", "Mount Everest", "everest in the himalayas", 9),
        ("What continent is Brazil in?", "South America", "south american continent", 10),
        ("What ocean is between Europe and America?", "Atlantic Ocean", "the atlantic", 10),
        ("What is the capital of Germany?", "Berlin", "berlin city", 10),
        ("What is the capital of Germany?", "Berlin", "Munich", 1),
        ("What desert is in Africa?", "Sahara Desert", "sahara is the largest desert", 9),
    ],
    
    "4. Math Questions": [
        ("What is 5 times 5?", "25", "twenty five", 10),
        ("What is 5 times 5?", "25", "30", 1),
        ("What is the square root of 16?", "4", "four", 10),
        ("What is the square root of 16?", "4", "8", 1),
        ("What is pi approximately?", "3.14", "approximately 3.14159", 9),
        ("What is a prime number?", "Number divisible only by 1 and itself", "a number that can only be divided by one and the number itself", 9),
        ("What is 100 divided by 4?", "25", "twenty-five", 10),
        ("What is the formula for area of a circle?", "Pi times radius squared", "πr²", 9),
        ("What is an integer?", "Whole number", "a number without decimals", 8),
        ("What is 12 plus 8?", "20", "twenty", 10),
    ],
    
    "5. Literature Questions": [
        ("Who wrote Hamlet?", "William Shakespeare", "shakespeare", 9),
        ("Who wrote Hamlet?", "William Shakespeare", "Charles Dickens", 1),
        ("What is a novel?", "Long fictional story", "extended work of fiction in prose", 9),
        ("What is poetry?", "Literary work with meter and rhythm", "writing that uses rhythmic language", 8),
        ("Who wrote 1984?", "George Orwell", "orwell wrote the dystopian novel", 9),
        ("What is a metaphor?", "Comparison without like or as", "figure of speech comparing two things directly", 9),
        ("Who is the protagonist?", "Main character", "the central character in a story", 9),
        ("What is fiction?", "Imaginary stories", "made up narratives that are not real", 9),
        ("Who wrote Pride and Prejudice?", "Jane Austen", "austen", 9),
        ("What is a sonnet?", "14-line poem", "poem with fourteen lines and specific rhyme scheme", 9),
    ],
    
    "6. Technology Questions": [
        ("What is a computer?", "Electronic device for processing data", "machine that processes information", 9),
        ("What is the internet?", "Global network of computers", "worldwide system of connected networks", 9),
        ("What is software?", "Computer programs", "instructions that tell a computer what to do", 9),
        ("What is a database?", "Organized collection of data", "structured set of information stored electronically", 9),
        ("What is an algorithm?", "Step-by-step procedure", "set of instructions to solve a problem", 9),
        ("What is artificial intelligence?", "Machine that simulates human intelligence", "computer systems that mimic human thinking", 9),
        ("What is a virus in computing?", "Malicious software", "harmful program that damages computers", 8),
        ("What is cloud computing?", "Internet-based computing services", "storing and accessing data over the internet", 9),
        ("What is coding?", "Writing computer programs", "creating instructions for computers using programming languages", 9),
        ("What is a website?", "Collection of web pages", "pages on the internet accessible via browser", 9),
    ],
    
    "7. General Knowledge": [
        ("What color is the sun?", "Yellow", "it appears yellow or white", 9),
        ("What color is the sun?", "Yellow", "blue", 1),
        ("How many days in a week?", "Seven", "7 days", 10),
        ("How many months in a year?", "Twelve", "12 months", 10),
        ("What is the largest mammal?", "Blue whale", "the blue whale is the biggest", 9),
        ("What do bees make?", "Honey", "bees produce honey", 10),
        ("What is the opposite of hot?", "Cold", "cool or cold", 9),
        ("What is the opposite of hot?", "Cold", "warm", 3),
        ("How many legs does a spider have?", "Eight", "8 legs", 10),
        ("What is the boiling point of water?", "100 degrees Celsius", "100°C or 212°F", 9),
    ],
    
    "8. Completely Wrong Answers": [
        ("What is the capital of France?", "Paris", "London is the capital", 0),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci", "Picasso painted it", 1),
        ("What is 10 plus 10?", "20", "the answer is 15", 1),
        ("What language is spoken in Japan?", "Japanese", "they speak Chinese", 1),
        ("What planet do we live on?", "Earth", "we live on Mars", 0),
        ("What is ice?", "Frozen water", "ice is a type of fire", 0),
        ("How many wheels does a bicycle have?", "Two", "a bicycle has four wheels", 1),
        ("What color is grass?", "Green", "grass is purple", 0),
        ("What is the sun?", "A star", "the sun is a planet", 2),
        ("What do fish breathe with?", "Gills", "fish breathe with lungs", 1),
    ],
    
    "9. Paraphrased Answers (Same Meaning)": [
        ("What is rain?", "Water falling from clouds", "precipitation from the sky in liquid form", 9),
        ("What is a teacher?", "Person who educates students", "someone who instructs and guides learners", 9),
        ("What is sleep?", "State of rest", "period when the body and mind are inactive", 9),
        ("What is food?", "Substance eaten for nutrition", "material consumed to provide energy", 9),
        ("What is music?", "Organized sound", "arrangement of sounds in a pleasing way", 9),
        ("What is a doctor?", "Medical professional", "physician who treats patients", 9),
        ("What is exercise?", "Physical activity", "body movement for fitness and health", 9),
        ("What is a book?", "Written work", "collection of pages with text or images", 9),
        ("What is a friend?", "Person you like", "someone with whom you share a bond of affection", 9),
        ("What is happiness?", "State of being happy", "feeling of joy and contentment", 9),
    ],
    
    "10. Partial Answers": [
        ("Name three primary colors", "Red, blue, and yellow", "red and blue", 6),
        ("List the four seasons", "Spring, summer, fall, winter", "summer and winter", 5),
        ("What are the states of matter?", "Solid, liquid, gas", "solid and liquid", 6),
        ("Name two planets", "Mercury and Venus", "just earth", 4),
        ("What are vowels?", "A, E, I, O, U", "a and e", 5),
        ("List three continents", "Asia, Africa, Europe", "Asia", 4),
        ("Name parts of a plant", "Roots, stem, leaves, flower", "leaves and roots", 6),
        ("What are the five senses?", "Sight, hearing, taste, smell, touch", "seeing and hearing", 5),
        ("List cardinal directions", "North, south, east, west", "north and south", 5),
        ("Name three oceans", "Pacific, Atlantic, Indian", "Pacific Ocean", 4),
    ],
    
    "11. Opposite Meanings (Should Score Low)": [
        ("What is love?", "Affection and care", "hatred and anger", 1),
        ("What is success?", "Achievement of goals", "failure to accomplish anything", 1),
        ("What does brave mean?", "Courageous", "fearful and cowardly", 1),
        ("What is truth?", "Accurate facts", "lies and deception", 1),
        ("What is health?", "State of being well", "sickness and disease", 1),
        ("What is peace?", "Absence of conflict", "war and violence", 1),
        ("What is wealth?", "Having money", "poverty and having nothing", 1),
        ("What is knowledge?", "Understanding information", "ignorance and not knowing", 1),
        ("What is safe?", "Free from danger", "dangerous and risky", 1),
        ("What is clean?", "Free from dirt", "dirty and filthy", 1),
    ],
    
    "12. Detailed/Extended Answers": [
        ("What is a car?", "Vehicle", "A car is a motor vehicle with four wheels used for transportation of people", 9),
        ("What is a tree?", "Plant", "Trees are tall plants with wooden trunks, branches, and leaves that produce oxygen", 9),
        ("What is water?", "Liquid", "Water is a transparent liquid essential for all life forms, chemically known as H2O", 9),
        ("What is a phone?", "Communication device", "A phone is an electronic device used to make calls, send messages, and access the internet", 9),
        ("What is school?", "Place of learning", "School is an educational institution where students are taught by teachers", 9),
    ],
}


def run_test_case(question, expected_answer, student_answer, expected_score):
    actual_score = evaluate(expected_answer, student_answer)
    diff = abs(actual_score - expected_score)
    
    if diff == 0:
        status = "PERFECT"
        color = Colors.GREEN
    elif diff <= 1:
        status = "GOOD"
        color = Colors.CYAN
    elif diff <= 2:
        status = "ACCEPTABLE"
        color = Colors.YELLOW
    else:
        status = "NEEDS WORK"
        color = Colors.RED
    
    return {
        'question': question,
        'expected_answer': expected_answer,
        'student_answer': student_answer,
        'expected_score': expected_score,
        'actual_score': actual_score,
        'difference': diff,
        'status': status,
        'color': color
    }


def run_all_tests():
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}EXTENDED TEST SUITE - 100+ UNSEEN QUESTIONS{Colors.END}")
    print(f"{Colors.BOLD}{Colors.YELLOW}Testing GENERALIZATION (not the tuned questions){Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    all_results = {}
    total_tests = 0
    perfect_count = 0
    good_count = 0
    acceptable_count = 0
    needs_work_count = 0
    total_diff = 0
    
    for category, cases in extended_test_cases.items():
        print(f"\n{Colors.BOLD}{Colors.CYAN}{category}{Colors.END}")
        print(f"{'-'*80}")
        
        category_results = []
        
        for i, (question, expected_ans, student_ans, expected_score) in enumerate(cases, 1):
            result = run_test_case(question, expected_ans, student_ans, expected_score)
            category_results.append(result)
            total_tests += 1
            total_diff += result['difference']
            
            if result['status'] == "PERFECT":
                perfect_count += 1
            elif result['status'] == "GOOD":
                good_count += 1
            elif result['status'] == "ACCEPTABLE":
                acceptable_count += 1
            else:
                needs_work_count += 1
            
            status_text = f"{result['color']}{result['status']}{Colors.END}"
            print(f"  {i}. Q: {question[:40]}...")
            print(f"     Exp: {expected_score} | Got: {result['actual_score']} | Diff: {result['difference']} | {status_text}")
        
        all_results[category] = category_results
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}EXTENDED TEST SUMMARY (GENERALIZATION TEST){Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    print(f"Total Test Cases: {total_tests}")
    print(f"{Colors.GREEN}Perfect (0 diff):    {perfect_count} ({perfect_count/total_tests*100:.1f}%){Colors.END}")
    print(f"{Colors.CYAN}Good (1 diff):       {good_count} ({good_count/total_tests*100:.1f}%){Colors.END}")
    print(f"{Colors.YELLOW}Acceptable (2 diff): {acceptable_count} ({acceptable_count/total_tests*100:.1f}%){Colors.END}")
    print(f"{Colors.RED}Needs Work (3+ diff): {needs_work_count} ({needs_work_count/total_tests*100:.1f}%){Colors.END}")
    print(f"\nAverage Difference: {total_diff/total_tests:.2f} points")
    
    acceptable_total = perfect_count + good_count + acceptable_count
    accuracy = (acceptable_total / total_tests) * 100
    
    print(f"\n{Colors.BOLD}GENERALIZATION ACCURACY: {accuracy:.1f}%{Colors.END}")
    
    if accuracy >= 90:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ EXCELLENT! Algorithm generalizes very well to new questions!{Colors.END}")
    elif accuracy >= 75:
        print(f"{Colors.CYAN}{Colors.BOLD}✓ GOOD! Algorithm generalizes reasonably well.{Colors.END}")
    elif accuracy >= 60:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ MODERATE! Some overfitting detected. May need more generic rules.{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ POOR! Significant overfitting. Algorithm too specialized.{Colors.END}")
    
    # Problem cases
    print(f"\n{Colors.BOLD}PROBLEM AREAS (Difference >= 3):{Colors.END}")
    problem_count = 0
    for category, results in all_results.items():
        problems = [r for r in results if r['difference'] >= 3]
        if problems:
            print(f"\n{Colors.YELLOW}{category}:{Colors.END}")
            for p in problems:
                problem_count += 1
                print(f"  • Exp: {p['expected_score']}, Got: {p['actual_score']} (diff: {p['difference']})")
                print(f"    Q: {p['question'][:50]}...")
                print(f"    Expected: {p['expected_answer'][:40]}...")
                print(f"    Student:  {p['student_answer'][:40]}...")
    
    if problem_count == 0:
        print(f"{Colors.GREEN}  None! All tests within acceptable range.{Colors.END}")
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    return accuracy


if __name__ == "__main__":
    try:
        accuracy = run_all_tests()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted.{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error: {e}{Colors.END}\n")
        import traceback
        traceback.print_exc()
