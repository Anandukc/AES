#!/usr/bin/env python3
"""
Automated Test Script for AI Answer Evaluation System
Tests the evaluate() function with predefined test cases
"""

import sys
import os

# Add current directory to path to import admin module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import evaluation function from admin.py
from admin import evaluate

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

# Test cases organized by category
test_cases = {
    "1. Exact & Near-Exact Matches": [
        ("What is Python?", "Python is a programming language", "python is a programming language", 10),
        ("Who invented the telephone?", "Alexander Graham Bell", "alexander graham bell", 10),
        ("What is the capital of France?", "Paris", "Paris", 10),
        ("What color is the sky?", "The sky is blue", "sky is blue", 10),
        ("Name a fruit", "Apple", "apple", 10),
    ],
    
    "2. Paraphrased Answers (Same Meaning)": [
        ("What does HTML stand for?", "HyperText Markup Language", "Markup language for hypertext", 9),
        ("What is photosynthesis?", "Plants make food using sunlight", "Process where plants use sunlight to create food", 9),
        ("Who is the president of USA?", "Joe Biden is the president", "President of USA is Joe Biden", 10),
        ("What is gravity?", "Force that pulls objects down", "A force pulling things toward Earth", 9),
        ("Define democracy", "Government by the people", "System where people govern themselves", 9),
    ],
    
    "3. Partially Correct Answers": [
        ("List three colors", "Red, blue, and green", "Red and blue", 6),
        ("What causes rain?", "Water evaporates and condenses in clouds", "Water comes from clouds", 6),
        ("Who wrote Romeo and Juliet?", "William Shakespeare wrote it", "Shakespeare", 7),
        ("What is the solar system?", "Sun and planets orbiting it", "Planets around the sun", 7),
        ("Explain DNA", "Genetic material in living things", "Something in our body", 5),
    ],
    
    "4. Minor Errors (Typos)": [
        ("What is a computer?", "Electronic device for processing data", "Electronik device for processing data", 8),
        ("Name the largest ocean", "Pacific Ocean", "Pasific Ocean", 9),
        ("What is water made of?", "Hydrogen and oxygen", "Hidrogen and oxygen", 9),
        ("Who painted Mona Lisa?", "Leonardo da Vinci", "Lionardo da Vinci", 8),
        ("What is the speed of light?", "300000 kilometers per second", "300000 kilometer per second", 10),
    ],
    
    "5. Word Order Changes": [
        ("What color is grass?", "Grass is green in color", "Green is the color of grass", 9),
        ("Where does the sun rise?", "The sun rises in the east", "In the east the sun rises", 10),
        ("What is milk?", "White liquid from cows", "Liquid white from cows", 9),
        ("Describe a cat", "Small furry animal", "Furry small animal", 10),
        ("What is music?", "Art form using sound", "Sound using art form", 8),
    ],
    
    "6. Wrong or Irrelevant Answers": [
        ("What is the capital of Japan?", "Tokyo", "Paris", 0),
        ("What is 2+2?", "Four", "Seven", 0),
        ("Who invented the light bulb?", "Thomas Edison", "Albert Einstein", 1),
        ("What is a lion?", "Large carnivorous cat", "A type of bird", 0),
        ("Name a vegetable", "Carrot", "Strawberry", 0),
    ],
    
    "7. Extra Information": [
        ("What is Earth?", "A planet", "Earth is a planet in the solar system that supports life", 9),
        ("Define a tree", "Plant with trunk and branches", "A tree is a tall plant with a woody trunk and branches that produces oxygen", 9),
        ("What is snow?", "Frozen water", "Snow is frozen water that falls from clouds in winter", 9),
        ("Who is Einstein?", "Famous physicist", "Einstein was a famous physicist who developed the theory of relativity", 10),
        ("What is reading?", "Looking at text", "Reading is the process of looking at written text and understanding its meaning", 9),
    ],
    
    "8. Concatenated Words & Formatting": [
        ("What is a computer?", "Electronic device", "Electronicdevice", 9),
        ("Name a country", "United States", "UnitedStates", 9),
        ("What is water?", "H2O liquid", "H2Oliquid", 9),
        ("Describe a dog", "Four legged animal", "Fourlegged animal", 10),
        ("What is air?", "Mixture of gases", "Mixtureof gases", 9),
    ],
    
    "9. Synonym Usage": [
        ("What is a house?", "Building where people live", "Structure where humans reside", 9),
        ("Describe happiness", "Feeling of joy", "Emotion of delight", 9),
        ("What is a car?", "Vehicle for transportation", "Automobile for travel", 9),
        ("Define intelligence", "Ability to learn and think", "Capacity to understand and reason", 9),
        ("What is warm?", "Having high temperature", "Being hot", 8),
    ],
    
    "10. Opposites": [
        ("What is hot?", "High temperature", "Low temperature", 2),
        ("Define day", "Period of sunlight", "Period of darkness", 2),
        ("What goes up?", "Rises or ascends", "Falls or descends", 2),
        ("Describe fast", "Moving quickly", "Moving slowly", 2),
        ("What is big?", "Large in size", "Small in size", 2),
    ],
}


def run_test_case(question, expected_answer, student_answer, expected_score):
    """Run a single test case and return results"""
    actual_score = evaluate(expected_answer, student_answer)
    
    # Calculate difference
    diff = abs(actual_score - expected_score)
    
    # Determine status
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
    """Run all test cases and generate report"""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}AI ANSWER EVALUATION SYSTEM - AUTOMATED TEST SUITE{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    all_results = {}
    total_tests = 0
    perfect_count = 0
    good_count = 0
    acceptable_count = 0
    needs_work_count = 0
    total_diff = 0
    
    # Run tests for each category
    for category, cases in test_cases.items():
        print(f"\n{Colors.BOLD}{Colors.CYAN}{category}{Colors.END}")
        print(f"{'-'*80}")
        
        category_results = []
        
        for i, (question, expected_ans, student_ans, expected_score) in enumerate(cases, 1):
            result = run_test_case(question, expected_ans, student_ans, expected_score)
            category_results.append(result)
            total_tests += 1
            total_diff += result['difference']
            
            # Count by status
            if result['status'] == "PERFECT":
                perfect_count += 1
            elif result['status'] == "GOOD":
                good_count += 1
            elif result['status'] == "ACCEPTABLE":
                acceptable_count += 1
            else:
                needs_work_count += 1
            
            # Print result
            status_text = f"{result['color']}{result['status']}{Colors.END}"
            print(f"  {i}. Q: {question[:45]}...")
            print(f"     Expected Score: {expected_score} | Actual Score: {result['actual_score']} | "
                  f"Diff: {result['difference']} | {status_text}")
        
        all_results[category] = category_results
    
    # Print summary
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}TEST SUMMARY{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    print(f"Total Test Cases: {total_tests}")
    print(f"{Colors.GREEN}Perfect (0 diff):    {perfect_count} ({perfect_count/total_tests*100:.1f}%){Colors.END}")
    print(f"{Colors.CYAN}Good (1 diff):       {good_count} ({good_count/total_tests*100:.1f}%){Colors.END}")
    print(f"{Colors.YELLOW}Acceptable (2 diff): {acceptable_count} ({acceptable_count/total_tests*100:.1f}%){Colors.END}")
    print(f"{Colors.RED}Needs Work (3+ diff): {needs_work_count} ({needs_work_count/total_tests*100:.1f}%){Colors.END}")
    print(f"\nAverage Difference: {total_diff/total_tests:.2f} points")
    
    # Calculate accuracy (within 2 points is acceptable)
    acceptable_total = perfect_count + good_count + acceptable_count
    accuracy = (acceptable_total / total_tests) * 100
    
    print(f"\n{Colors.BOLD}Overall Accuracy (within 2 points): {accuracy:.1f}%{Colors.END}")
    
    if accuracy >= 90:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ EXCELLENT! System is working very well!{Colors.END}")
    elif accuracy >= 75:
        print(f"{Colors.CYAN}{Colors.BOLD}✓ GOOD! Minor tuning may improve results.{Colors.END}")
    elif accuracy >= 60:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠ ACCEPTABLE! Significant improvements needed.{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ NEEDS WORK! Major adjustments required.{Colors.END}")
    
    # Detailed problem areas
    print(f"\n{Colors.BOLD}PROBLEM AREAS (Difference >= 3):{Colors.END}")
    problem_count = 0
    for category, results in all_results.items():
        problems = [r for r in results if r['difference'] >= 3]
        if problems:
            print(f"\n{Colors.YELLOW}{category}:{Colors.END}")
            for p in problems:
                problem_count += 1
                print(f"  • Expected: {p['expected_score']}, Got: {p['actual_score']} (diff: {p['difference']})")
                print(f"    Q: {p['question']}")
                print(f"    Expected Ans: {p['expected_answer'][:50]}...")
                print(f"    Student Ans:  {p['student_answer'][:50]}...")
    
    if problem_count == 0:
        print(f"{Colors.GREEN}  None! All tests within acceptable range.{Colors.END}")
    
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    return all_results


def run_quick_test():
    """Run only priority test cases for quick feedback"""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}QUICK TEST - Priority Cases Only{Colors.END}")
    print(f"{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    priority_cases = [
        ("What is Python?", "Python is a programming language", "python is a programming language", 10),
        ("Who is the president of USA?", "Joe Biden is the president", "President of USA is Joe Biden", 10),
        ("List three colors", "Red, blue, and green", "Red and blue", 6),
        ("Name the largest ocean", "Pacific Ocean", "Pasific Ocean", 9),
        ("What is the capital of Japan?", "Tokyo", "Paris", 0),
        ("What is Earth?", "A planet", "Earth is a planet in the solar system that supports life", 9),
        ("Name a country", "United States", "UnitedStates", 9),
        ("What is a house?", "Building where people live", "Structure where humans reside", 9),
    ]
    
    results = []
    for i, (q, exp, stu, expected) in enumerate(priority_cases, 1):
        result = run_test_case(q, exp, stu, expected)
        results.append(result)
        
        status_text = f"{result['color']}{result['status']}{Colors.END}"
        print(f"{i}. Expected: {expected} | Actual: {result['actual_score']} | {status_text}")
        print(f"   Q: {q}")
        print(f"   Student: {stu[:60]}...\n")
    
    # Quick summary
    perfect = sum(1 for r in results if r['difference'] == 0)
    acceptable = sum(1 for r in results if r['difference'] <= 2)
    
    print(f"{Colors.BOLD}Quick Summary:{Colors.END}")
    print(f"Perfect: {perfect}/{len(results)}")
    print(f"Acceptable (≤2 diff): {acceptable}/{len(results)}")
    print(f"Accuracy: {acceptable/len(results)*100:.1f}%\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AI Answer Evaluation System")
    parser.add_argument('--quick', '-q', action='store_true', 
                       help='Run only quick priority tests (8 cases)')
    parser.add_argument('--full', '-f', action='store_true', 
                       help='Run full test suite (50 cases)')
    parser.add_argument('--category', '-c', type=int, 
                       help='Run specific category (1-10)')
    
    args = parser.parse_args()
    
    try:
        if args.quick:
            run_quick_test()
        elif args.category:
            # Run specific category
            categories = list(test_cases.keys())
            if 1 <= args.category <= len(categories):
                cat_name = categories[args.category - 1]
                print(f"\nTesting Category: {cat_name}\n")
                for i, (q, exp, stu, expected) in enumerate(test_cases[cat_name], 1):
                    result = run_test_case(q, exp, stu, expected)
                    print(f"{i}. Expected: {expected} | Actual: {result['actual_score']} | "
                          f"{result['color']}{result['status']}{Colors.END}")
            else:
                print(f"Invalid category. Choose 1-{len(categories)}")
        else:
            # Default: run full test suite
            run_all_tests()
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Test interrupted by user.{Colors.END}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error during testing: {e}{Colors.END}\n")
        import traceback
        traceback.print_exc()
