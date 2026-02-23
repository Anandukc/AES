# src/models.py
from .extensions import mysql
from collections import defaultdict

# ---------- Admin related ----------
def get_all_students():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Students")
    students = cur.fetchall()
    cur.close()
    return students

def add_student(username, password):
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO Students (username, password) VALUES (%s, %s)", (username, password))
    mysql.connection.commit()
    cur.close()

def update_student(student_id, username, password):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE Students SET username = %s, password = %s WHERE student_id = %s", (username, password, student_id))
    mysql.connection.commit()
    cur.close()

def delete_student(student_id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM Students WHERE student_id = %s", (student_id,))
    mysql.connection.commit()
    cur.close()

def get_student_scores(student_id):
    cur = mysql.connection.cursor()
    query = """
        SELECT DISTINCT sa.answer_id, sa.test_id, t.test_name, q.question_id,
            q.question_text, ea.answer_text AS expected_answer, 
            sa.answer_text AS student_answer, sa.score
        FROM studentanswers sa
        JOIN tests t ON sa.test_id = t.test_id
        JOIN questions q ON sa.question_id = q.question_id
        JOIN expectedanswers ea ON q.question_id = ea.question_id
        WHERE sa.student_id = %s
        ORDER BY sa.test_id, q.question_id;
    """
    cur.execute(query, (student_id,))
    scores = cur.fetchall()
    cur.close()
    return scores

def delete_student_score(answer_id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM studentanswers WHERE answer_id = %s", (answer_id,))
    mysql.connection.commit()
    cur.close()

# ---------- Teacher related ----------
def get_all_teachers():
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Teachers")
    teachers = cur.fetchall()
    cur.close()
    return teachers

def add_teacher(username, password):
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO Teachers (username, password) VALUES (%s, %s)", (username, password))
    mysql.connection.commit()
    cur.close()

def update_teacher(teacher_id, username, password):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE Teachers SET username = %s, password = %s WHERE teacher_id = %s", (username, password, teacher_id))
    mysql.connection.commit()
    cur.close()

def delete_teacher(teacher_id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM teacherstudentrelationship WHERE teacher_id = %s", (teacher_id,))
    cur.execute("DELETE FROM teachers WHERE teacher_id = %s", (teacher_id,))
    mysql.connection.commit()
    cur.close()

def get_teacher_tests(teacher_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Tests WHERE teacher_id = %s", (teacher_id,))
    tests = cur.fetchall()
    cur.close()
    return tests

def get_test_questions(test_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Questions WHERE test_id = %s", (test_id,))
    questions = cur.fetchall()
    cur.close()
    return questions

def get_expected_answers_for_question(question_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM ExpectedAnswers WHERE question_id = %s", (question_id,))
    answers = cur.fetchall()
    cur.close()
    return answers

def add_question_to_test(test_id, question_text, expected_answers):
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO Questions (question_text, test_id) VALUES (%s, %s)", (question_text, test_id))
    question_id = cur.lastrowid
    for answer in expected_answers:
        cur.execute("INSERT INTO ExpectedAnswers (answer_text, question_id) VALUES (%s, %s)", (answer, question_id))
    mysql.connection.commit()
    cur.close()

def delete_question(question_id):
    cur = mysql.connection.cursor()
    cur.execute("DELETE FROM ExpectedAnswers WHERE question_id = %s", (question_id,))
    cur.execute("DELETE FROM Questions WHERE question_id = %s", (question_id,))
    mysql.connection.commit()
    cur.close()

def add_test(test_name, teacher_id):
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO Tests (test_name, teacher_id) VALUES (%s, %s)", (test_name, teacher_id))
    mysql.connection.commit()
    cur.close()

def update_test_name(test_id, new_name):
    cur = mysql.connection.cursor()
    cur.execute("UPDATE Tests SET test_name = %s WHERE test_id = %s", (new_name, test_id))
    mysql.connection.commit()
    cur.close()

def delete_test(test_id):
    cur = mysql.connection.cursor()
    # Delete related student answers first
    cur.execute("DELETE FROM studentanswers WHERE test_id = %s", (test_id,))
    # Delete expected answers
    cur.execute("DELETE FROM expectedanswers WHERE question_id IN (SELECT question_id FROM questions WHERE test_id = %s)", (test_id,))
    # Delete questions
    cur.execute("DELETE FROM questions WHERE test_id = %s", (test_id,))
    # Delete test
    cur.execute("DELETE FROM tests WHERE test_id = %s", (test_id,))
    mysql.connection.commit()
    cur.close()

def get_teacher_scores(teacher_id):
    cur = mysql.connection.cursor()
    query = """
        SELECT s.student_id, s.username AS student_username, t.test_name, q.question_text, ea.answer_text AS expected_answer, sa.answer_text AS student_answer, sa.score
        FROM StudentAnswers sa
        JOIN Students s ON sa.student_id = s.student_id
        JOIN Tests t ON sa.test_id = t.test_id
        JOIN Questions q ON sa.question_id = q.question_id
        JOIN ExpectedAnswers ea ON q.question_id = ea.question_id
        WHERE t.teacher_id = %s
    """
    cur.execute(query, (teacher_id,))
    results = cur.fetchall()
    cur.close()
    return results

# ---------- Student related ----------
def get_student_by_credentials(username, password):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Students WHERE username = %s AND password = %s", (username, password))
    student = cur.fetchone()
    cur.close()
    return student

def get_available_tests_for_student(student_id):
    cur = mysql.connection.cursor()
    cur.execute("""
        SELECT t.test_id, t.test_name 
        FROM Tests t 
        LEFT JOIN StudentAnswers sa ON t.test_id = sa.test_id AND sa.student_id = %s
        WHERE sa.test_id IS NULL
    """, (student_id,))
    tests = cur.fetchall()
    cur.close()
    return tests

def get_test_by_id(test_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Tests WHERE test_id = %s", (test_id,))
    test = cur.fetchone()
    cur.close()
    return test

def get_questions_for_test(test_id):
    cur = mysql.connection.cursor()
    cur.execute("SELECT * FROM Questions WHERE test_id = %s", (test_id,))
    questions = cur.fetchall()
    cur.close()
    return questions

def insert_student_answer(student_id, test_id, question_id, answer_text):
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO StudentAnswers (student_id, test_id, question_id, answer_text) VALUES (%s, %s, %s, %s)",
                (student_id, test_id, question_id, answer_text))
    mysql.connection.commit()
    cur.close()

def get_student_answers_with_expected(student_id):
    cur = mysql.connection.cursor()
    query = """
        SELECT t.test_id, t.test_name, q.question_text, ea.answer_text AS expected_answer, sa.answer_text AS student_answer
        FROM StudentAnswers sa
        JOIN Tests t ON sa.test_id = t.test_id
        JOIN Questions q ON sa.question_id = q.question_id
        JOIN ExpectedAnswers ea ON q.question_id = ea.question_id
        WHERE sa.student_id = %s
    """
    cur.execute(query, (student_id,))
    results = cur.fetchall()
    cur.close()
    return results

def update_student_score(student_id, test_id, question_text, score):
    cur = mysql.connection.cursor()
    # This is a bit tricky because question_text might not be unique; better to use question_id.
    # For simplicity, we'll assume question_text is unique per test. A better approach is to pass question_id.
    # In your original code you used a subquery: 
    # "UPDATE studentanswers SET score = %s WHERE student_id = %s AND test_id = %s AND question_id IN (SELECT question_id FROM questions WHERE question_text = %s)"
    # We'll keep that.
    cur.execute("""
        UPDATE studentanswers SET score = %s 
        WHERE student_id = %s AND test_id = %s AND question_id IN (
            SELECT question_id FROM questions WHERE question_text = %s AND test_id = %s
        )
    """, (score, student_id, test_id, question_text, test_id))
    mysql.connection.commit()
    cur.close()