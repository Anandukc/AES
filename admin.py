# admin.py
from flask import Flask, render_template, request, redirect, url_for, session
from collections import defaultdict

from src.config import Config
from src.extensions import mysql
from src import models, evaluation

app = Flask(__name__)
app.config.from_object(Config)
app.template_folder = 'templates'

# Disable caching to prevent stale templates
@app.after_request
def set_response_headers(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

# Initialize MySQL
mysql.init_app(app)

# -------------------------------------------------------------------
# Homepage
# -------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('Homepage.html')

# -------------------------------------------------------------------
# Admin routes
# -------------------------------------------------------------------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Admins WHERE username = %s AND password = %s", (username, password))
        admin = cur.fetchone()
        cur.close()

        if admin:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_home'))
        else:
            return render_template('adminlogin.html', error='Invalid username or password')

    return render_template('adminlogin.html')

@app.route('/admin/home')
def admin_home():
    if 'admin_logged_in' in session:
        return render_template('adminhome.html')
    return redirect(url_for('admin_login'))

@app.route('/admin/students')
def admin_students():
    if 'admin_logged_in' in session:
        students = models.get_all_students()
        return render_template('admin_students.html', students=students)
    return redirect(url_for('admin_login'))

@app.route('/admin/add_student', methods=['POST'])
def add_student():
    if 'admin_logged_in' in session:
        username = request.form['username']
        password = request.form['password']
        models.add_student(username, password)
        return redirect(url_for('admin_students'))
    return redirect(url_for('admin_login'))

@app.route('/admin/update_student/<int:student_id>', methods=['POST'])
def update_student(student_id):
    if 'admin_logged_in' in session:
        username = request.form['username']
        password = request.form['password']
        models.update_student(student_id, username, password)
        return redirect(url_for('admin_students'))
    return redirect(url_for('admin_login'))

@app.route('/admin/delete_student/<int:student_id>', methods=['POST'])
def delete_student(student_id):
    if 'admin_logged_in' in session:
        models.delete_student(student_id)
        return redirect(url_for('admin_students'))
    return redirect(url_for('admin_login'))

@app.route('/admin/view_student_scores/<int:student_id>')
def view_student_scores(student_id):
    if 'admin_logged_in' in session:
        scores = models.get_student_scores(student_id)
        # Convert tuples to dictionaries for easier template handling
        scores = [
            {
                'answer_id': s[0],
                'test_id': s[1],
                'test_name': s[2],
                'question_text': s[4],
                'expected_answer': s[5],
                'student_answer': s[6],
                'score': s[7]
            }
            for s in scores
        ]
        return render_template('student_scores.html', scores=scores)
    return redirect(url_for('admin_login'))

@app.route('/admin/delete_student_score/<int:answer_id>', methods=['POST'])
def delete_student_score(answer_id):
    if 'admin_logged_in' in session:
        models.delete_student_score(answer_id)
        return redirect(url_for('admin_students'))
    return redirect(url_for('admin_login'))

# Admin teacher management
@app.route('/admin/teachers')
def admin_teachers():
    if 'admin_logged_in' in session:
        teachers = models.get_all_teachers()
        return render_template('admin_teachers.html', teachers=teachers)
    return redirect(url_for('admin_login'))

@app.route('/admin/add_teacher', methods=['GET', 'POST'])
def add_teacher():
    if 'admin_logged_in' in session:
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            models.add_teacher(username, password)
            return redirect(url_for('admin_teachers'))
        else:
            return render_template('add_teacher.html')
    return redirect(url_for('admin_login'))

@app.route('/admin/update_teacher/<int:teacher_id>', methods=['GET', 'POST'])
def update_teacher(teacher_id):
    if 'admin_logged_in' in session:
        if request.method == 'POST':
            try:
                username = request.form['username']
                password = request.form['password']
                models.update_teacher(teacher_id, username, password)
                return redirect(url_for('admin_teachers'))
            except Exception as e:
                print("Error updating teacher:", e)
                # In a real app you would flash an error message
                return redirect(url_for('admin_teachers'))
        else:
            # GET: fetch teacher data and show form
            cur = mysql.connection.cursor()
            cur.execute("SELECT * FROM Teachers WHERE teacher_id = %s", (teacher_id,))
            teacher = cur.fetchone()
            cur.close()
            if teacher:
                return render_template('update_teacher.html', teacher=teacher, teacher_id=teacher_id)
            else:
                return "Teacher not found"
    return redirect(url_for('admin_login'))

@app.route('/admin/delete_teacher/<int:teacher_id>', methods=['POST'])
def delete_teacher(teacher_id):
    if 'admin_logged_in' in session:
        try:
            models.delete_teacher(teacher_id)
            return redirect(url_for('admin_teachers'))
        except Exception as e:
            print("Error deleting teacher:", e)
            return redirect(url_for('admin_teachers'))
    return redirect(url_for('admin_login'))

@app.route('/admin/view_teacher_tests/<int:teacher_id>')
def view_teacher_tests(teacher_id):
    if 'admin_logged_in' in session:
        tests = models.get_teacher_tests(teacher_id)
        return render_template('view_teacher_tests.html', tests=tests, teacher_id=teacher_id)
    return redirect(url_for('admin_login'))

@app.route('/admin/view_test_questions/<int:test_id>')
def view_test_questions(test_id):
    if 'admin_logged_in' in session:
        questions = models.get_test_questions(test_id)
        question_answers = {}
        for q in questions:
            answers = models.get_expected_answers_for_question(q[0])
            question_answers[q[0]] = answers
        return render_template('view_test_questions.html', teacher_id=test_id, questions=questions, question_answers=question_answers)
    return redirect(url_for('admin_login'))

@app.route('/admin/view_question_answers/<int:question_id>')
def view_question_answers(question_id):
    if 'admin_logged_in' in session:
        answers = models.get_expected_answers_for_question(question_id)
        return render_template('view_question_answers.html', answers=answers)
    return redirect(url_for('admin_login'))

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))

# -------------------------------------------------------------------
# Teacher routes
# -------------------------------------------------------------------
@app.route('/teacher_login', methods=['GET', 'POST'])
def teacher_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM Teachers WHERE username = %s AND password = %s", (username, password))
        teacher = cur.fetchone()
        cur.close()

        if teacher:
            session['teacher_logged_in'] = True
            session['teacher_id'] = teacher[0]
            return redirect(url_for('teacher_home'))
        else:
            return render_template('teacher_login.html', error='Invalid username or password')

    return render_template('teacher_login.html')

@app.route('/teacher_home', methods=['GET', 'POST'])
def teacher_home():
    if 'teacher_logged_in' not in session:
        return redirect(url_for('teacher_login'))

    if request.method == 'POST':
        if 'add_test_name' in request.form:
            test_name = request.form['test_name']
            models.add_test(test_name, session['teacher_id'])
        elif 'update_test_name' in request.form:
            test_id = request.form['test_id']
            updated_test_name = request.form['updated_test_name']
            models.update_test_name(test_id, updated_test_name)
        elif 'delete_test_name' in request.form:
            test_id = request.form['test_id']
            models.delete_test(test_id)

    tests = models.get_teacher_tests(session['teacher_id'])
    return render_template('teacher_home.html', tests=tests)

@app.route('/teacher_logout')
def teacher_logout():
    session.pop('teacher_logged_in', None)
    session.pop('teacher_id', None)
    return redirect(url_for('teacher_login'))

@app.route('/teacher/view_test_questions/<int:test_id>', methods=['GET', 'POST'])
def view_teacher_test_questions(test_id):
    if 'teacher_logged_in' not in session:
        return redirect(url_for('teacher_login'))

    if request.method == 'POST':
        if 'add_question' in request.form:
            question_text = request.form['question_text']
            expected_answers = request.form.getlist('expected_answer')
            models.add_question_to_test(test_id, question_text, expected_answers)
        elif 'delete_question' in request.form:
            question_id = request.form['question_id']
            models.delete_question(question_id)

    questions = models.get_test_questions(test_id)
    question_answers = {}
    for q in questions:
        answers = models.get_expected_answers_for_question(q[0])
        question_answers[q[0]] = answers

    return render_template('view_teacher_test_questions.html', teacher_id=test_id, questions=questions, question_answers=question_answers)

@app.route('/teacher_view_score')
def teacher_view_score():
    if 'teacher_logged_in' not in session:
        return redirect(url_for('teacher_login'))

    teacher_id = session['teacher_id']
    results = models.get_teacher_scores(teacher_id)

    student_scores = defaultdict(lambda: {'student_username': None, 'tests': defaultdict(list)})
    for row in results:
        student_id, student_username, test_name, question_text, expected_answer, student_answer, score = row
        score = score if score is not None else 0
        student_scores[student_id]['student_username'] = student_username
        student_scores[student_id]['tests'][test_name].append({
            'question_text': question_text,
            'expected_answer': expected_answer,
            'student_answer': student_answer,
            'score': score
        })

    return render_template('teacher_view_score.html', student_scores=student_scores)

# -------------------------------------------------------------------
# Student routes
# -------------------------------------------------------------------
@app.route('/student_login', methods=['GET', 'POST'])
def student_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        student = models.get_student_by_credentials(username, password)

        if student:
            session['student_logged_in'] = True
            session['student_id'] = student[0]
            return redirect(url_for('student_home'))
        else:
            return render_template('student_login.html', error='Invalid username or password')

    return render_template('student_login.html')

@app.route('/student_home')
def student_home():
    if 'student_logged_in' in session:
        return render_template('student_home.html')
    return redirect(url_for('student_login'))

@app.route('/student_logout')
def student_logout():
    session.pop('student_logged_in', None)
    session.pop('student_id', None)
    return redirect(url_for('student_login'))

@app.route('/student_take_test', methods=['GET', 'POST'])
def student_take_test():
    if 'student_logged_in' not in session:
        return redirect(url_for('student_login'))

    if request.method == 'POST':
        test_id = request.form.get('test_id')
        student_id = session['student_id']

        # Loop through form data to retrieve answers for each question
        for key, answer in request.form.items():
            if key.startswith('question_'):
                question_id = int(key.split('_')[1])
                models.insert_student_answer(student_id, test_id, question_id, answer)

        return redirect(url_for('student_view_score'))
    else:
        tests = models.get_available_tests_for_student(session['student_id'])
        # Convert tuples to dicts for template
        tests = [{'test_id': t[0], 'test_name': t[1]} for t in tests]
        return render_template('student_take_test.html', tests=tests)

@app.route('/student_take_test/<int:test_id>', methods=['GET', 'POST'])
def student_take_test_questions(test_id):
    if 'student_logged_in' not in session:
        return redirect(url_for('student_login'))

    if request.method == 'POST':
        student_id = session['student_id']
        for key, answer in request.form.items():
            if key.startswith('question_'):
                question_id = int(key.split('_')[1])
                models.insert_student_answer(student_id, test_id, question_id, answer)
        return redirect(url_for('student_home'))
    else:
        test = models.get_test_by_id(test_id)
        questions = models.get_questions_for_test(test_id)
        return render_template('student_take_test_questions.html', test=test, questions=questions, test_id=test_id)

@app.route('/student_view_score')
def student_view_score():
    if 'student_logged_in' not in session:
        return redirect(url_for('student_login'))

    student_id = session['student_id']
    results = models.get_student_answers_with_expected(student_id)

    student_scores = {}
    for row in results:
        test_id, test_name, question_text, expected_answer, student_answer = row
        score = evaluation.evaluate(expected_answer, student_answer)
        # Ensure score is an integer
        score = int(score)
        # Update the score in the database
        models.update_student_score(student_id, test_id, question_text, score)

        if test_id not in student_scores:
            student_scores[test_id] = {
                'test_id': test_id,
                'test_name': test_name,
                'total_score': 0,
                'max_score': 0,
                'scores': []
            }

        student_scores[test_id]['scores'].append({
            'question': question_text,
            'expected_answer': expected_answer,
            'student_answer': student_answer,
            'score': score
        })
        student_scores[test_id]['total_score'] += score
        student_scores[test_id]['max_score'] += 10

    # Format total_score as "X / Y"
    for test_data in student_scores.values():
        test_data['total_score'] = f"{test_data['total_score']} / {test_data['max_score']}"

    return render_template('student_view_score.html', student_scores=student_scores.values())

# -------------------------------------------------------------------
# Run the app
# -------------------------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)