import mysql.connector

# Database configuration (update if needed)
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'akc2sysit25$',  # Enter your MySQL password here
    'database': 'answer_evaluation'
}

table_definitions = [
    '''CREATE TABLE IF NOT EXISTS Admins (
        admin_id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        password VARCHAR(100) NOT NULL
    )''',
    '''CREATE TABLE IF NOT EXISTS Students (
        student_id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        password VARCHAR(100) NOT NULL
    )''',
    '''CREATE TABLE IF NOT EXISTS Teachers (
        teacher_id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(100) NOT NULL,
        password VARCHAR(100) NOT NULL
    )''',
    '''CREATE TABLE IF NOT EXISTS Tests (
        test_id INT AUTO_INCREMENT PRIMARY KEY,
        test_name VARCHAR(255) NOT NULL,
        teacher_id INT,
        FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id) ON DELETE CASCADE
    )''',
    '''CREATE TABLE IF NOT EXISTS Questions (
        question_id INT AUTO_INCREMENT PRIMARY KEY,
        question_text TEXT NOT NULL,
        test_id INT,
        FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE
    )''',
    '''CREATE TABLE IF NOT EXISTS ExpectedAnswers (
        answer_id INT AUTO_INCREMENT PRIMARY KEY,
        answer_text TEXT NOT NULL,
        question_id INT,
        FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE
    )''',
    '''CREATE TABLE IF NOT EXISTS StudentAnswers (
        answer_id INT AUTO_INCREMENT PRIMARY KEY,
        student_id INT,
        test_id INT,
        question_id INT,
        answer_text TEXT,
        score INT,
        FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE,
        FOREIGN KEY (test_id) REFERENCES Tests(test_id) ON DELETE CASCADE,
        FOREIGN KEY (question_id) REFERENCES Questions(question_id) ON DELETE CASCADE
    )''',
    '''CREATE TABLE IF NOT EXISTS teacherstudentrelationship (
        id INT AUTO_INCREMENT PRIMARY KEY,
        teacher_id INT,
        student_id INT,
        FOREIGN KEY (teacher_id) REFERENCES Teachers(teacher_id) ON DELETE CASCADE,
        FOREIGN KEY (student_id) REFERENCES Students(student_id) ON DELETE CASCADE
    )'''
]

def create_tables():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        for ddl in table_definitions:
            cursor.execute(ddl)
        conn.commit()

        # Insert initial admin if not exists
        cursor.execute("SELECT COUNT(*) FROM Admins")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO Admins (username, password) VALUES (%s, %s)", ("admin", "adminpassword"))
            conn.commit()
            print("All tables created successfully and initial admin user added.")
        else:
            print("All tables created successfully.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_tables()
