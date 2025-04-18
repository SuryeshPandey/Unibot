# setup_db.py
import sqlite3

def setup_database():
    conn = sqlite3.connect("campus_chatbot.db")
    cursor = conn.cursor()

    # Professors data for chatbot queries (NOT login)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS professors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            specialization TEXT NOT NULL,
            cabin TEXT NOT NULL,
            email TEXT,
            office_hours TEXT,
            UNIQUE (name, specialization)
        )
    ''')

    # Mess menu data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mess_menu (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meal_type TEXT NOT NULL,
            day TEXT NOT NULL DEFAULT 'All',
            items TEXT NOT NULL,
            UNIQUE (meal_type, day)
        )
    ''')

    # Navigation routes
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS navigation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_location TEXT NOT NULL,
            end_location TEXT NOT NULL,
            directions TEXT NOT NULL,
            estimated_time TEXT,
            UNIQUE (start_location, end_location)
        )
    ''')

    # User login tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            verified INTEGER DEFAULT 1
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS login_professors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            verified INTEGER DEFAULT 1
        )
    ''')

    # Pending OTP table for unverified users
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pending_otp (
            email TEXT PRIMARY KEY,
            name TEXT,
            password TEXT,
            user_type TEXT,
            otp TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database setup complete.")

if __name__ == "__main__":
    setup_database()
