'''# insert_data.py (modified)
import sqlite3
from datetime import date

def insert_sample_data():
    conn = sqlite3.connect("campus_chatbot.db")
    cursor = conn.cursor()

    # Insert sample professors with more complete data
    professors_data = [
        ("Dr. John Doe", "Machine Learning", "N Block, 3rd Floor, Cabin 312", 
         "john.doe@university.edu", "Mon-Wed 10am-12pm"),
        ("Dr. Jane Smith", "Cyber Security", "N Block, 1st Floor, Cabin 105",
         "jane.smith@university.edu", "Tue-Thu 2pm-4pm"),
        ("Dr. Emily Davis", "Artificial Intelligence", "S Block, 2nd Floor, Cabin 204",
         "emily.davis@university.edu", "Mon-Fri 11am-1pm")
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO professors (name, specialization, cabin, email, office_hours) VALUES (?, ?, ?, ?, ?)", 
        professors_data
    )

    # Insert more detailed mess menu
    mess_data = [
        ("Breakfast", "All", "Poha, Idli, Bread Butter, Tea/Coffee"),
        ("Lunch (Vegetarian)", "All", "Paneer Butter Masala, Aloo Gobi, Dal Tadka, Rice, Roti"),
        ("Lunch (Non-vegetarian)", "All", "Chicken Curry, Fish Fry, Rice, Roti"),
        ("Dinner", "All", "Dal Khichdi, Kadhi Chawal, Salad")
    ]
    cursor.executemany(
        "INSERT OR IGNORE INTO mess_menu (meal_type, day, items) VALUES (?, ?, ?)",
        mess_data
    )

    conn.commit()
    conn.close()
    print("Sample data inserted successfully.")

if __name__ == "__main__":
    insert_sample_data()'''