import sqlite3
import pandas as pd
import os

# --- SETUP ---
db_path = "campus_chatbot.db"
base_folder = "professors_data"
professors_excel = os.path.join(base_folder, "cabin_and_specialization.xlsx")
timetable_folder = os.path.join(base_folder, "Timetables")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# --- PART 1: PROFESSORS TABLE ---

cursor.execute("DROP TABLE IF EXISTS professors")
cursor.execute("""
    CREATE TABLE professors (
        email TEXT PRIMARY KEY,
        name TEXT,
        specialization TEXT,
        cabin TEXT
    )
""")
conn.commit()

df_professors = pd.read_excel(professors_excel)
df_professors.to_sql("professors", conn, if_exists="append", index=False)
print("✅ Professors imported successfully.")

# --- PART 2: PROFESSOR TIMETABLES TABLE ---

cursor.execute("""
    CREATE TABLE IF NOT EXISTS professor_timetables (
        email TEXT,
        day TEXT,
        slot TEXT,
        course_code TEXT,
        room TEXT
    )
""")
cursor.execute("DELETE FROM professor_timetables")
conn.commit()

for file in os.listdir(timetable_folder):
    if file.endswith(".xlsx"):
        email = file.replace(".xlsx", "")
        filepath = os.path.join(timetable_folder, file)
        df = pd.read_excel(filepath, header=None)

        days = df.iloc[1, 1:6].tolist()

        for i in range(2, len(df), 2):
            slot = str(df.iloc[i, 0])
            course_row = df.iloc[i, 1:6].tolist()
            room_row = df.iloc[i + 1, 1:6].tolist()

            for day, course, room in zip(days, course_row, room_row):
                course_clean = None if pd.isna(course) or "lunch" in str(course).lower() else course.strip()
                room_clean = None if pd.isna(room) else room.strip()

                if course_clean or room_clean:
                    cursor.execute("""
                        INSERT INTO professor_timetables (email, day, slot, course_code, room)
                        VALUES (?, ?, ?, ?, ?)
                    """, (email, day, slot, course_clean, room_clean))

conn.commit()
conn.close()

print("✅ Timetables successfully imported into 'professor_timetables' table.")
