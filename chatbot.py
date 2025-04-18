# Cleaned-up and fixed chatbot.py with correct routing and unified DB
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import sqlite3, random, smtplib, re, requests
from email.mime.text import MIMEText

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_NAME = "campus_chatbot.db"

# 1. CHATBOT QUERY HANDLER
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("query", "").strip()
    db_context = get_db_context(user_input)
    prompt = generate_llm_prompt(user_input, db_context)

    response = requests.post(
        "http://localhost:8080/completion",
        json={
            "prompt": prompt,
            "n_predict": 150,
            "temperature": 0.3,
            "top_k": 30,
            "top_p": 0.7,
            "stop": ["###", "\n\n"],
            "repeat_penalty": 1.2
        }
    ).json()

    final_response = response.get("content", "I couldn't generate a response. Please try again.")
    final_response = final_response.replace("building", "block").replace("room", "cabin")
    return jsonify({"response": final_response})

# 2. LLM PROMPT LOGIC
def get_db_context(query):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    context = {}
    clean_query = query.lower().strip()

    if any(keyword in clean_query for keyword in ['professor', 'dr.', 'doctor', 'cabin', 'office', 'location']):
        name_match = re.search(r'(dr\.?\s+[\w\s]+)', query, re.IGNORECASE)
        if name_match:
            prof_name = name_match.group(1)
            cursor.execute("SELECT name, specialization, cabin FROM professors WHERE name LIKE ?", ('%' + prof_name + '%',))
        else:
            cursor.execute("SELECT name, specialization, cabin FROM professors WHERE specialization LIKE ?", ('%' + clean_query + '%',))
        context['professors'] = [{
            'name': n, 'specialization': s, 'cabin': c, 'formatted': f"{n} ({s}) - Cabin: {c}"
        } for n, s, c in cursor.fetchall()]

    if any(k in clean_query for k in ['mess', 'menu', 'food', 'lunch', 'dinner']):
        cursor.execute("SELECT meal_type, items FROM mess_menu")
        context['mess_menu'] = [{'meal_type': m, 'items': i} for m, i in cursor.fetchall()]

    conn.close()
    return context

def generate_llm_prompt(query, db_context):
    system_prompt = """You are CampusGPT, a friendly and precise university assistant. Follow these rules:
1. ALWAYS use EXACT database information provided in the Context
2. Respond conversationally but keep professional tone
3. For locations, use precise terms like \"N Block\" and \"Cabin\"
4. If unsure, say \"I don't have that information\" rather than guessing"""

    context_str = ""
    if 'professors' in db_context:
        context_str += "\nProfessors:\n" + "\n".join([p['formatted'] for p in db_context['professors']])
    if 'mess_menu' in db_context:
        context_str += "\nMess Menu:\n" + "\n".join([f"{m['meal_type']}: {m['items']}" for m in db_context['mess_menu']])

    return f"""### System: {system_prompt}\n### Context:{context_str}\n### User Query: {query}\n### CampusGPT:"""

# 3. AUTH & OTP SYSTEM
@app.route("/")
def login():
    return render_template("login.html")

@app.route("/login", methods=["POST"])
def handle_login():
    email = request.form["email"]
    password = request.form["password"]
    role = request.form["role"]

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    if role == "student":
        cursor.execute("SELECT * FROM students WHERE email = ? AND password = ?", (email, password))
    else:
        cursor.execute("SELECT * FROM login_professors WHERE email = ? AND password = ?", (email, password))

    user = cursor.fetchone()
    conn.close()

    if user:
        session["email"] = email
        session["role"] = role
        return render_template("chat.html")
    else:
        return "Invalid credentials or user not found"

@app.route("/register", methods=["GET"])
def register():
    return render_template("register.html")

def send_otp(email, otp):
    sender = "isweathebot@gmail.com"
    password = "iknjqjzlrgzvnhtb"

    msg = MIMEText(f"Your OTP for Campus Compass registration is: {otp}")
    msg["Subject"] = "Campus Compass OTP Verification"
    msg["From"] = sender
    msg["To"] = email

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, email, msg.as_string())
    except Exception as e:
        print(f"Failed to send OTP: {e}")

@app.route("/register", methods=["POST"])
def handle_register():
    name = request.form["name"]
    email = request.form["email"]
    password = request.form["password"]
    confirm_password = request.form["confirm_password"]
    user_type = request.form["user_type"]

    if not email.endswith("@bennett.edu.in"):
        return "Only @bennett.edu.in emails allowed"
    if password != confirm_password:
        return "Passwords do not match"

    otp = str(random.randint(100000, 999999))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("REPLACE INTO pending_otp (email, name, password, user_type, otp) VALUES (?, ?, ?, ?, ?)",
                   (email, name, password, user_type, otp))
    conn.commit()
    conn.close()

    send_otp(email, otp)
    return redirect(url_for("verify_otp_page", email=email))

@app.route("/verify")
def verify_otp_page():
    email = request.args.get("email")
    return render_template("verify.html", email=email)

@app.route("/verify", methods=["POST"])
def verify_otp():
    email = request.form["email"]
    entered_otp = request.form["otp"]

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM pending_otp WHERE email = ?", (email,))
    row = cursor.fetchone()

    if not row:
        return "OTP expired or email not found"

    db_email, name, password, user_type, otp, _ = row
    if entered_otp != otp:
        return "Incorrect OTP"

    if user_type == "student":
        cursor.execute("INSERT INTO students (name, email, password) VALUES (?, ?, ?)", (name, email, password))
    else:
        cursor.execute("INSERT INTO login_professors (name, email, password) VALUES (?, ?, ?)", (name, email, password))

    session.permanent = True

    session["email"] = email
    session["name"] = name
    session["user_type"] = user_type

    cursor.execute("DELETE FROM pending_otp WHERE email = ?", (email,))
    conn.commit()
    conn.close()

    return redirect(url_for("login"))

# 4. OPTIONAL ROUTES
@app.route("/update-cabin")
def update_cabin():
    return render_template("update_cabin.html")

if __name__ == "__main__":
    app.run(debug=True)
