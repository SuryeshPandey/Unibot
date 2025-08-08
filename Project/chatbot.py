from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pandas as pd
import sqlite3, random, smtplib, re, requests
from email.mime.text import MIMEText
import spacy
import requests
from bs4 import BeautifulSoup
import openai
import re
from duckduckgo_search import DDGS
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import neuralcoref
from openai import OpenAI
import json
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import time
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
from datetime import timedelta
import dateparser

intent_model = load_model("intent_model.h5", compile=False)
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

client = OpenAI(api_key="") # Insert OpenAI API key inside "".
PROXYCURL_API_KEY = "" # Insert Proxycurl API key inside "".

nltk.download("punkt")
nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
neuralcoref.add_to_pipe(nlp)
print("✅ NeuralCoref successfully added to SpaCy pipeline")
stop_words = set(stopwords.words("english"))

app = Flask(__name__)
app.secret_key = "supersecretkey"

DB_NAME = "campus_chatbot.db"

max_len = 20

def rewrite_response_with_mistral(final_response):
    try:
        prompt = f"""You are a helpful campus assistant. 
Rephrase the following response to sound more natural, friendly, and conversational.
Keep all factual details such as names, cabin numbers, email addresses, and timetables unchanged.

Response to rewrite:
{final_response}
"""

        url = "http://10.12.102.229:8080"
        payload = {
            "prompt": prompt,
            "temperature": 0.7,
            "max_new_tokens": 100,
            "stop": ["\nUser:"],
        }

        response = requests.post(url, json=payload)
        response_data = response.json()

        if "text" in response_data:
            polished_response = response_data["text"].strip()
            return polished_response
        else:
            return final_response

    except Exception as e:
        print(f"Error contacting Mistral model: {e}")
        return final_response


def predict_intent(user_input):
    padded = pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=max_len, padding='post')
    pred = intent_model.predict(padded, verbose=0)[0]
    intent_idx = np.argmax(pred)
    label = label_encoder.classes_[intent_idx]
    confidence = pred[intent_idx]
    return label, confidence

def fetch_professor_timetable(prof_name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT day, slot, course_code, room
        FROM professor_timetables
        WHERE email = (SELECT email FROM login_professors WHERE name LIKE ?)
        ORDER BY day, slot
    """, (f"%{prof_name}%",))
    timetable = cursor.fetchall()
    conn.close()
    return timetable

def get_current_time_slot():
    now = datetime.now()
    current_hour = now.hour
    current_minute = now.minute
    current_time = current_hour * 60 + current_minute

    slot_times = [
        ("8:30–9:25", 510, 565),
        ("9:30–10:25", 570, 625),
        ("10:40–11:35", 640, 695),
        ("11:40–12:35", 700, 755),
        ("12:40–1:30", 760, 810),
        ("1:30–2:25", 810, 865),
        ("2:30–3:25", 870, 925),
        ("3:40–4:35", 940, 995),
        ("4:40–5:35", 1000, 1055)
    ]

    for slot_name, start_min, end_min in slot_times:
        if start_min <= current_time <= end_min:
            return slot_name
    return None

slot_start_times = {
    "8:30–9:25": 510,
    "9:30–10:25": 570,
    "10:40–11:35": 640,
    "11:40–12:35": 700,
    "12:40–1:30": 760,
    "1:30–2:25": 810,
    "2:30–3:25": 870,
    "3:40–4:35": 940,
    "4:40–5:35": 1000
}

def get_or_create_chat_session(email, first_message="New Chat"):
    session_id = session.get("chat_session_id")

    if session_id:
        return session_id

    title = (first_message[:30] + "...") if len(first_message) > 30 else first_message

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_sessions (user_email, title)
        VALUES (?, ?)
    ''', (email, title))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    session["chat_session_id"] = session_id
    return session_id

def save_chat_message(session_id, email, message, is_user):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_messages (session_id, user_email, message, is_user)
        VALUES (?, ?, ?, ?)
    ''', (session_id, email, message, is_user))
    conn.commit()
    conn.close()

def get_linkedin_profile_data(linkedin_url):
    headers = {
        "Authorization": f"Bearer {PROXYCURL_API_KEY}"
    }

    params = {
        "url": linkedin_url,
        "use_cache": "if-present",
        "skills": "include",
        "inferred_salary": "include",
        "personal_contact_number": "include",
        "personal_email": "include"
    }

    response = requests.get("https://nubela.co/proxycurl/api/v2/linkedin", headers=headers, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.text)
        return None

def extract_professor_name(text, history=[]):
    from flask import session
    print("Step 1: Cleaning possessive forms...")
    text_cleaned = re.sub(r"\b([A-Za-z]+ [A-Za-z]+)'s\b", r"\1", text)

    print("Step 2: Resolving coreferences...")
    resolved_text = nlp(text_cleaned)._.coref_resolved or text_cleaned
    print(f"Resolved Text: {resolved_text}")

    print("Step 3: Running SpaCy NER...")
    doc = nlp(resolved_text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            professor_name = ent.text.strip().title()
            print(f"SpaCy NER matched: {professor_name}")
            session["last_professor"] = professor_name
            return professor_name

    print("Step 4: Trying NNP + NNP rule-based fallback...")
    proper_nouns = []
    for token in doc:
        if token.pos_ == "PROPN":
            proper_nouns.append(token.text)
        elif proper_nouns:
            break
    if len(proper_nouns) >= 2:
        name_guess = " ".join(proper_nouns[:2]).title()
        print(f"Rule-based match: {name_guess}")
        session["last_professor"] = name_guess
        return name_guess

    print("Step 5: Falling back to OpenAI GPT...")
    try:
        gpt_prompt = f"""
User's question: "{text}"
Conversation history: {history}

Extract the full professor name from the user's query, if one is present. Respond ONLY with the name, or "None".
"""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": gpt_prompt}],
            temperature=0.3
        )
        print(response)
        content = response.choices[0].message.content.strip()
        if content.lower() == "none":
            print("GPT did not find any name.")
            return session.get("last_professor")
        print(f"GPT extracted: {content}")
        session["last_professor"] = content.title()
        return content.title()
    except Exception as e:
        print(f"GPT fallback failed: {e}")
        return session.get("last_professor")

import pickle

with open("label_encoder.pkl", "rb") as f:
    label_tokenizer = pickle.load(f)



def search_linkedin(prof_name):
    query = f"{prof_name} site:linkedin.com/in/ Bennett University"
    serp_api_key = "" # Insert Serp API key inside "".
    params = {
        "q": query,
        "api_key": serp_api_key,
        "engine": "google"
    }
    resp = requests.get("https://serpapi.com/search", params=params)
    data = resp.json()
    for result in data.get("organic_results", []):
        if "linkedin.com/in/" in result.get("link", ""):
            return result["link"]
    return None

def fetch_linkedin_profile(link):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        page = requests.get(link, headers=headers)
        soup = BeautifulSoup(page.text, "html.parser")
        text = soup.get_text(separator=" ")
        return text
    except Exception as e:
        print("Error fetching LinkedIn:", e)
        return ""

def preprocess_text(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered = [w.lower() for w in words if w.isalnum() and w.lower() not in stop_words]
    
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "NORP", "WORK_OF_ART", "FIELD", "PERSON"]]
    
    tfidf = TfidfVectorizer(stop_words="english", max_features=10)
    tfidf_matrix = tfidf.fit_transform([text])
    keywords = tfidf.get_feature_names_out()
    
    return filtered, entities, keywords

def build_nlp_summary(name, entities, keywords):
    sentence = f"Dr. {name} appears to be affiliated with "
    university = next((e for e in entities if "bennett" in e.lower()), None)
    sentence += f"{university or 'a reputed institution'}."

    if keywords is not None and len(keywords) > 0:
        sentence += f" Their areas of expertise likely include {', '.join(keywords)}."
    else:
        sentence += " Their specialization is currently unclear."

    return sentence

def get_professor_specialization(name):
    print(f"\nLooking up specialization for: {name}")

    try:
        print("Searching LinkedIn using SerpAPI...")
        serp_api_key = "" # Insert Serp API key inside "".
        params = {
            "q": f"{name} Bennett University site:linkedin.com/in/",
            "api_key": serp_api_key,
            "engine": "google"
        }
        serp_response = requests.get("https://serpapi.com/search", params=params)
        data = serp_response.json()
        results = data.get("organic_results", [])
        profile_url = next((res["link"] for res in results if "linkedin.com/in/" in res.get("link", "")), None)

        if not profile_url:
            return f"No LinkedIn URL found for {name}"

        print(f"LinkedIn profile found: {profile_url}")

        print("Fetching data from Proxycurl...")
        proxycurl_key = "" # Insert Proxycurl API key inside "".
        headers = {"Authorization": f"Bearer {proxycurl_key}"}
        proxycurl_response = requests.get(
            "https://nubela.co/proxycurl/api/v2/linkedin",
            headers=headers,
            params={"url": profile_url}
        )

        enriched_data = proxycurl_response.json()
        combined_text = (
            enriched_data.get("headline", "") + "\n" +
            (enriched_data.get("summary", "") or "") + "\n" +
            "\n".join([
                f"{exp.get('title', '')} at {exp.get('company', '')}"
                for exp in enriched_data.get("experiences", [])
            ])
        )

        print(f"\nProxycurl Data (first 500 chars):\n{combined_text[:500]}")

    except Exception as e:
        print(f"Proxycurl/SerpAPI failed: {e}")
        return f"Could not fetch specialization info for {name}."

    print("\nRunning NLP pipeline...")
    tokens = word_tokenize(combined_text)
    print(f"Tokens (first 20): {tokens[:20]}")

    filtered = [w.lower() for w in tokens if w.isalnum() and w.lower() not in stop_words]
    print(f"Filtered (first 20): {filtered[:20]}")

    try:
        tfidf = TfidfVectorizer(stop_words="english", max_features=6)
        tfidf_matrix = tfidf.fit_transform([" ".join(filtered)])
        keywords = tfidf.get_feature_names_out()
    except:
        keywords = ["education", "professor", "research"]

    print(f"Top TF-IDF Keywords: {list(keywords)}")

    doc = nlp(combined_text)
    entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PERSON", "GPE", "WORK_OF_ART", "NORP"]]
    print(f"Named Entities: {entities}")

    prompt = f"""
"You are writing a concise academic summary (max 3 sentences) for Dr. {name} to display in a chatbot. Mention their designation, department, and specialization area. Avoid redundant phrases, dates, or extended biography."

Entities: {', '.join(entities)}
Keywords: {', '.join(keywords)}

Text: {combined_text[:1000]}
"""

    try:
        completion = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print("LLM Error:", str(e))
        return build_nlp_summary(name, entities, keywords)

def get_db_context(query):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    context = {}
    clean_query = query.lower().strip()

    if any(keyword in clean_query for keyword in ['professor', 'dr.', 'doctor', 'cabin', 'office', 'location']):
        name_match = re.search(r"(?:professor|dr\.?|mr\.?|ms\.?)?\s*([A-Z][a-z]+\s+[A-Z][a-z]+)", query, re.IGNORECASE)
        if name_match:
            prof_name = name_match.group(1)
            cursor.execute("SELECT name, specialization, cabin FROM professors WHERE name LIKE ?", ('%' + prof_name + '%',))
        else:
            cursor.execute("SELECT name, specialization, cabin FROM professors WHERE specialization LIKE ?", ('%' + clean_query + '%',))
        context['professors'] = [{
            'name': n, 'specialization': s, 'cabin': c, 'formatted': f"{n} ({s}) - Cabin: {c}"
        } for n, s, c in cursor.fetchall()]

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

    return f"""### System: {system_prompt}\n### Context:{context_str}\n### User Query: {query}\n### CampusGPT:"""


@app.route("/chat", methods=["GET", "POST"])
def chat():
    print("/chat route HIT")
    if request.method == "GET":
        return render_template("chat.html")

    user_input = request.json.get("query", "").strip()
    print(f"📌 Last Professor in session: {session.get('last_professor')}")
    user_input_lower = user_input.lower()

    email = session.get("email")
    session_id = get_or_create_chat_session(email, user_input)
    save_chat_message(session_id, email, user_input, is_user=True)

    intent, confidence = predict_intent(user_input)
    print(f"Incoming Query: {user_input}")
    print(f"Predicted Intent: {intent} (confidence: {confidence:.2f})")

    if session.get("user_type") == "professor" and any(word in user_input_lower for word in ["my", "i", "me", "mine"]):
        prof_name = session.get("name")
        session["last_professor"] = prof_name
        print(f"🧠 Auto-using logged-in professor's name: {prof_name}")
    else:
        prof_name = extract_professor_name(user_input, session.get("chat_history", []))
        if prof_name:
            session["last_professor"] = prof_name
        elif session.get("last_professor"):
            prof_name = session["last_professor"]

    final_response = ""

    if ("cabin" in user_input_lower or "sit" in user_input_lower) and ("specialize" in user_input_lower or "specialization" in user_input_lower or "teach" in user_input_lower):
        if not prof_name:
            final_response = "Please mention the professor's name clearly."
        else:
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()

            cursor.execute("SELECT name, cabin FROM professors WHERE name LIKE ?", (f"%{prof_name}%",))
            row1 = cursor.fetchone()
            cabin_response = f"{row1[0]} sits in cabin {row1[1]}." if row1 else f"I couldn't find {prof_name}'s cabin."

            specialization_response = get_professor_specialization(prof_name)

            conn.close()
            final_response = f"{cabin_response}\n\n{specialization_response}"

    elif intent == "get_cabin":
        if prof_name:
            print(f"🧠 Looking up cabin for: {prof_name}")
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT name, cabin FROM professors WHERE name LIKE ?", (f"%{prof_name}%",))
            row = cursor.fetchone()
            conn.close()
            final_response = f"{row[0]} sits in cabin {row[1]}." if row else f"I couldn't find {prof_name}'s cabin."
        else:
            final_response = "Please mention the professor's name clearly."

    elif intent == "get_specialization":
        if prof_name:
            final_response = get_professor_specialization(prof_name)
        else:
            final_response = "Please specify the professor's name."

    elif intent == "get_professor_free_time":

        DAYS_OF_WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        text_cleaned = re.sub(r"\b([A-Za-z]+ [A-Za-z]+)'s\b", r"\1", user_input)
        resolved_text = nlp(text_cleaned)._.coref_resolved or text_cleaned
        resolved_text = resolved_text.lower()

        target_day = None
        for day in DAYS_OF_WEEK:
            if day in resolved_text:
                target_day = day
                break

        if not target_day:
            if any(word in resolved_text for word in ["today", "later", "evening", "this day"]):
                target_day = datetime.now().strftime('%A').lower()
            elif "tomorrow" in resolved_text:
                target_day = (datetime.now() + timedelta(days=1)).strftime('%A').lower()
            else:
                target_day = datetime.now().strftime('%A').lower()


        print(f"Target day extracted: {target_day}")

        if prof_name:
            if session.get("user_type") == "professor" and prof_name == session.get("name"):
                timetable = session.get("timetable", [])
            else:
                timetable = fetch_professor_timetable(prof_name)

            timetable.sort(key=lambda x: slot_start_times.get(x[1], float('inf')))

            if not timetable:
                final_response = f"I couldn't find a timetable for {prof_name}."
            else:
                busy_slots = [slot for day_, slot, code, room in timetable if day_.lower() == target_day and code and room]
                ALL_SLOTS = [
                    "8:30–9:25", "9:30–10:25", "10:40–11:35", "11:40–12:35",
                    "12:40–1:30", "1:30–2:25", "2:30–3:25", "3:40–4:35", "4:40–5:35"
                ]
                free_slots = [slot for slot in ALL_SLOTS if slot not in busy_slots]

                if free_slots:
                    if prof_name == session.get("name"):
                        final_response = f"You are free during {', '.join(free_slots)} on {target_day.title()}."
                    else:
                        final_response = f"{prof_name} is free during {', '.join(free_slots)} on {target_day.title()}."
                else:
                    if prof_name == session.get("name"):
                        final_response = f"You have no free slots on {target_day.title()}."
                    else:
                        final_response = f"{prof_name} has no free slots on {target_day.title()}."
        else:
            final_response = "Please mention the professor's name clearly."


    elif intent == "get_next_class":

        DAYS_OF_WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        text_cleaned = re.sub(r"\b([A-Za-z]+ [A-Za-z]+)'s\b", r"\1", user_input)
        resolved_text = nlp(text_cleaned)._.coref_resolved or text_cleaned
        resolved_text = resolved_text.lower()

        target_day = None
        for day in DAYS_OF_WEEK:
            if day in resolved_text:
                target_day = day
                break

        if not target_day:
            if any(word in resolved_text for word in ["today", "later", "evening", "this day"]):
                target_day = datetime.now().strftime('%A').lower()
            elif "tomorrow" in resolved_text:
                target_day = (datetime.now() + timedelta(days=1)).strftime('%A').lower()
            else:
                target_day = datetime.now().strftime('%A').lower()

        print(f"Target day extracted: {target_day}")

        if prof_name:
            timetable = fetch_professor_timetable(prof_name)
        else:
            timetable = session.get("timetable", [])

        timetable.sort(key=lambda x: slot_start_times.get(x[1], float('inf')))

        if not timetable:
            final_response = "Your timetable is empty." if session.get("user_type") == "professor" else f"I couldn't find a timetable for {prof_name}."
        else:
            current_slot = get_current_time_slot()
            next_classes = []
            for day_, slot, code, room in timetable:
                if day_.lower() == target_day and code and room:
                    if current_slot is None or slot_start_times.get(slot, float('inf')) > slot_start_times.get(current_slot, float('-inf')):
                        next_classes.append((slot, code, room))

            if next_classes:
                next_slot, next_code, next_room = next_classes[0]
                if prof_name and session.get("user_type") == "student":
                    final_response = f"{prof_name}'s next class today is {next_code} at {next_slot} in {next_room}."
                else:
                    final_response = f"Your next class today is {next_code} at {next_slot} in {next_room}."
            else:
                final_response = "You have no more classes today."


    elif intent == "ask_timetable":

        DAYS_OF_WEEK = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        text_cleaned = re.sub(r"\b([A-Za-z]+ [A-Za-z]+)'s\b", r"\1", user_input)
        resolved_text = nlp(text_cleaned)._.coref_resolved or text_cleaned
        resolved_text = resolved_text.lower()

        target_day = None
        for day in DAYS_OF_WEEK:
            if day in resolved_text:
                target_day = day
                break

        if not target_day:
            if any(word in resolved_text for word in ["today", "later", "evening", "this day"]):
                target_day = datetime.now().strftime('%A').lower()
            elif "tomorrow" in resolved_text:
                target_day = (datetime.now() + timedelta(days=1)).strftime('%A').lower()
            else:
                target_day = datetime.now().strftime('%A').lower()


        print(f"Target day extracted: {target_day}")

        if prof_name:
            timetable = fetch_professor_timetable(prof_name)
        else:
            timetable = session.get("timetable", [])

        timetable.sort(key=lambda x: slot_start_times.get(x[1], float('inf')))

        if not timetable:
            final_response = "Your timetable is empty." if session.get("user_type") == "professor" else f"I couldn't find a timetable for {prof_name}."
        else:
            day_classes = [
                (slot, code, room)
                for day_, slot, code, room in timetable
                if day_.lower() == target_day and code and room
            ]
            if day_classes:
                formatted = "\n".join([f"{slot}: {code} in {room}" for slot, code, room in day_classes])
                if prof_name and session.get("user_type") == "student":
                    final_response = f"{prof_name}'s classes on {target_day.title()}:\n{formatted}"
                else:
                    final_response = f"Your classes on {target_day.title()}:\n{formatted}"
            else:
                if prof_name and session.get("user_type") == "student":
                    final_response = f"{prof_name} has no classes on {target_day.title()}."
                else:
                    final_response = f"You have no classes on {target_day.title()}."

    else:
        if intent in ["greeting", "goodbye"]:
            db_context = get_db_context(user_input)
            prompt = generate_llm_prompt(user_input, db_context)
            response = requests.post(
                "http://10.12.102.229:8080/completion",
                json={
                    "prompt": prompt,
                    "n_predict": 100,
                    "temperature": 0.3,
                    "top_k": 30,
                    "top_p": 0.7,
                    "stop": ["###", "\n\n"],
                    "repeat_penalty": 1.2
                }
            ).json()
            final_response = response.get("content", "I couldn't generate a response. Please try again.")
            final_response = final_response.replace("building", "block").replace("room", "cabin")
        else:
            final_response = "I'm sorry, I couldn't understand your request. Could you please rephrase?"

    final_response = rewrite_response_with_mistral(final_response)
    save_chat_message(session_id, email, final_response, is_user=False)
    return jsonify({"response": final_response, "session_id": session.get("chat_session_id")})

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
    if user:
        session["email"] = email
        session["user_type"] = role
        session["name"] = user[1]

        if role == "professor":
            conn = sqlite3.connect(DB_NAME)
            cursor = conn.cursor()
            cursor.execute("SELECT day, slot, course_code, room FROM professor_timetables WHERE email = ?", (email,))
            rows = cursor.fetchall()
            conn.close()
            session["timetable"] = rows

    conn.close()

    if user:
        session["email"] = email
        session["user_type"] = role
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

@app.route("/update-cabin")
def update_cabin():
    return render_template("update_cabin.html")

@app.route("/timetable", methods=["GET", "POST"])
def view_timetable():
    if session.get("user_type") != "professor":
        return "Unauthorized", 403

    email = session.get("email")
    if not email:
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    slots = [
        "8:30–9:25", "9:30–10:25", "10:40–11:35", "11:40–12:35",
        "12:40–1:30", "1:30–2:25", "2:30–3:25", "3:40–4:35", "4:40–5:35"
    ]

    if request.method == "POST":
        for day in days:
            for slot in slots:
                slot_key = slot.replace("–", "-")
                course_field = f"{day}_{slot_key}_course"
                room_field = f"{day}_{slot_key}_room"
                course_code = request.form.get(course_field, "").strip()
                room = request.form.get(room_field, "").strip()

                cursor.execute("""
                    INSERT INTO professor_timetables (email, day, slot, course_code, room)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(email, day, slot) DO UPDATE SET
                    course_code=excluded.course_code,
                    room=excluded.room
                """, (email, day, slot, course_code, room))

        conn.commit()

    cursor.execute("SELECT day, slot, course_code, room FROM professor_timetables WHERE email = ?", (email,))
    rows = cursor.fetchall()
    conn.close()

    timetable = {(day, slot): {"course": c, "room": r} for day, slot, c, r in rows}

    return render_template("timetable.html", days=days, slots=slots, timetable=timetable)

@app.route("/sidebar-sessions")
def get_sidebar_sessions():
    email = session.get("email")

    if not email:
        return jsonify([])

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, title, created_at FROM chat_sessions
        WHERE user_email = ?
        ORDER BY created_at DESC
    ''', (email,))
    sessions = cursor.fetchall()
    conn.close()

    return jsonify([
        {"id": sid, "title": title or "Untitled", "created_at": created}
        for sid, title, created in sessions
    ])

@app.route("/session/<int:session_id>/messages")
def get_session_messages(session_id):
    email = session.get("email")

    if not email:
        return jsonify([])

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT message, is_user, timestamp FROM chat_messages
        WHERE session_id = ? AND user_email = ?
        ORDER BY timestamp ASC
    ''', (session_id, email))
    messages = cursor.fetchall()
    conn.close()

    return jsonify([
        {
            "text": msg,
            "is_user": bool(is_user),
            "timestamp": timestamp
        } for msg, is_user, timestamp in messages
    ])

@app.route("/create-session", methods=["POST"])
def create_empty_session():
    email = session.get("email")
    title = "New Chat"
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO chat_sessions (user_email, title)
        VALUES (?, ?)
    ''', (email, title))
    session_id = cursor.lastrowid
    conn.commit()
    conn.close()

    session["chat_session_id"] = session_id
    return jsonify({"session_id": session_id})

@app.route("/rename-session", methods=["POST"])
def rename_chat_session():
    data = request.json
    session_id = data.get("session_id")
    new_title = data.get("new_title", "").strip()
    email = session.get("email")

    if not session_id or not new_title:
        return jsonify({"success": False, "error": "Missing fields"})

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE chat_sessions
        SET title = ?
        WHERE id = ? AND user_email = ?
    ''', (new_title, session_id, email))
    conn.commit()
    conn.close()

    return jsonify({"success": True})

@app.route("/delete-session", methods=["POST"])
def delete_chat_session():
    data = request.json
    session_id = data.get("session_id")
    email = session.get("email")

    if not session_id:
        return jsonify({"success": False, "error": "Missing session_id"})

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM chat_messages WHERE session_id = ? AND user_email = ?", (session_id, email))
    
    cursor.execute("DELETE FROM chat_sessions WHERE id = ? AND user_email = ?", (session_id, email))
    
    conn.commit()
    conn.close()

    if session.get("chat_session_id") == session_id:
        session.pop("chat_session_id", None)

    return jsonify({"success": True})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
    #app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
