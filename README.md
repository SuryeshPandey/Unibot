# 🤖 UniBot — AI-Powered Academic Chatbot for Bennett University

UniBot is an intelligent, NLP-powered chatbot that helps students and faculty access academic information — professor cabin locations, specializations, and real-time class schedules — using natural, conversational language. Built with Python and Flask, it combines custom machine learning with live web scraping to deliver contextual, real-time responses, all fully containerized for consistent deployment.

> Just ask it where your professor's cabin is — like texting a friend.

---

## ✨ Features

- **Locate Professor Cabins** via simple chat queries
- **Professor Specializations** fetched in real time from the web
- **Timetable Queries** — next class, free slots, schedules, and more
- **Multi-Turn Context** — understands follow-ups like *"What about him?"* or *"Where is she?"*
- **Intent Classification** using a custom-trained feedforward neural network (Keras)
- **Secure Authentication** with role-based access
- **Professor-Only Interfaces** to view and update cabin details and timetables
- **Dockerized Deployment** for environment consistency

---

## 🧠 How It Works

Rather than relying on pre-trained platforms like Dialogflow or Rasa, UniBot runs a **custom intent-classification model trained from scratch in Keras**. The model is a feedforward neural network that predicts user intent from a handcrafted dataset (`intent_dataset.json`), giving full control and transparency over its behavior.

For understanding conversational context, UniBot uses a modern coreference-resolution model (`coref-spanbert-large`) alongside **SpaCy** and **NLTK**, enabling it to track references across a multi-turn conversation.

Real-world professor data (such as specializations) is pulled live through external APIs including **Proxycurl**, **SerpAPI**, and **PhantomBuster**, keeping responses dynamic and connected to real sources.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Python, Flask |
| **Database** | SQLite |
| **ML / NLP** | Keras (custom intent model), SpaCy, NLTK, coref-spanbert-large |
| **Data Sources** | Proxycurl, SerpAPI, PhantomBuster |
| **Frontend** | HTML5, CSS3 |
| **Deployment** | Docker, Render |

---

## 👥 Two Users, One Experience

| Capability | Students | Professors |
|---|:---:|:---:|
| Query timetables & cabins | ✅ | ✅ |
| Ask about specializations | ✅ | ✅ |
| Update personal timetable & cabin | ❌ | ✅ |

Role-based access keeps schedule updates authentic and private without complicating the experience. Professors edit their timetable through a clean, spreadsheet-style grid; students simply type their questions.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional, for containerized runs)
- API keys for Proxycurl, SerpAPI, and/or PhantomBuster

### Run locally

```bash
# Clone the repository
git clone https://github.com/SuryeshPandey/Unibot.git
cd Unibot

# Install dependencies
pip install -r requirements.txt

# Add your API keys to a .env file
# PROXYCURL_API_KEY=your_key
# SERPAPI_API_KEY=your_key
# PHANTOMBUSTER_API_KEY=your_key

# Run the app
python app.py
```

### Run with Docker

```bash
docker build -t unibot .
docker run -p 5000:5000 unibot
```

The app will be available at `http://localhost:5000`.

---

## 🧗 Challenges & Learnings

- Training a neural model on a small, handcrafted dataset while balancing accuracy and generalization
- Fixing coreference-resolution bugs (older packages like NeuralCoref broke on ARM-based Macs)
- Implementing multi-turn conversation memory with careful session tracking
- Isolating chat sessions so users never access each other's history
- Managing API credit limits by rotating between providers during testing

Each problem turned into a crash course in real-world AI system deployment — dependency management, working with external APIs, and containerizing environments to avoid OS-level issues.

---

## 🔮 Future Scope

- Expand the training dataset for a more robust intent model
- Add voice-based input
- Integrate with the university's official timetable APIs
- Add analytics to surface the most common student queries

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo, open an issue, or submit a pull request.

---

## 📄 License

Released under the MIT License. See `LICENSE` for details.
