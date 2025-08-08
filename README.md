# UniBot – AI-Powered Academic Chatbot for Bennett University

UniBot is an intelligent, NLP-powered chatbot designed to help students and faculty access academic information like professor cabin locations, specializations, and real-time class schedules using natural, conversational language. Built with Python and Flask, UniBot integrates multiple machine learning and web scraping techniques to provide contextual, real-time responses - all while being fully containerized for scalable deployment.

---

## Features

- **Locate Professor Cabins** via simple chat queries  
- **Intent Classification** using a custom-trained ANN model  
- **Fetch Professor Specializations** via Proxycurl and SerpAPI  
- **Timetable Queries** including next class, free time, etc.  
- **Follow-Up Questions** with session-based memory  
- **Secure Authentication** for login and role-based access  
- **Professor-only Interfaces** to update cabins and timetables  
- **Dockerized Deployment** for environment consistency  
- **Real-time scraping**, NLP, TF-IDF, Regex, and Coreference resolution

---

## Tech Stack

### Backend:
- Python 3.7
- Flask Microframework
- SQLite (File-based database)

### Machine Learning & NLP:
- Keras (Sequential ANN for intent classification)
- TF-IDF for specialization queries
- SpaCy, NLTK for text preprocessing
- AllenNLP (coref-spanBERT-large for coreference resolution)
- Regex + GPT (fallback logic for name/entity extraction)
- Mistral/OpenAI (for final response polishing)

### APIs:
- Proxycurl API (LinkedIn scraping for professor background)
- SerpAPI (Google search for LinkedIn URL fallback)
- PhantomBuster (for deeper profile data)

### Deployment:
- Docker (complete containerization)
- Render (public deployment platform)
- GitHub (source code hosting)

---

## Project Structure

```bash
chatbot_project/
│
├── chatbot.py              # Main Flask app
├── intent_model.h5         # Trained intent classification model
├── intent_dataset.json     # Custom dataset for training
├── tokenizer.pkl           # Tokenizer for input preprocessing
├── label_encoder.pkl       # Label encoder for intents
├── coref_resolver.py       # Coreference resolution logic
├── extract_professor.py    # TF-IDF, Regex, NER, and GPT fallback logic
├── templates/              # HTML templates (login, chat, update cabin/timetable)
├── static/                 # CSS, JS, and assets
├── Dockerfile              # Docker container setup
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

```

## Setup Instructions

### Prerequisites

- Python 3.7  
- Git  
- LLM Model: [Mistral-7B-Instruct-v0.3-Q4](https://huggingface.co/jfer1015/Mistral-7B-Instruct-v0.3-Q4_K_M-GGUF). After downloading the model, please save it in a new folder called 'models'.
- Docker (optional for deployment)  
- API Keys for:
  - [Proxycurl](https://proxycurl.com)
  - [SerpAPI](https://serpapi.com)
  - [PhantomBuster](https://phantombuster.com)

---

### Local Setup (Without Docker)

```bash
git clone https://github.com/JustJay7/CampusCompass.git
cd CampusCompass
pip install -r requirements.txt
python chatbot.py
```
- Then visit: http://localhost:5000 in your browser.

---

## Docker Setup
```bash
docker build -t unibot-chatbot .
docker run -d -p 5000:5000 unibot-chatbot
```

---

## Login Credentials
### Only email IDs that end with @bennett.edu.in can access the chatbot. Therefore, please use one of the email IDs from the professor or student login deatils table. 

---

## Contributors

- Jay Malhotra – BCA, Bennett University
- Suryesh Pandey - BSC, Bennett University
