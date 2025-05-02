# Unibot - AI-Powered Campus Chatbot for Bennett University

Developed a full-stack intelligent chatbot using Flask, SQLite, and a locally hosted Mistral 7B Instruct model, tailored to answer campus-specific queries. The chatbot handles tasks like retrieving professors’ cabin numbers, timetable-based availability. Integrated LinkedIn and university website scraping to suggest professors based on their specialization and display their credentials. Enabled role-based login (student/professor) with OTP verification, allowing professors to update their timetables and cabin info. Used NLP ( Natural Language Processing ) techniques to parse Excel-based schedules, and designed an interactive chat UI using HTML, CSS, and JavaScript with local Storage-based chat history.


## Key Features

- **Campus-Specific Knowledge**: Answers queries about professors' cabin locations, timetables, and more
- **Local AI Processing**: Uses Mistral 7B Instruct model running locally for privacy and reliability
- **Role-Based Access**: Separate interfaces for students and professors with appropriate permissions
- **Secure Authentication**: OTP verification for user registration
- **Persistent Chat History**: Browser-based chat storage with conversation management
- **Professor Portal**: Allows faculty to update their cabin information and office hours

## Technology Stack

**Backend:**
- Python Flask web framework
- SQLite database
- Mistral 7B Instruct LLM (local deployment)
- SMTP for OTP delivery

**Frontend:**
- HTML5, CSS3, JavaScript
- Responsive design for all devices
- LocalStorage for chat persistence

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager
- SQLite3
- llama.cpp server (for local model hosting)

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SuryeshPandey/Unibot.git
   cd Unibot

```markdown
## 📁 Project Structure

```bash
unibot/
├── chatbot.py          # Main Flask application
├── setup_db.py         # Database schema setup
├── insert_data.py      # Sample data insertion
├── start_chatbot.sh    # Startup script
├── static/
│   └── style.css       # Shared CSS styles
├── templates/          # HTML templates
│   ├── chat.html       # Chat interface
│   ├── login.html      # Login page
│   ├── register.html   # Registration page
│   ├── verify.html     # OTP verification
│   └── update_cabin.html # Professor portal
└── models/             # LLM model storage
    └── mistral-7b-instruct-v0.3-q4_k_m.gguf
