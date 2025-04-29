# 1. Base image
FROM python:3.7-slim

# 2. Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# 3. Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    git \
    curl \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# 4. Install pip packages
RUN pip install --upgrade pip && \
    pip install cython==0.29.36 && \
    pip install spacy==2.3.5 && \
    pip install flask pandas duckduckgo_search nltk scikit-learn keras openai beautifulsoup4 && \
    pip install tensorflow==2.10.1 && \
    pip install https://github.com/huggingface/neuralcoref/archive/master.zip

# 5. Download SpaCy model + NLTK data
RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader punkt stopwords

# 6. Set working directory
WORKDIR /app

# 7. Copy project files into image
COPY . /app

# 8. Default command
CMD ["python", "chatbot.py"]
