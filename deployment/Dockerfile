# Base Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies (you can skip g++ if not needed)
RUN apt-get update && apt-get install -y gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download spaCy model (compatible with spacy==2.3.5)
RUN python -m spacy download en_core_web_sm

# Expose the Flask app port
EXPOSE 7860

# Run the app from deployment folder
CMD ["python", "deployment/app.py"]
