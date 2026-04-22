FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data during build to save time at runtime
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy the rest of the application files
COPY . .

# Expose port 5000 for the Flask API
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=api/app.py
ENV PYTHONUNBUFFERED=1

# Command to run the API using Gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "api.app:app"]
