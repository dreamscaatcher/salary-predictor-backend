FROM python:3.9-slim

WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Create model directory
RUN mkdir -p /app/model

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# First train the model, then start the server
CMD python app/train.py && uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4