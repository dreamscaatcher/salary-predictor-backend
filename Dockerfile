# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p /app/model && chmod 777 /app/model

# Run the training script to generate the model with verbose output
RUN echo "Current directory structure:" && \
    find . -type f -ls && \
    echo "Training model..." && \
    python model/train.py && \
    echo "Model directory contents:" && \
    ls -la /app/model && \
    echo "Model file permissions:" && \
    ls -la /app/model/lin_regress.sav

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
