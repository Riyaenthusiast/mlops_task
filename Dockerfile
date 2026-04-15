FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Run the pipeline; outputs go to /app by default
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]
