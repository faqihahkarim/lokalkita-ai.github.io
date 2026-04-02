# Use a Python image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy all your files into the container
COPY . .

# Install your requirements
RUN pip install --no-cache-dir -r requirements.txt

# Run the app (Hugging Face uses port 7860 by default)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
