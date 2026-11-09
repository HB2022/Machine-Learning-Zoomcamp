# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

# Copy the prediction script and the saved model
COPY predict.py .
COPY model.bin .

# Expose the port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8000"]