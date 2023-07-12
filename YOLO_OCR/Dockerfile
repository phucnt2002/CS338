# Use the official Python base image

FROM python:3.11.3

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the required Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY . .

# Expose the container port
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"]
