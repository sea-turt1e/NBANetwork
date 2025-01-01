# Docker build command: docker build -t <image-name> .

# Base image
FROM python:3.12.0a1-alpine3.13

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

