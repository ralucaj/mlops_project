# Base image
FROM python:3.8-slim

# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

# Copy necessary file
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/

# Select working directory
WORKDIR /

# Install dependencies
RUN pip install -r requirements.txt --no-cache-dir

# Define the application to run when the image is executed
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]