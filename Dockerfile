FROM python:3.11-slim

# Install system libraries required by OpenCV, dlib, and face_recognition
RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libx11-6 \
  libxcb1 \
  libdlib-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]