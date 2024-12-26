FROM tensorflow/tensorflow:latest-gpu

RUN pip3 install tensorflow[and-cuda]

WORKDIR /app
COPY requirements.txt /app/
RUN pip3 install -r requirements.txt


COPY . /app

