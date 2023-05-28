FROM tensorflow/tensorflow:2.12.0-gpu 
FROM python:3.8
ENV PYTHONUNBUFFERED=1
RUN mkdir /liveness
WORKDIR /liveness
COPY ./ /liveness/
COPY requirements.txt /liveness/
RUN apt-get update
RUN apt-get install -y python3-opencv 
RUN pip install opencv-python
RUN pip --timeout=1000 install --no-cache-dir --upgrade -r /liveness/requirements.txt
CMD python run.py
