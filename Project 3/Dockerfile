FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
   apt-get upgrade -y && \
   apt-get install -y python3 python3-pip

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY app.py .
COPY best_hurricane_damage_model.h5 .

EXPOSE 5000

CMD ["python", "app.py"]