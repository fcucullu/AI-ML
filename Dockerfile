FROM python:3.7

ENV PYTHONUNBUFFERED 1

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

RUN chmod 777 /app/entrypoint.sh

ENTRYPOINT '/app/entrypoint.sh'