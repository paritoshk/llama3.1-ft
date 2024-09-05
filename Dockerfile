FROM pytorch/pytorch:latest

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src /app/src
COPY .env /app/.env

CMD ["python", "src/test_model.py"]