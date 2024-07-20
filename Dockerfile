FROM python:3.10.12
RUN apt update -y && apt install awscli -y


WORKDIR /code
COPY . /code

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
