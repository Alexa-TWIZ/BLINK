# syntax=docker/dockerfile:1

FROM python:3.7

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install --upgrade flair

RUN pip3 install --upgrade yolk3k
RUN echo $(yolk -l)


COPY blink blink
COPY blink_api.py .
COPY data/.flair /root/.flair

CMD [ "uvicorn", "blink_api:app" , "--workers", "1", "--host", "0.0.0.0"]