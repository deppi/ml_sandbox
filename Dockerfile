from python:3.6

WORKDIR /code

ADD requirements.txt /code/requirements.txt

RUN pip install -r /code/requirements.txt

ADD . /code

CMD tail -f /bin/sed