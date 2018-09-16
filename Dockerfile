# use python base image instead of building your own.
from python:3.6

WORKDIR /code

# add and install requirements before anything else to make debugging faster.
ADD requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

ADD . /code

CMD tail -f /bin/sed