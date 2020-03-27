FROM python:3.6.5

RUN mkdir /code
WORKDIR /code

ADD requirements.txt /code/
RUN pip3 install -r requirements.txt
ADD . /code/
COPY register.html /usr/local/lib/python3.6/site-packages/flask_user/templates/flask_user/register.html

EXPOSE 5000