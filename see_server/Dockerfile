From python:3.8.3

RUN pip install cherrypy
WORKDIR = /usr/src/
COPY server.py ./server.py

ENTRYPOINT ["python", "server.py"]