FROM python:3.8.3

RUN pip install numpy scikit-image argparse deap opencv-python scoop

WORKDIR = /usr/src/
COPY see/ ./see
COPY container.py ./container.py

ENTRYPOINT ["python", "container.py"]