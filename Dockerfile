FROM python:3.8.3

RUN pip install numpy scikit-image argparse deap opencv-python scoop requests

WORKDIR = /usr/src/
COPY see/ ./see
COPY segment_container.py ./segment_container.py

ENTRYPOINT ["python", "segment_container.py"]