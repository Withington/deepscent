FROM tensorflow/tensorflow:latest-gpu-py3
COPY requirements.txt requirements.txt
RUN pip install --requirement requirements.txt
COPY . src
CMD [ "nosetests", "--with-coverage", "--cover-tests"]
