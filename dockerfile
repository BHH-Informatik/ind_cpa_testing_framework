FROM tensorflow/tensorflow:2.15.0


WORKDIR /rl_tester
COPY /rl_tester .
RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "python", "main.py" ]