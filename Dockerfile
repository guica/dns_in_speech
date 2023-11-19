FROM tensorflow/tensorflow:latest-gpu-jupyter
ENV PYTHONUNBUFFERED 1

# ADD requirements.txt /tf/

RUN pip install pip --upgrade
#RUN pip install -r requirements.txt
RUN apt update && apt install -y ffmpeg
RUN pip install scikit-learn glob2 tqdm pesq pystoi pandas jupyterlab openpyxl librosa pydub seaborn

CMD jupyter lab --ip=0.0.0.0 --port=8888 --allow-root