FROM python:3.9
LABEL authors="soulde"
ENV DEBIAN_FRONTEND noninteractive
SHELL ["/bin/bash", "-c"]

COPY . /root/defending
WORKDIR /root/defending

RUN export TZ="Asia/Shanghai" \
    && ln -snf /usr/share/zoneinfo/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

RUN apt update && apt upgrade -y && apt install -y build-essential \
    xserver-xorg \
    libgl1-mesa-dev  \
    libgl1-mesa-glx  \
    libosmesa6-dev  \
    libglew-dev  \
    libglfw3-dev \
    libglu1-mesa-dev \
    freeglut3-dev

RUN pip install -r requirements.txt && pip install -e ./sandbox

ENTRYPOINT ["/bin/bash"]