FROM continuumio/anaconda3

RUN apt update \
 && apt install libgl1-mesa-swx11
