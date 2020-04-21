FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         ca-certificates \
         libjpeg-dev \
         libpng-dev \
         python3-pip \
         python3-dev \
         python3-setuptools\
         libxkbcommon-x11-0 \
         swig \
         eog\ 
         tmux\
         nano\
         htop\
         octave\
         wget\
         && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install opencv-python
RUN pip3 install cython
RUN pip3 install pyyaml packaging scipy matplotlib pycocotools sklearn
RUN pip3 install Pillow==6.1
RUN pip3 install tensorboardX
     
# enables tmux to use mouse
RUN echo "set -g mouse on" > ~/.tmux.conf

# enable pdb autocomplete
RUN  printf "import rlcompleter\nimport pdb\npdb.Pdb.complete = rlcompleter.Completer(locals()).complete" > ~/.pdbrc

WORKDIR "/root/Bosted-OICR/"