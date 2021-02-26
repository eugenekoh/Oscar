FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
WORKDIR /workspace

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y default-jre git golang-go pigz python3-pip unzip vim wget && \
    go get -u github.com/odeke-em/drive/cmd/drive

# install pytorch
RUN pip3 install numpy && \
    pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu112/torch_nightly.html

# install apex
RUN git clone https://github.com/NVIDIA/apex.git && \
    cd apex && \
    sed 's/check_cuda_torch_binary_vs_bare_metal(torch.utils.cpp_extension.CUDA_HOME)/pass/g' -i setup.py && \
    pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# install oscar
RUN cd /workspace && \
    git clone --recursive https://github.com/eugenekoh/Oscar.git && \
    cd Oscar/coco_caption && \
    ./get_stanford_models.sh && \
    cd .. && \
    python3 setup.py build develop && \
    pip3 install -r requirements.txt

# get models
RUN mkdir models && \
    wget https://biglmdiag.blob.core.windows.net/oscar/exp/coco_caption/base/checkpoint.zip && \
    unzip checkpoint.zip -d models && \
    rm checkpoint.zip && \
    wget https://biglmdiag.blob.core.windows.net/oscar/pretrained_models/base-vg-labels.zip && \
    unzip base-vg-labels.zip -d models && \
    rm base-vg-labels.zip

