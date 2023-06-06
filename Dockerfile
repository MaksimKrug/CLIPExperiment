FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive

WORKDIR /workdir

COPY ./predict.py categories.json ./
RUN apt-get update -y
RUN apt-get install -y python3.10 python3-pip git
RUN pip install --upgrade torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install git+https://github.com/openai/CLIP.git
RUN ["python3", "-c", "import clip; clip.load('ViT-L/14', 'cpu')"]

ENTRYPOINT ["python3", "./predict.py"]
