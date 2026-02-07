# =========================
# Stage 1: build nvdiffrast
# =========================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 AS builder

RUN sed -i 's@http://archive.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    ca-certificates \
    build-essential \
    curl \
    cmake \
    git \
    curl \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fL https://mirrors.aliyun.com/pypi/get-pip.py -o /tmp/get-pip.py \
 && python3.9 /tmp/get-pip.py

RUN ln -sf /usr/bin/python3.9 /usr/local/bin/python \ 
    && ln -sf /usr/bin/python3.9 /usr/local/bin/python3
RUN python -m pip -V
RUN python -m pip install --no-cache-dir -U "pip<24.1" setuptools wheel 
RUN python -m pip install --no-cache-dir numpy==1.23.5
# 安装 PyTorch（用于编译扩展）https://download.pytorch.org/whl/cu118/torch-2.0.1%2Bcu118-cp39-cp39-linux_x86_64.whl
COPY ./assets/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl /tmp/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl
RUN python -m pip install --no-cache-dir --no-build-isolation /tmp/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl \
 && rm -f /tmp/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl
#RUN python -m pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN echo "force rebuild $(date)" && \
    python -c "import torch; print(torch.cuda.is_available())"

# 编译 nvdiffrast
# 使用本地已下载的 nvdiffrast 源码包进行安装
COPY ./assets/nvdiffrast.tar.gz /tmp/nvdiffrast.tar.gz
ENV TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6"
RUN python -m pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast.tar.gz \
 && rm -f /tmp/nvdiffrast.tar.gz

# =========================
# Stage 2: runtime
# =========================
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

RUN sed -i 's@http://archive.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list && \
    sed -i 's@http://security.ubuntu.com@https://mirrors.aliyun.com@g' /etc/apt/sources.list

RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    echo "Asia/Shanghai" > /etc/timezone

RUN DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    ca-certificates \
    git \ 
    build-essential \ 
    cmake \ 
    pkg-config \ 
    software-properties-common \ 
    ffmpeg \ 
    libboost-all-dev \ 
    libgl1 \ 
    libglib2.0-0 \ 
    libsm6 \ 
    libxext6 \ 
    libxrender1 \ 
    libsndfile1 \ 
    libeigen3-dev \
 && rm -rf /var/lib/apt/lists/*


RUN curl -fL https://mirrors.aliyun.com/pypi/get-pip.py -o /tmp/get-pip.py \
 && python3.9 /tmp/get-pip.py

RUN ln -sf /usr/bin/python3.9 /usr/local/bin/python \ 
&& ln -sf /usr/bin/python3.9 /usr/local/bin/python3

RUN python -m pip -V
RUN python -m pip install --no-cache-dir -U "pip<24.1" setuptools wheel 
RUN python -m pip install --no-cache-dir numpy==1.23.5
# 安装 PyTorch（runtime）
COPY ./assets/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl /tmp/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl
RUN python -m pip install /tmp/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl \
 && rm -f /tmp/torch-2.0.1+cu118-cp39-cp39-linux_x86_64.whl
#RUN python -m pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 🔥 关键：只拷贝已编译好的 nvdiffrast
COPY --from=builder /usr/local/lib/python3.9/dist-packages/nvdiffrast \
                     /usr/local/lib/python3.9/dist-packages/nvdiffrast
COPY --from=builder /usr/local/lib/python3.9/dist-packages/nvdiffrast*.dist-info \
                     /usr/local/lib/python3.9/dist-packages/

CMD ["bash"]
