FROM ubuntu:18.04

LABEL version=2020.07.09.1

ARG http_proxy
ARG https_proxy
ENV http_proxy ${http_proxy}
ENV https_proxy ${https_proxy}

ARG model=googlenet-v3-pytorch

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED 1

# Install base dependencies
RUN apt-get update && \
    apt-get install -y locales=2.27-3ubuntu1.2 && \
    apt-get clean autoclean && \
    apt-get autoremove -y

# Set the locale to en_US.UTF-8
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

##1 Install Ubuntu 18 requirements

RUN apt-get update && \
    apt-get -y --no-install-recommends install \

# OpenVINO dependencies
    autoconf=2.69-11 \
    automake=1:1.15.1-3ubuntu2 \
    build-essential=12.4ubuntu1 \
    cmake=3.10.2-1ubuntu2.18.04.1 \
    curl=7.58.0-2ubuntu3.9 \
    git=1:2.17.1-1ubuntu0.7 \
    libtool=2.4.6-2 \
    ocl-icd-opencl-dev=2.2.11-1ubuntu1 \
    pkg-config=0.29.1-0ubuntu2 \
    unzip=6.0-21ubuntu1 \
    wget=1.19.4-1ubuntu2.2 \
    libusb-1.0-0-dev=2:1.0.21-2 \
# cmake 13 dependencies
    libssl-dev=1.1.1-1ubuntu2.1~18.04.6 \
    libcurl4-openssl-dev=7.58.0-2ubuntu3.9 \
# OpenCV dependencies
    libgtk2.0-dev=2.24.32-1ubuntu1 \
    libavcodec-dev=7:3.4.8-0ubuntu0.2 \
    libavformat-dev=7:3.4.8-0ubuntu0.2 \
    libswscale-dev=7:3.4.8-0ubuntu0.2 \
    python-dev=2.7.15~rc1-1 \
    python-numpy=1:1.13.3-2ubuntu1 \
    libtbb2=2017~U7-8 \
    libtbb-dev=2017~U7-8 \
    libpng-dev=1.6.34-1ubuntu0.18.04.2 \
    libtiff-dev=4.0.9-5ubuntu0.3 \
    libjpeg-dev=8c-2ubuntu8 \
    libdc1394-22-dev=2.2.5-1 \
# Python dependencies
    python3=3.6.7-1~18.04 \
    python3-pip=9.0.1-2.3~ubuntu1.18.04.1 \
    python3-dev=3.6.7-1~18.04 \
    python3-virtualenv=15.1.0+ds-1.1 \
    cython3=0.26.1-0.4 \
    tox=2.5.0-1 \
# ONNX dependencies
    git-lfs=2.3.4-1 \
    protobuf-compiler=3.0.0-9.1ubuntu1 \
    libprotobuf-dev=3.0.0-9.1ubuntu1 && \
    apt-get clean autoclean && \
    apt-get autoremove -y

##2 Downloader prerequisites
# Clone model zoo directory and install downloader requirements
RUN git clone https://github.com/openvinotoolkit/open_model_zoo /open_model_zoo
RUN pip3 install networkx==2.4 numpy==1.19.1 wheel==0.34.2
RUN pip3 install -r /open_model_zoo/tools/downloader/requirements.in
RUN pip3 install -r /open_model_zoo/tools/downloader/requirements-pytorch.in
ENV PATH=/open_model_zoo/tools/downloader:$PATH

# Install new version of cmake
RUN wget https://www.cmake.org/files/v3.13/cmake-3.13.3.tar.gz --no-check-certificate && \
    tar xf cmake-3.13.3.tar.gz && \
    (cd cmake-3.13.3 && ./bootstrap --system-curl --parallel=$(nproc --all) && make --jobs=$(nproc --all) && make install) && \
    rm -rf cmake-3.13.3 cmake-3.13.3.tar.gz
ENV PATH=/openvino/bin/intel64/Release:$PATH


##3 Download and install OpenCV
WORKDIR /OpenCV
RUN wget -q https://github.com/opencv/opencv/archive/4.4.0.zip
RUN unzip -qq 4.4.0.zip
WORKDIR /OpenCV/opencv-4.4.0/build
RUN cmake -D CMAKE_BUILD_TYPE=Release ..
RUN make -j$(nproc) install


##4 Clone and build OpenVINO
RUN git clone https://github.com/openvinotoolkit/openvino.git /openvino
WORKDIR /openvino
RUN git submodule init && \
    git submodule update \
        --init \
        --no-fetch \
        --recursive

WORKDIR /openvino/build
ENV D=true
RUN cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_VPU=ON \
    -DENABLE_GNA=OFF \
    -DENABLE_OPENCV=OFF \
    -DENABLE_CPPLINT=OFF \
    -DENABLE_TESTS=OFF \
    -DENABLE_BEH_TESTS=OFF \
    -DENABLE_FUNCTIONAL_TESTS=OFF \
    -DENABLE_MKL_DNN=ON \
    -DENABLE_CLDNN=OFF \
    -DENABLE_PROFILING_ITT=OFF \
    -DENABLE_SAMPLES=ON \
    -DENABLE_SPEECH_DEMO=OFF \
    -DENABLE_PYTHON=OFF \
    -DPYTHON_EXECUTABLE=/usr/bin/python3 \
    -DNGRAPH_ONNX_IMPORT_ENABLE=ON \
    -DNGRAPH_IE_ENABLE=ON \
    -DNGRAPH_INTERPRETER_ENABLE=ON \
    -DNGRAPH_DEBUG_ENABLE=OFF \
    -DNGRAPH_DYNAMIC_COMPONENTS_ENABLE=ON \
    -DCMAKE_INSTALL_PREFIX=/openvino/dist
RUN cmake --build . --config Release -j $(nproc) --target benchmark_app
RUN make -j $(nproc) install

##5 Install Model Optimizer requirements
RUN pip3 install  -r /openvino/model-optimizer/requirements.txt

##6 Download Googlenet-V3-Pytorch in .pth file
RUN downloader.py --name ${model} -o /models/

##7 Convert .pth file to ONNX and IR
RUN converter.py --name ${model} --mo /openvino/model-optimizer/mo.py --download_dir /models/ --precisions=FP32

##8 Run benchmark_app in several modes and save results in /models/ directory:

## ONNX model path (googlenet-v3.onnx):
# /models/public/googlenet-v3-pytorch
## IR model path (googlenet-v3-pytorch.xml):
# /models/public/googlenet-v3-pytorch/FP32

##8 Run benchmark_app for ONNX and IR:
# dldt_cpu32 - CPU in synchronous mode with 1 stream
# results files saved in: /models/
# IR model
WORKDIR /models/synchronous
RUN benchmark_app -m /models/public/${model}/FP32/${model}.xml \
    -api sync -d CPU | tee googlenet-v3-IR-execution.txt
# ONNX model
RUN benchmark_app -m /models/public/${model}/googlenet-v3.onnx \
    -api sync -d CPU | tee googlenet-v3-ONNX-execution.txt

# dldt_cpu32tp - CPU in throughput mode with autodetected number of streams
# results files saved in: /models/
# IR model
WORKDIR /models/asynchronous
RUN benchmark_app -m /models/public/${model}/FP32/${model}.xml \
    -api async -d CPU | tee googlenet-v3-IR-execution.txt
# ONNX model
RUN benchmark_app -m /models/public/${model}/googlenet-v3.onnx \
    -api async -d CPU | tee googlenet-v3-ONNX-execution.txt

# dldt_cpu32tp - CPU in throughput mode with autodetected number of streams
# C++ profiler enabled
# profiled results files saved in: /models/
# IR model
WORKDIR /models/profiling
RUN benchmark_app -m /models/public/${model}/FP32/${model}.xml \
    -api async -d CPU -pc | tee googlenet-v3-async-IR-profiling.txt
# ONNX model
RUN benchmark_app -m /models/public/${model}/googlenet-v3.onnx \
    -api async -d CPU -pc | tee googlenet-v3-async-ONNX-profiling.txt

# dldt_cpu32tp - CPU in throughput mode with autodetected number of streams
# Execution graph dumping
# Exported results files saved in: /models/
# IR model
WORKDIR /models/graphs
RUN benchmark_app -m /models/public/${model}/FP32/${model}.xml \
    -api async -d CPU -exec_graph_path googlenet-v3-async-IR-execution-graph.xml
# ONNX model
RUN benchmark_app -m /models/public/${model}/googlenet-v3.onnx \
    -api async -d CPU -exec_graph_path googlenet-v3-async-ONNX-execution-graph.xml
