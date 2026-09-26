#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT license.

FROM ubuntu:jammy

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git make cmake g++ libaio-dev libgoogle-perftools-dev libunwind-dev clang-format libboost-dev libboost-program-options-dev libmkl-full-dev libcpprest-dev python3.10

RUN mkdir -p /app/BatchANN
WORKDIR /app/BatchANN