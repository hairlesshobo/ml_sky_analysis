#!/bin/bash

docker inspect msa > /dev/null

if [ $? -ne 0 ]; then
    docker build -t msa .
fi

docker run -it -u $(id -u):$(id -g) -v $(pwd):/app --gpus all --rm msa:latest python3 train.py
