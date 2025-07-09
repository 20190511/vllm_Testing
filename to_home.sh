#!/bin/bash
DEST=".."

cp -r -- * "$DEST/"
cd ..
git clone https://github.com/20190511/vllm.git
cd vllm
sudo docker build -f docker/Dockerfile.tpu -t vllm-tpu .