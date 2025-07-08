# conda create -n vllm python=3.10 -y
conda init
conda activate vllm

git clone https://github.com/vllm-project/vllm.git && cd vllm

sudo pip uninstall torch torch-xla -y

pip install -r requirements/tpu.txt
sudo apt-get install --no-install-recommends --yes libopenblas-base libopenmpi-dev libomp-dev

VLLM_TARGET_DEVICE="tpu" python -m pip install -e .