#https://hub.docker.com/r/rocm/pytorch
docker pull rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 

pip install uv
uv pip install modelscope
modelscope download Wan-AI/Wan2.2-I2V-A14B-BF16 --local_dir ./Wan2.2-I2V-A14B

uv sync --python /opt/conda/envs/py_3.12/bin/python

# ================================ 8 cards I2V ================================ 
uv run torchrun --nproc_per_node=8 generate.py --offload_model True --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 81 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." > output.log 2>&1
