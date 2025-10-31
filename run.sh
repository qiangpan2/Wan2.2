docker pull rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1

pip install uv
uv pip install modelscope
modelscope download Wan-AI/Wan2.2-I2V-A14B --local_dir ./Wan2.2-I2V-A14B
# check weights form model scope
# diffusion_pytorch_model-([0-9]+)-of-([0-9]+)\.safetensors
# diffusion_pytorch_model-$1-of-$2-bf16.safetensors

#triton
# cd /var/lib/jenkins/triton/python
# uv pip install build
# python -m build --wheel

#apex apex@file:///var/lib/jenkins/apex

# uv pip install \
#     amdsmi@file:///opt/rocm-6.4.4/share/amd_smi \
#     torch@file:///var/lib/jenkins/pytorch/dist/torch-2.7.1+git99ccf24-cp312-cp312-linux_x86_64.whl \
#     flash_attn@file:///workspace/repo/flash-attention/dist/flash_attn-2.8.3-py3-none-any.whl \
#     triton@file:///var/lib/jenkins/triton/python/dist/triton-3.3.1-cp312-cp312-linux_x86_64.whl

uv pip install -e . --system


#rccl check
export NCCL_DEBUG=INFO
python -c "import torch; print(torch.distributed.is_nccl_available())"
LD_LIBRARY_PATH=/workspace/repo/rccl/build/release/rccl_deps/lib:$LD_LIBRARY_PATH torchrun --nproc_per_node=8 test_nccl.py >  test_nccl.log 2>&1 


find -name "ring_flashinfer_attn.py"

LD_LIBRARY_PATH=/workspace/repo/rccl/build/release/rccl_deps/lib:$LD_LIBRARY_PATH torchrun --nproc_per_node=8 generate.py --offload_model True --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." > output.log 2>&1