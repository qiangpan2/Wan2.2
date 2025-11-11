#https://hub.docker.com/r/rocm/pytorch
docker pull rocm/pytorch:rocm6.4.4_ubuntu24.04_py3.12_pytorch_release_2.7.1 

pip install uv
uv pip install modelscope
modelscope download Wan-AI/Wan2.2-I2V-A14B-BF16 --local_dir ./Wan2.2-I2V-A14B
modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B --revision bf16

# check weights form model scope
# diffusion_pytorch_model-([0-9]+)-of-([0-9]+)\.safetensors
# diffusion_pytorch_model-$1-of-$2-bf16.safetensors

# MIopen
git clone --no-checkout --filter=blob:none
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/miopen 
# CK

uv sync --python /opt/conda/envs/py_3.12/bin/python

#rccl check
export NCCL_DEBUG=INFO
python -c "import torch; print(torch.distributed.is_nccl_available())"
LD_LIBRARY_PATH=/workspace/repo/rccl/build/debug/rccl_deps/lib:$LD_LIBRARY_PATH torchrun --nproc_per_node=8 test_nccl.py >  test_nccl.log 2>&1


find -name "ring_flashinfer_attn.py"

#Profile
apt install rocprofiler-systems
source /opt/rocm/share/rocprofiler-systems/setup-env.sh
rocprof-sys-avail -G rocprof-sys.cfg
uv run  python test_profile.py > test_profile.log 2>&1

# 8 cards I2V
LD_LIBRARY_PATH=/workspace/repo/rocm-libraries/projects/miopen/build/lib:\
    /workspace/repo/rccl/build/debug/rccl_deps/lib:$LD_LIBRARY_PATH \
    uv run torchrun --nproc_per_node=4 generate.py --offload_model True --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 4 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." > output.log 2>&1

#T2V
LD_LIBRARY_PATH=/workspace/repo/rocm-libraries/projects/miopen/build/lib:$LD_LIBRARY_PATH uv run python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --t5_cpu --image examples/i2v_input.JPG --frame_num 3  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." > Wan2.2-TI2V-5B_output.log 2>&1
