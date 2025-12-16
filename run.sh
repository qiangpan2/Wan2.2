#https://hub.docker.com/r/rocm/pytorch
docker pull rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.9.1
docker run -itd \
  --name qiang_rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --privileged \
  --ipc=host \
  --network=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v qiang:/workspace \
  -w /workspace \
  rocm/pytorch:rocm7.1_ubuntu24.04_py3.12_pytorch_release_2.9.1 \
  /bin/bash  

export PATH=/opt/ompi/bin:/opt/ucx/bin:/opt/cache/bin:/opt/rocm/llvm/bin:/opt/rocm/opencl/bin:/opt/rocm/hip/bin:/opt/rocm/hcc/bin:/opt/rocm/bin:/opt/conda/envs/py_3.12/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin;

pip install uv
uv pip install modelscope
modelscope download Wan-AI/Wan2.2-I2V-A14B-BF16 --local_dir ./Wan2.2-I2V-A14B
modelscope download Wan-AI/Wan2.2-TI2V-5B --local_dir ./Wan2.2-TI2V-5B --revision bf16

# check weights form model scope
# diffusion_pytorch_model-([0-9]+)-of-([0-9]+)\.safetensors
# diffusion_pytorch_model-$1-of-$2-bf16.safetensors

# MIopen
git clone git@github.com:qiangpan2/rocm-libraries.git --no-checkout --filter=blob:none
cd rocm-libraries
git sparse-checkout init --cone
git sparse-checkout set projects/miopen 
# CK

uv sync --python /opt/venv/bin/python

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

LD_LIBRARY_PATH=${DEPS_PREFIX}/lib:$LD_LIBRARY_PATH \
    PATH=${DEPS_PREFIX}/bin:$PATH \
    which MIOpenDriver

export DEPS_PREFIX="${HOME}/miopen-deps"
export FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
LD_LIBRARY_PATH=${DEPS_PREFIX}/lib:$LD_LIBRARY_PATH \
    HIP_VISIBLE_DEVICES=5 \
    PATH=${DEPS_PREFIX}/bin:$PATH \
    MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 1 -y 1 -x 1 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1 -i 1 > failure.log 2>&1

LD_LIBRARY_PATH=${DEPS_PREFIX}/lib:$LD_LIBRARY_PATH \
    HIP_VISIBLE_DEVICES=5 \
    PATH=${DEPS_PREFIX}/bin:$PATH \
    MIOpenDriver convfp16 -n 1 -c 16 --in_d 5 -H 104 -W 60 -k 16 --fil_d 3 -y 3 -x 3 --pad_d 1 -p 1 -q 1 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1 -i 1 > failure.log 2>&1

#hang case
LD_LIBRARY_PATH=${DEPS_PREFIX}/lib:$LD_LIBRARY_PATH \
    PATH=${DEPS_PREFIX}/bin:$PATH \
    MIOpenDriver convbfp16 -n 1 -c 3 --in_d 3 -H 1106 -W 834 -k 96 --fil_d 3 -y 3 -x 3 --pad_d 0 -p 0 -q 0 --conv_stride_d 1 -u 1 -v 1 --dilation_d 1 -l 1 -j 1 --spatial_dim 3 --in_layout NDHWC --fil_layout NDHWC --out_layout NDHWC -m conv -g 1 -F 1 -t 1 > failure.log 2>&1

# ================================ 8 cards I2V ================================ 
export DEPS_PREFIX="${HOME}/miopen-deps"
export MIOPEN_DEBUG_3D_CONV_IMPLICIT_GEMM_HIP_CHANNEL_LAST_FWD_WMMAOPS=1
    #export MIOPEN_DEBUG_CONV_DIRECT_NAIVE_CONV_FWD=0 #avoid native evaluation time
export MIOPEN_ENABLE_LOGGING=1
export MIOPEN_ENABLE_LOGGING_CMD=1
export MIOPEN_LOG_LEVEL=7
LD_LIBRARY_PATH=${DEPS_PREFIX}/lib:$LD_LIBRARY_PATH \
    PATH=${DEPS_PREFIX}/bin:$PATH \
    uv run torchrun --nproc_per_node=8 generate.py --offload_model True --task i2v-A14B --size 1280*720 --ckpt_dir ./Wan2.2-I2V-A14B --image examples/i2v_input.JPG --dit_fsdp --t5_fsdp --ulysses_size 8 --frame_num 81 --sample_step 1 --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." > output.log 2>&1

#T2V
export LD_LIBRARY_PATH=/workspace/repo/rocm-libraries/projects/miopen/build/lib:$LD_LIBRARY_PATH

#I2V 8 cards 
LD_LIBRARY_PATH=${DEPS_PREFIX}/lib:$LD_LIBRARY_PATH \
    uv run python generate.py --task ti2v-5B --size 1280*704 --ckpt_dir ./Wan2.2-TI2V-5B --offload_model True --convert_model_dtype --image examples/i2v_input.JPG --frame_num 81  --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside." > Wan2.2-TI2V-5B_output.log 2>&1
