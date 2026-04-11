# Auto-Regressive Soft Q (ARSQ)

This is the official implementation of the paper [Learning from Suboptimal Data in Continuous Control via Auto-Regressive Soft Q-Network](https://arxiv.org/abs/2502.00288).

## D4RL

### Installation

```bash
cd d4rl 

conda create -n arsq_d4rl -y python=3.10
conda activate arsq_d4rl

conda install -y ffmpeg -c conda-forge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install hydra-core wandb scikit-video matplotlib
pip install -e .

mkdir -p thirdparty
# d4rl
cd thirdparty
git clone https://github.com/Farama-Foundation/d4rl.git && cd d4rl && git checkout 68afbc3c640cf17c7ef07a5352bf26902278fffe
pip install -e .
# mujoco
cd ..
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xf mujoco210-linux-x86_64.tar.gz

sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
conda install -c conda-forge -y libstdcxx-ng
pip install "cython<3" "numpy<2" patchelf

export MUJOCO_PY_MUJOCO_PATH=$(pwd)/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/mujoco210/bin
export D4RL_DATASET_DIR=$(pwd)/datasets
conda env config vars set MUJOCO_PY_MUJOCO_PATH=$MUJOCO_PY_MUJOCO_PATH
conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH
conda env config vars set D4RL_DATASET_DIR=$D4RL_DATASET_DIR

# reactive conda env
conda deactivate
conda activate arsq_d4rl
```

### Run

check `d4rl/arsq_d4rl/cfgs/env_cfg` for all tasks.

```bash
# arsq
CUDA_VISIBLE_DEVICES=0 python arsq_d4rl/run/sqar_run.py \
 env_cfg@_global_="hmr" \
 wandb.name="arsq-hmr" \
 seed=0

# cqn
CUDA_VISIBLE_DEVICES=0 python arsq_d4rl/run/cqn_run.py \
 env_cfg@_global_="hmr" \
 wandb.name="cqn-hmr" \
 seed=0
```

## RLBench

### Installation

```bash
cd rlbench

conda create -n arsq_rlb -y python=3.10
conda activate arsq_rlb

conda install -y pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install imageio==2.34.1 imageio_ffmpeg==0.5.1 hydra-core hydra-submitit-launcher opencv-python-headless==4.10.0.82 \
  tensorboard dm_env==1.6 dm_control==1.0.20 gymnasium==0.29.1 numpy==1.26.4 wandb
pip install -e .

source set_env.sh
  
# Coppelia sim
# download CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz from https://github.com/stepjam/PyRep
mkdir -p thirdparty
cd thirdparty
tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz

# Install RLBench
git clone https://github.com/stepjam/RLBench
cd RLBench
git checkout b80e51feb3694d9959cb8c0408cd385001b01382
pip install .
```

Follow [RLBench](https://github.com/stepjam/RLBench) installation instructions to setup a virtual display.

### Run

check `rlbench/arsq_rlb/cfgs/rlbench_task` for all tasks.

```bash
source set_env.sh

task="open_oven"

cd thirdparty/RLBench/rlbench
CUDA_VISIBLE_DEVICES=0 DISPLAY=:99 python dataset_generator.py --save_path=${CQN_PATH}/arsq_rlb/exp_local/dataset \
 --image_size 84 84 --renderer opengl3 --episodes_per_task 100 --variations 1 --processes 1 \
 --tasks "${task}" --arm_max_velocity 2.0 --arm_max_acceleration 8.0

cd ${CQN_PATH}
CUDA_VISIBLE_DEVICES=0 DISPLAY=:99.0 python arsq_rlb/runner/train_rlbench_sqar.py \
   dataset_root=${CQN_PATH}/arsq_rlb/exp_local/dataset \
   rlbench_task=${task} \
   wandb.name="arsq-${task}" \
   seed=0
```
