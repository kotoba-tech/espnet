#!/bin/bash

set -e

# module load
source /etc/profile.d/modules.sh
module load cuda/11.8/11.8.0
module load cudnn/8.9/8.9.2
module load nccl/2.16/2.16.2-1
module load hpcx/2.12

# activate python
cd /groups/gcd50698/fujii/work/espnet/
source .env/bin/activate

pip install espnet
pip install openai==20230308

# torch torchaudio
pip install -r requirements.txt

pip install packaging

# logging packaging
pip install wandb tensorboard

# visualizer
pip install matplotlib
