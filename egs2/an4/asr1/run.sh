#!/bin/bash
#$ -l rt_AF=1
#$ -l h_rt=5:00:00
#$ -j y
#$ -o outputs/
#$ -cwd

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

# change directory
cd /groups/gcd50698/fujii/work/espnet/egs2/an4/asr1

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

export PATH=$PATH:/groups/gcd50698/fujii/work/espnet/egs2/an4/asr1/sph2pipe

./asr.sh \
    --lang en \
    --asr_config conf/train_asr_transformer.yaml \
    --inference_config conf/decode_asr.yaml \
    --lm_config conf/train_lm.yaml \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --train_set train_nodev \
    --valid_set train_dev \
    --test_sets "train_dev test" \
    --bpe_train_text "dump/raw/train_nodev_sp/text" \
    --lm_train_text "data/train_nodev_sp/text" "$@"
