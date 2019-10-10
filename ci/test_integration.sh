#!/usr/bin/env bash

. tools/venv/bin/activate
PATH=$(pwd)/utils:${PATH:-}

set -euo pipefail

# test asr recipe
(
    cd ./egs/mini_an4/asr1 || exit 1

    echo "==== ASR (backend=pytorch) ==="
    ./run.sh

    echo "==== ASR (backend=pytorch, dtype=float64) ==="
    ./run.sh --stage 3 \
             --train-config "$(change_yaml.py conf/train.yaml -a train-dtype=float64)" \
             --decode-config "$(change_yaml.py conf/decode.yaml -a api=v2 -a dtype=float64)"

    echo "==== ASR (backend=chainer) ==="
    ./run.sh --stage 3 --backend chainer
)
# test asr_mix recipe
(
    cd ./egs/mini_an4/asr_mix1 || exit 1
    echo "==== ASR Mix (backend=pytorch) ==="
    ./run.sh
)
# test tts recipe
(
    cd ./egs/mini_an4/tts1 || exit 1
    echo "==== TTS (backend=pytorch) ==="
    ./run.sh
)

# TODO(karita): test mt, st?
