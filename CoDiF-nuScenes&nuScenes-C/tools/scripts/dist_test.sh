# #!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

while true
do
    #PORT=47576 # 47576 34512
    PORT=$((RANDOM % (65000 - 34001) + 34001))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
CUDA_VISIBLE_DEVICES='1,2,3' \
python -m torch.distributed.launch --nproc_per_node=${NGPUS}  --rdzv_endpoint=localhost:${PORT} test.py --launcher pytorch ${PY_ARGS}
# #0,1,2
