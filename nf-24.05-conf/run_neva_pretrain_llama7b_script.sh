# ref: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron/gpt/gpt_training.html#train-gpt-style-model

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"
wandb login ${WANDB}

DATASET="158k"
JOB_ID="0001"
NAME="NeVA-llama7b-finetue-fp8-${DATASET}_dataset-${JOB_ID}"

RESULTS="${WORK_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

python /opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_pretrain.py \
    --config-path=/workspace/data/mm/nf-24.05-conf \
    --config-name=neva_config-7b \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}