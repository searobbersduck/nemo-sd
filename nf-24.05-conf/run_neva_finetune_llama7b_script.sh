# ref: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/nlp/nemo_megatron/gpt/gpt_training.html#train-gpt-style-model

rm nemo_experiments -rf

WANDB="1ee66e27d1e97b6018dda9793bd6cccac7d988bc"
WANDB_PROJECT="NeVA-llama7b-pretrain"
wandb login ${WANDB}

DATASET="158k"
JOB_ID="0001"
NAME="NeVA-llama7b-finetue-fp8-${DATASET}_dataset-${JOB_ID}"

WORK_DIR="/workspace/data/mm/neva/7b"
WORK_EXP_DIR="${WORK_DIR}/${NAME}"
RESULTS="${WORK_EXP_DIR}/results_${NAME}"
mkdir -p ${RESULTS}

NSYS_DIR="${WORK_EXP_DIR}/timeline"
mkdir -p ${NSYS_DIR}

TP_SIZE=$1
PP_SIZE=$2


TAP_LAUNCH = "TAP_WARMUP_STEPS=10 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_NVTX=pytorch,apex,python \
    TAP_RECORD_SHAPES=true TAP_SAVE_DIR=${NSYS_DIR} \
    TAP_CAPTURE_GRAPH=false TAP_EXIT_ON_STOP=true TAP_PROFILE_MEMORY=true TAP_BACKWARD_NVTX=true \
    TAP_NSIGHT_FLAGS='--trace cuda,nvtx,osrt,cudnn,cublas --gpu-metrics-devices=all' "


# ${TAP_LAUNCH} python /opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_finetune.py \
#     --config-path=/workspace/data/mm/nf-24.05-conf \
#     --config-name=neva_finetune-7b \
#     model.tensor_model_parallel_size=${TP_SIZE} \
#     model.pipeline_model_parallel_size=${PP_SIZE} \
#     model.data.data_path="/workspace/data/mm/LLaVA-Instruct-150K/llava_instruct_150k.json" \
#     exp_manager.explicit_log_dir=${RESULTS} \
#     exp_manager.create_wandb_logger=True \
#     exp_manager.wandb_logger_kwargs.name=${NAME} \
#     exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}


TAP_WARMUP_STEPS=10 TAP_ACTIVE_STEPS=1 TAP_MODE=nsight TAP_NVTX=pytorch,apex,python \
    TAP_RECORD_SHAPES=true TAP_SAVE_DIR=${NSYS_DIR} \
    TAP_CAPTURE_GRAPH=false TAP_EXIT_ON_STOP=true TAP_PROFILE_MEMORY=true TAP_BACKWARD_NVTX=true \
    TAP_NSIGHT_FLAGS="--trace cuda,nvtx,osrt,cudnn,cublass --nic-metrics=true" \
    python /opt/NeMo/examples/multimodal/multimodal_llm/neva/neva_finetune.py \
    --config-path=/workspace/data/mm/nf-24.05-conf \
    --config-name=neva_finetune-7b \
    model.tensor_model_parallel_size=${TP_SIZE} \
    model.pipeline_model_parallel_size=${PP_SIZE} \
    model.data.data_path="/workspace/data/mm/LLaVA-Instruct-150K/llava_instruct_150k.json" \
    exp_manager.explicit_log_dir=${RESULTS} \
    exp_manager.create_wandb_logger=True \
    exp_manager.wandb_logger_kwargs.name=${NAME} \
    exp_manager.wandb_logger_kwargs.project=${WANDB_PROJECT}

chmod 777 -R ./tmp