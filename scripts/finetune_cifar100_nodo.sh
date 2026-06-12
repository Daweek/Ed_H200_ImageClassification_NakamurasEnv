#!/bin/bash
# To save everything on the 
echo "========================================"
echo "Started: $(date)"
echo "PID: $$"
echo "Command: $0 $*"
echo "========================================"

logfile="output/${$}.nodo.OU"
exec >"$logfile" 2>&1

echo "Script contents:"
echo "----------------------------------------"
cat "$0"
echo "----------------------------------------"

# =========== Configuration =========================================================
source $HOME/utils/main_config.sh

cecho gray "Re load the main modules for this repo..."
source /etc/profile.d/modules.sh
module purge

module load hpcx-mt/2.20 cuda/12.1/12.1.1 cudnn/9.21/9.21.1 nccl/2.30/2.30.4-1

eval "$(pyenv init - bash)"
pyenv local 3.11.1

# ============ Debug on Python version and Packages ============================
blank_lines 2
python --version
python -c "import torch; print(torch.__version__)"
blank_lines 2

# =========== Basic Information =========================================================
blank_lines 2
cecho green "Job started on: $(date)"
cecho green "ABCI 3.0 ............................................................"
JOB_ID=$(echo "${PBS_JOBID:-$$}"  | cut -d '.' -f 1)
cecho green "JOB ID: ------- >>>>>>  $JOB_ID"
cecho red   "Hostname:------ >>>>>>  $(hostname)"

# ========== For MPI
NUM_GPU_PER_NODE=8
NUM_NODES=1
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"
echo "NUM_NODES: ${NUM_NODES}"


export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"


# =========== Create Output Directory and Experiments ARGS ============================================
blank_lines 2

export HYDRA_FULL_ERROR=1

# =========== Start of Job =========================================================
for iter in {0..0}; do
    blank_lines 4
    cecho orange "##### START - Iteration $((iter + 1)) ##############"
    cecho orange "______Start Computing_________"
    cecho orange "Job Started on: $(date)"
    start_time=$(date +%s%3N)
    
    # Calculate current JOB_ID by incrementing
    CURRENT_JOB_ID=$((JOB_ID + iter))

###################################### Untar to SSD
    # Finte tune dataset data
    FT_DATASET_NAME=cifar100
    NUM_CLS=100
    TRAIN_IMG=50000
    VAL_IMG=10000
    export SSD='/local/acc12930pb'

    # export PRT_DATASET='VA1k'
    # export PRT_DATASET='1pF'
    # export EXPERIMENT=${JOB_ID}_${FT_DATASET_NAME}_${PRT_DATASET}_Org_H_Random
    # From Ferran
    export PRT_DATASET='VA1k'
    export EXPERIMENT=${CURRENT_JOB_ID}_${FT_DATASET_NAME}_${PRT_DATASET}_Ferran_SoraOldStack_H_Random

    # Mine pre-trained using the Original VA dataset
    # export CKPT=/home/acc12930pb/working/transformer/beforedali_timm_main_sora/checkpoint/tiny/va1k/pre_training/pretrain_deit_tiny_va1k_lr1.0e-3_epochs300_bs1024_files_512x_VAconfig_V/last.pth.tar
    # Original from SORA
    # export CKPT=/home/acc12930pb/working/transformer/nakamura/Ed_H200_ImageClassification_NakamurasEnv/checkpoints/original_fromSora/vit_tiny_with_visualatom_1k.pth.tar
    # Original from 1p_Fractal
    # export CKPT=/home/acc12930pb/working/transformer/nakamura/Ed_H200_ImageClassification_NakamurasEnv/checkpoints/org_1pFractal/vit_tiny_patch16_224_sigma3.5_delta0.1_sample1000_80000ep.pth
    # Original from Ferran
    export CKPT=/home/acc12930pb/working/transformer/nakamura/Ed_H200_ImageClassification_NakamurasEnv/checkpoints/fromFerran/SoraOldStack.pth.tar

    # If the number of samples used for pre-training is 1k
    mpirun -np ${NUM_GPUS} --use-hwthread-cpus --bind-to socket --oversubscribe -mca pml ob1 -mca btl self,vader -x MASTER_ADDR=${MASTER_ADDR} -x MASTER_PORT=${MASTER_PORT} python -B main.py \
        data=colorimagefolder data.baseinfo.name=${FT_DATASET_NAME} data.baseinfo.train_imgs=${TRAIN_IMG} data.baseinfo.val_imgs=${VAL_IMG} data.baseinfo.num_classes=${NUM_CLS} \
        data.trainset.root=$SSD/${FT_DATASET_NAME}/train data.valset.root=$SSD/${FT_DATASET_NAME}/val \
        data.loader.batch_size=96 model=vit model.arch.model_name=deit_tiny_patch16_224 epochs=1000  \
        model.optim.opt=sgd model.optim.lr=0.01 model.optim.weight_decay=1.0e-4 \
        model.scheduler.args.warmup_epochs=10 \
        logger.save_epoch_freq=100 \
        logger.group=${EXPERIMENT} \
        ckpt=${CKPT} \
        output_dir=./checkpoints/${EXPERIMENT} \
        mode=finetune +script=tar +logwandb=True seed=-1 +prtdataset=${PRT_DATASET}
    
# =========== End of RUNNING ============================================================

    blank_lines 4

    end_time=$(date +%s%3N)
    total_duration=$((end_time - start_time))

    cecho orange "This experiment Duration: "
    convert_milliseconds "$total_duration"
    cecho orange "JOB ID: ------- >>>>>>  $CURRENT_JOB_ID"
    cecho red    "Hostname:------ >>>>>>  $(hostname)"
    cecho bold magenta "Experiment: $EXPERIMENT"
    cecho orange "Job finished on: $(date)"
    cecho orange "______Finish_________"
    echo "                   "
done

# =========== End of Job ========================================================
## Debuggin purposes???
cecho cyan "code=$?"
