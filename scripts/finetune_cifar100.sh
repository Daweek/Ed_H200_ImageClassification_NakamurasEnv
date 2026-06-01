#!/bin/bash
#PBS -q rt_HF
#PBS -l select=1:ncpus=192:ngpus=8:mpiprocs=192
#PBS -N finet_c100
#PBS -l walltime=01:30:00
#PBS -P gaf51130
#PBS -j oe
#PBS -V
#PBS -koed
#PBS -o output/
#PBS -v USE_SSH=1

# =========== Configuration =========================================================
source $HOME/utils/main_config.sh

cecho gray "Re load the main modules for this repo.."
source /etc/profile.d/modules.sh
module purge

module load hpcx-mt/2.20 cuda/12.1/12.1.1 cudnn/9.21/9.21.1 nccl/2.30/2.30.4-1
eval "$(pyenv init - bash)"
pyenv local 3.11.1

# =========== Debug on packages and others ==========================================
blank_lines 2
env | grep PBS 
blank_lines 2
qstat -f

# ========= Get local Directory ======================================================
if [  "$PBS_O_WORKDIR" = "$HOME" ]; then
    cecho gray "PBS_O_WORKDIR is set to HOME. Using current directory."
    PBS_O_WORKDIR=$(pwd)
else
    cecho gray "PBS_O_WORKDIR is set to: $PBS_O_WORKDIR"
fi

cecho gray "Changing to working directory: $PBS_O_WORKDIR"
cd $PBS_O_WORKDIR

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
export HYDRA_FULL_ERROR=1

# =========== Start of Job =========================================================
blank_lines 4
cecho orange "##### START ##############"
cecho orange "______Start Computing_________"
cecho orange "Job Started on: $(date)"
start_time=$(date +%s%3N)

# ========== Log The script job before running it
export OUTPUT_DIR=$PBS_O_WORKDIR/output
# Print this file script
SCRIPT_LAUNCHED=$(realpath "$0")
cecho blue "Script launched: $SCRIPT_LAUNCHED"
# Copy the script that was launched to the output directory
SCRIPT_NAME=$(basename "$0")
cecho blue "Copied script that was launched: $SCRIPT_LAUNCHED"
cp "$SCRIPT_LAUNCHED" "$OUTPUT_DIR/${JOB_ID}_ran.sh"


###################################### Untar to SSD

FT_DATASET_NAME=cifar100
NUM_CLS=100
TRAIN_IMG=50000
VAL_IMG=10000

export SSD=$PBS_LOCALDIR
    echo "LOCAL_SSD: ${SSD}"

echo "Copy and Untar..."
tar -xf /groups/gaf51130/dataset/${FT_DATASET_NAME}.tar -C $SSD
readlink -f ${SSD}/${FT_DATASET_NAME}
ls ${SSD}/${FT_DATASET_NAME} 
echo "Finished copying and Untar..."


###################################### Untar to SSD
# export PRT_DATASET='VA1k'
export PRT_DATASET='1pF'
export EXPERIMENT=${JOB_ID}_${FT_DATASET_NAME}_${PRT_DATASET}_Org_1pF_H_Random
# Mine pre-trained using the Original VA dataset
# export CKPT=/home/acc12930pb/working/transformer/beforedali_timm_main_sora/checkpoint/tiny/va1k/pre_training/pretrain_deit_tiny_va1k_lr1.0e-3_epochs300_bs1024_files_512x_VAconfig_V/last.pth.tar
# Original from SORA
# export CKPT=/home/acc12930pb/working/transformer/nakamura/Ed_H200_ImageClassification_NakamurasEnv/checkpoints/original_fromSora/vit_tiny_with_visualatom_1k.pth.tar
# Original from 1p_Fractal
 export CKPT=/home/acc12930pb/working/transformer/nakamura/Ed_H200_ImageClassification_NakamurasEnv/checkpoints/org_1pFractal/vit_tiny_patch16_224_sigma3.5_delta0.1_sample1000_80000ep.pth

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
    mode=finetune +script=pth +logwandb=True seed=-1
    
# =========== End of RUNNING ============================================================

blank_lines 4

end_time=$(date +%s%3N)
total_duration=$((end_time - start_time))

cecho orange "This experiment Duration: "
convert_milliseconds "$total_duration"
cecho orange "JOB ID: ------- >>>>>>  $JOB_ID"
cecho red    "Hostname:------ >>>>>>  $(hostname)"
cecho bold magenta "Experiment: $EXPERIMENT"
cecho orange "Job finished on: $(date)"
cecho orange "______Finish_________"
echo "                   "

# =========== End of Job ========================================================
## Debuggin purposes???
cecho cyan "code=$?"
