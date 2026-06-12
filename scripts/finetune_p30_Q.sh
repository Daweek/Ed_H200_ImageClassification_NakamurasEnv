#!/bin/bash
#PBS -l rt_QF=2
#PBS -N finet_p30
#PBS -l walltime=10:00:00
#PBS -W group_list=qgai50157
#PBS -j oe
#PBS -V
#PBS -koed
#PBS -o output/
#PBS -l USE_SSH=1
#PBS -v SSH_PORT=2299,ALLOW_GROUP_SSH=1

# =========== Configuration =========================================================
source $HOME/utils/main_config.sh

cecho gray "Re load the main modules for this repo..."
source /etc/profile.d/modules.sh
module purge

module load openmpi/4.1.7 cuda/12.6/12.6.2 cudnn/9.8/9.8.0 nccl/2.24/2.24.3
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
NUM_GPU_PER_NODE=4
NUM_NODES=2
NUM_GPUS=$((${NUM_NODES} * ${NUM_GPU_PER_NODE}))

echo "NUM_GPUS: ${NUM_GPUS}"
echo "NUM_GPU_PER_NODE: ${NUM_GPU_PER_NODE}"
echo "NUM_NODES: ${NUM_NODES}"

export MASTER_ADDR=$(/usr/sbin/ip a show dev enp5s0f1 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1|head -n 1)
export MASTER_PORT=$((10000 + ($JOB_ID % 50000)))
    echo "MASTER_ADDR: ${MASTER_ADDR}"
    echo "MASTER_PORT: ${MASTER_PORT}"


# =========== Create Output Directory and Experiments ARGS ============================================
export HYDRA_FULL_ERROR=1

# ========== Log The script job before running it
export OUTPUT_DIR=$PBS_O_WORKDIR/output
# Print this file script
SCRIPT_LAUNCHED=$(realpath "$0")
cecho blue "Script launched: $SCRIPT_LAUNCHED"
# Copy the script that was launched to the output directory
SCRIPT_NAME=$(basename "$0")
cecho blue "Copied script that was launched: $SCRIPT_LAUNCHED"
cp "$SCRIPT_LAUNCHED" "$OUTPUT_DIR/${JOB_ID}_ran.sh"

# =========== Start of Job =========================================================
cecho blue "##### START - Whole work ##############"
cecho blue "______Start Computing_________"
cecho blue "Job Started on: $(date)"
t_start_time=$(date +%s%3N)

# which mpirun
mpirun --version
# which mpiexec
mpiexec --version

###################################### Untar to SSD
export SSD=$PBS_LOCALDIR
    echo "LOCAL_SSD: ${SSD}"

# Finte tune dataset data
FT_DATASET_NAME=places30
NUM_CLS=30
TRAIN_IMG=149254
VAL_IMG=3000

blank_lines 2
cecho red "Copy and Untar..."
# --display-map --display-allocation
mpiexec  -npernode 1 -np 2 -hostfile $PBS_NODEFILE --use-hwthread-cpus --bind-to none --oversubscribe tar -xf /home/qai10413uh/dataset/${FT_DATASET_NAME}.tar -C $SSD
readlink -f ${SSD}/${FT_DATASET_NAME}
ls ${SSD}/${FT_DATASET_NAME} 
cecho red "Finished copying and Untar..."
###########################################################

# iter=0
# =========== Start of Job =========================================================
for iter in {0..0}; do
    blank_lines 2
    echo "JOBID=$PBS_JOBID"
    echo "HOST=$(hostname)"
    echo "USER=$(whoami)"
    echo "PBS_NODEFILE=$PBS_NODEFILE"

    echo "Allocated nodes:"
    cat "$PBS_NODEFILE" | sort | uniq -c

   

    blank_lines 4
    cecho orange "##### START - Iteration $((iter + 1)) ##############"
    cecho orange "______Start Computing_________"
    cecho orange "Job Started on: $(date)"
    start_time=$(date +%s%3N)

    # Calculate current JOB_ID by incrementing
    CURRENT_JOB_ID=$((JOB_ID + iter))

    # export PRT_DATASET='VA1k'
    export PRT_DATASET='1pF'
    export EXPERIMENT=${CURRENT_JOB_ID}_${FT_DATASET_NAME}_${PRT_DATASET}_Org_Q_Random

    # export CKPT=/home/qai10413uh/working/transformer/nakamura/Ed_H200_ImageClassification_NakamurasEnv/checkpoints/fromABCI_tofinetune/pretrain_deit_tiny_va1k_lr1.0e-3_epochs300_bs1024_files_512x_VAconfig_V_last.pth.tar

    export CKPT=/home/qai10413uh/working/transformer/nakamura/Ed_H200_ImageClassification_NakamurasEnv/checkpoints/fromABCI_tofinetune/vit_tiny_patch16_224_sigma3.5_delta0.1_sample1000_80000ep.pth
    # If the number of samples used for pre-training is 1k
    mpiexec --display-map --display-allocation -npernode ${NUM_GPU_PER_NODE} -np ${NUM_GPUS} -hostfile $PBS_NODEFILE --use-hwthread-cpus --bind-to none --oversubscribe python -B main.py \
        data=colorimagefolder data.baseinfo.name=${FT_DATASET_NAME} data.baseinfo.train_imgs=${TRAIN_IMG} data.baseinfo.val_imgs=${VAL_IMG} data.baseinfo.num_classes=${NUM_CLS} \
        data.trainset.root=$SSD/${FT_DATASET_NAME}/train data.valset.root=$SSD/${FT_DATASET_NAME}/val \
        data.loader.batch_size=96 model=vit model.arch.model_name=deit_tiny_patch16_224 epochs=1000  \
        model.optim.opt=sgd model.optim.lr=0.01 model.optim.weight_decay=1.0e-4 \
        model.scheduler.args.warmup_epochs=10 \
        logger.save_epoch_freq=100 \
        logger.group=${EXPERIMENT} \
        ckpt=${CKPT} \
        output_dir=./checkpoints/${EXPERIMENT} \
        mode=finetune +script=pth +logwandb=True seed=-1 +prtdataset=${PRT_DATASET}
        
    # =========== End of RUNNING ============================================================

    blank_lines 4

    end_time=$(date +%s%3N)
    total_duration=$((end_time - start_time))

    cecho orange "This experiment Duration: "
    convert_milliseconds "$total_duration"
    cecho orange "JOB ID: ------- >>>>>>  $CURRENT_JOB_ID"
    cecho red    "Hostname:------ >>>>>>  $(hostname)"
    cecho bold magenta "Experiment: $EXPFULL_EXPERIMENT"
    cecho orange "Job finished on: $(date) Iteration: $((iter + 1)) "
    cecho orange "______Finish_________"
    echo "                   "

 done   


t_end_time=$(date +%s%3N)
t_total_duration=$((t_end_time - t_start_time))
cecho blue "This experiment Duration: "
convert_milliseconds "$t_total_duration"
cecho red    "Hostname:------ >>>>>>  $(hostname)"
cecho blue "Job finished on: $(date)"
cecho blue "______Finish_________"
echo "                           "
# =========== End of Job ========================================================
## Debuggin purposes???
cecho cyan "code=$?"
