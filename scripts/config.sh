#!/bin/bash
# Configuration file to define variables
# =========== Print Blank Function ==================================================
blank_lines() {
    local n=${1:-1}
    for ((i=0; i<n; i++)); do
        echo " "
    done
}

# =========== Color Echo Function ==================================================

cecho() {
    # Usage: cecho [bold] COLOR MESSAGE
    local bold=""
    local color=""
    local color_code=""
    local reset='\033[0m'

    # Check for bold argument
    if [[ "$1" == "bold" ]]; then
        bold='\033[1m'
        shift
    fi

    color="$1"
    shift

    case "$color" in
        black)        color_code='\033[0;30m' ;;
        red)          color_code='\033[0;31m' ;;
        green)        color_code='\033[0;32m' ;;
        yellow)       color_code='\033[0;33m' ;;
        blue)         color_code='\033[0;34m' ;;
        magenta)      color_code='\033[0;35m' ;;
        cyan)         color_code='\033[0;36m' ;;
        white)        color_code='\033[0;37m' ;;
        orange)       color_code='\033[38;5;208m' ;;
        gray)         color_code='\033[0;90m' ;;
        light_red)    color_code='\033[1;31m' ;;
        light_green)  color_code='\033[1;32m' ;;
        light_yellow) color_code='\033[1;33m' ;;
        light_blue)   color_code='\033[1;34m' ;;
        light_magenta)color_code='\033[1;35m' ;;
        light_cyan)   color_code='\033[1;36m' ;;
        light_white)  color_code='\033[1;37m' ;;
        reset|*)      color_code="$reset" ;;
    esac

    echo -e "${bold}${color_code}$*${reset}"

## Usage examples:
# cecho blue "Normal blue text"
# cecho bold red "Bold red text"
# cecho bold orange "Bold orange text"
# cecho light_green "Light green text"
# cecho bold light_magenta "Bold light magenta text"
}

# Modules to run on ABCI
# ======== Modules ========
# cecho gray "Include main ABCI modules for 3.0 .."
# source /etc/profile.d/modules.sh
# module purge

####### MPI
# module load hpcx-mt/2.20
# module load hpcx-mt/2.26
# Load CUSTOM OpenMPI with CUDA support
# export PATH=$HOME/apps/openmpi/bin:$PATH
# module load cuda/12.8/12.8.1 cudnn/9.5/9.5.1 nccl/2.25/2.25.1-1 
# module load cuda/12.9/12.9.1 cudnn/9.12/9.12.0 nccl/2.28/2.28.3-1
# module load cuda/13.0/13.0.1 cudnn/9.13/9.13.0 nccl/2.28/2.28.3-1 
# module load cuda/12.8/12.8.1 cudnn/9.18/9.18.1 nccl/2.28/2.28.3-1
# module load cuda/13.2/13.2.1 cudnn/9.21/9.21.1 nccl/2.30/2.30.4-1 

# Module for building Megatron test - Python 3.13.10 - CUDA 13.0
# module load hpcx-mt/2.26 cuda/13.0/13.0.1 cudnn/9.21/9.21.1 nccl/2.30/2.30.4-1

# ======== Pyenv/ ========
cecho gray "Include main python environment..."
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init - bash)"
eval "$(pyenv virtualenv-init -)"


export PYTHONUNBUFFERED=1
export PYTHONWARNINGS="ignore"

## Debug ON
# ============ Debug on Python version and Packages ============================
# blank_lines 2
# python --version
# python -c "import torch; print(torch.__version__)"
# blank_lines 2
# Print all environment variables using cecho gray
# blank_lines 2
# env | grep PBS 
# blank_lines 2
# qstat -f
# Extra routines 
cecho gray "Including extra util routines to measure time..."

convert_milliseconds() {
    local total_ms=$1

    # Calculate total seconds and remaining milliseconds
    local total_seconds=$(echo "$total_ms / 1000" | bc)
    local remaining_ms=$((total_ms % 1000))  # Remaining milliseconds

    # Calculate days, hours, minutes, and seconds using bc
    local days=$(echo "$total_seconds / 86400" | bc)
    local hours=$(echo "($total_seconds % 86400) / 3600" | bc)
    local minutes=$(echo "($total_seconds % 3600) / 60" | bc)
    local seconds=$(echo "$total_seconds % 60" | bc)

    # Print in D:H:M:S.ms format using cecho red
    cecho magenta "\t${days} days, ${hours} hours, ${minutes} minutes, ${seconds} seconds, ${remaining_ms} milliseconds"
}

