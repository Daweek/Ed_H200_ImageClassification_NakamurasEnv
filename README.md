# Repository dedicated to reproduce results on FDSL datasets.

We tested this repository on ABCI and ABCI-Q.

We used Python 3.11.1 using Pyenv (https://github.com/pyenv/pyenv)

## Modules

For ABCI we used the following modules:

- hpcx-mt/2.20 cuda/12.1/12.1.1 cudnn/9.21/9.21.1 nccl/2.30/2.30.4-1

For ABCI-Q we used the following modules:

- openmpi/4.1.7 cuda/12.6/12.6.2 cudnn/9.8/9.8.0 nccl/2.24/2.24.3


## PyTorch and other pagckages

We tested the repository using PyTorch __torch==2.1.0+cu121__ in order to run on H100 and H200 GPUs. For more details on the rest of the packages, check __requirements.txt__.

## Scripts

Be aware that for each system there are slighty differences on the scripts. However, the hyperparameters are the same.
We included fine-tune scripts for each dataset. Also, we included ABCI and ABCI-Q (*_Q.sh).

We utilized some bash functions to measure time included in the __main_config.sh__ into __config.sh__ script.

Take also into account in case you want to load (pth or tar) checkpoints.
