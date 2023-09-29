# instilling-inductiva-bias

This is the repository accompanying the submission "Instilling Inductive Bias with Subnetworks"

## Requirements

All experiments are performed on various number of 3090's with CUDA 11.7 and Python 3.9. Packages in the environment can be found at `requirements.txt`.

## Navigating the Repository

Code related to our arithmetic experiments can be found in `math/`, and those related to our vision experiments are in `vision/`. Experiment configs and logs are saved in the `checkpoints/` directory of each experiment folder. Note that the core method relies on `NeuroSurgeon`, for which a modified version is included in this repo.

All of our scripts can be ran by calling the script and supplying a `.yaml` config file. Example config files can be found in the `configs/` directory of each experiment folder.

## Reproducing Results

Subnetwork discovery and subnetwork transfer are done by two different scripts, because one would often want to tune the first before moving on to the second. `ablate.py`, `transfer.py`, `vision_ablate.py`, and `vision_transfer.py` handles these respectively. Note that the vision scripts are capable of handling both ViT and ResNet18 automatically. For space reasons we cannot release all checkpoints, but our released checkpoints folder contains all log files, configs, and random seed used for each experiment.

All of our experiments fix random seed for Numpy and PyTorch rngs. We do so in a way without losing randomness: when each script is called, if `reproduce` is set to `False` in the config file, a true random number is taken from the system's entropy pool and used to seed the prngs. This seed is recorded in the `seed.txt` file of the output directory for each checkpoint. Therefore running a script and supplying it with the config file in the output folder except where `reproduce` is set to `True` and `random_seed` is replaced by the seed in `seed.txt` will reproduce the results. 
