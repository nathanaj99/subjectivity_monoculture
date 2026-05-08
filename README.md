# README

**Manuscript title**: The Subjectivity of Monoculture

# 1. Overview
This directory replicates the results from our paper. There are three sub-directories: 1) `data/` contains all the raw and processed data used in the experiments. 2) `results/` stores the results from all methods and experiments conducted. 3) `scripts/` contains all the code relevant to running the methods. 4) `analysis/` contains code relevant to visualizing the results similar to the figures and tables in the paper.



# 2. Code Structure
The following describes the structure of our scripts:
```
scripts
├── experiments
│   ├── irt: Inferring latent parameters of MIRT model, see Appendix C.1 for details
├── analysis
│   ├── fig#.py
├── slurm
│   ├── fig#.sh: set of experiments used to run the corresponding analysis script
```