#!/usr/bin/env bash
scp edgar_log_clean/*.py ADIDISHEIM@vm-172-26-151-140.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/RevealedLLM/
scp *.py ADIDISHEIM@vm-172-26-151-140.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/RevealedLLM/
scp utils_local/*.py ADIDISHEIM@vm-172-26-151-140.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/RevealedLLM/utils_local/
scp scripts/slurm_vm/* ADIDISHEIM@vm-172-26-151-140.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/RevealedLLM/
scp clean/*.py ADIDISHEIM@vm-172-26-151-140.desktop.cloud.unimelb.edu.au:/home/unimelb.edu.au/adidisheim/RevealedLLM/
# scp summary_stat/* ADIDISHEIM@vm-172-26-151-140.desktop.cloud.unimelb.edu.au:/home/adidishe/RevealedLLM/summary_stat/


#scp data/raw/crsp_monthly.csv adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/RevealedLLM/data/raw/
