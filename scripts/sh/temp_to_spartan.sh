#!/usr/bin/env bash
scp data/wsj_tar/* adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/RevealedLLM/data/wsj_tar/
scp data//p/load_icf_complete.p adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/RevealedLLM/data//p/
scp data//raw/ff5.csv adidishe@spartan.hpc.unimelb.edu.au:/data/gpfs/projects/punim2039/RevealedLLM/data/raw/ff5.csv



echo "code pushed:"
date | cut -d " " -f5 | cut -d ":" -f1-2