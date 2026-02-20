#!/usr/bin/env bash
scp predict_market/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/RevealedLLM/
scp *.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/RevealedLLM/
scp util_locals/*.py adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/RevealedLLM/util_locals/
scp scripts/slurm/* adidishe@spartan.hpc.unimelb.edu.au:/home/adidishe/RevealedLLM/
