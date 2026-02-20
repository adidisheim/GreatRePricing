#!/bin/bash -l
#SBATCH --job-name process_openai_
#SBATCH --output=/home/adidishe/RevealedLLM/out/process_openai_%a.out
#SBATCH --error=/home/adidishe/RevealedLLM/out/process_openai_%a.err
#SBATCH --chdir=/home/adidishe/RevealedLLM
#SBATCH --array=0-30 # 40for lopez v1,v2, 49 for ravenpack

#SBATCH --mail-type=ALL # notifications for job start, end, and failure
#SBATCH --mail-user=antoine.didisheim@unimelb.edu.au

#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --mem 16G
#SBATCH --cpus-per-task 4
#SBATCH --time 24:00:00


module load foss/2022a
module load GCCcore/11.3.0; module load Python/3.10.4
module load  cuDNN/8.4.1.50-CUDA-11.7.0
module load TensorFlow/2.11.0-CUDA-11.7.0-deeplearn
module load OpenMPI/4.1.4; module load PyTorch/1.12.1-CUDA-11.7.0
source  ~/venvs/nlp_gpu/bin/activate

python3 _03_process_data.py ${SLURM_ARRAY_TASK_ID} --parallel=1
