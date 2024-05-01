#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00


set -e  # exit on error.
echo "Date:     $(date)"
echo "Hostname: $(hostname)"

module --quiet purge
#module --quiet load anaconda/3

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/cvmfs/ai.mila.quebec/apps/x86_64/debian/anaconda/3/lib


source  ~/.virtualenvs/crewai/bin/activate
echo "Starting at "`date`
huggingface-cli login --token "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"
cp final_gpt_answer_extraction_SciQ_original.py final_gpt_answer_extraction_3.py
sed -i "s/XXXXX/result_round_3_before_science_analysis/" final_gpt_answer_extraction_3.py
python -u final_gpt_answer_extraction_3.py  > result_round_3_after_gpt_analysis.out
echo "Ending at "`date`





