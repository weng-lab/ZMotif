#!/bin/bash

ACC=$1
source /opt/conda/etc/profile.d/conda.sh

conda activate zmotif
python motif_finder_from_bg.py -i $ACC.input.bed -g /home/andrewsg/genome/hg38/hg38.fa -o $ACC -e 10 -c 1 -k 1 -w 20 -l1 0.0 -l 200
conda deactivate 
