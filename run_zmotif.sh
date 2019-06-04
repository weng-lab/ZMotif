#!/bin/bash
set -e
source /home/ga42w/anaconda3/etc/profile.d/conda.sh

ACC="$1"
BASE_DIR=/project/umw_zhiping_weng/andrewsg/motif_finder/cistrome
DATA_DIR=/project/umw_zhiping_weng/andrewsg/data/cistrome/human_factor
#TMP_DIR=/tmp/andrewsg/$ACC
PEAK_FILE=$DATA_DIR/"$ACC"_sort_peaks.narrowPeak.bed
RESULTS_DIR=$BASE_DIR/results/$ACC
SCRIPTS_DIR=$BASE_DIR/scripts

rm -rf $RESULTS_DIR
mkdir $RESULTS_DIR
cd $RESULTS_DIR

conda activate bedtools 
cat $PEAK_FILE | cut -f1-3 | \
    awk 'BEGIN{OFS="\t"; OFMT = "%.0f"}{seqlen=$3-$2; center=$2+seqlen/2; summit=$2+$10; if (seqlen > 500) {print $1, center-250, center+250} else {print $1, $2, $3}}' | \
    sort -k1,1 -k2,2n | uniq | awk '$2 > 0' > $ACC.pos.bed
cat $PEAK_FILE | cut -f1-3 | \
    awk 'BEGIN{OFS="\t"; OFMT = "%.0f"}{seqlen=$3-$2; center=$2+seqlen/2; summit=$2+$10; if (seqlen > 500) {print $1, $2-500, $2} else {print $1, $2-seqlen, $2}}' | \
    sort -k1,1 -k2,2n | uniq  > $ACC.left.neg.bed
cat $PEAK_FILE | cut -f1-3 | \
    awk 'BEGIN{OFS="\t"; OFMT = "%.0f"}{seqlen=$3-$2; center=$2+seqlen/2; if (seqlen > 500) {print $1, $3, $3+500} else {print $1, $3, $3+seqlen}}' | \
    sort -k1,1 -k2,2n | uniq  > $ACC.right.neg.bed

cat $ACC.left.neg.bed $ACC.right.neg.bed | awk '$2 > 0' | sort -k1,1 -k2,2n | uniq > $ACC.neg.bed
bedtools getfasta -fi /project/umw_zhiping_weng/andrewsg/genome/hg38/hg38.fa -bed $ACC.pos.bed > $ACC.pos.fasta
bedtools getfasta -fi /project/umw_zhiping_weng/andrewsg/genome/hg38/hg38.fa -bed $ACC.neg.bed > $ACC.neg.fasta
conda deactivate

rm $ACC.pos.bed
rm $ACC.neg.bed
rm $ACC.left.neg.bed
rm $ACC.right.neg.bed

conda activate zmotif
PYTHONHASHSEED=12 python $SCRIPTS_DIR/motif_finder_from_fasta.py -p $ACC.pos.fasta -n $ACC.neg.fasta -o $ACC -e 2000 -c 1 -k 16 -w 48 -aug 100
#python $SCRIPTS_DIR/motif_finder_from_fasta.py -p $ACC.pos.fasta -n $ACC.neg.fasta -o $ACC -e 10 -c 1 -k 32 -w 32 -aug 100
conda deactivate 

conda activate meme
tomtom $ACC.raw.meme $BASE_DIR/HOCOMOCOv11_full_HUMAN_mono_meme_format.meme --text > $ACC.hoc.tomtom.out
tomtom $ACC.raw.meme $BASE_DIR/JASPAR2018_CORE_vertebrates_redundant_pfms_meme.txt --text > $ACC.jaspar.tomtom.out
conda deactivate 

python $SCRIPTS_DIR/get_json.py $ACC.raw.meme $ACC.hoc.tomtom.out $ACC.jaspar.tomtom.out $ACC

rm $ACC.pos.fasta
rm $ACC.neg.fasta
