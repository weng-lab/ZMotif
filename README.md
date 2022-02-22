### Arguments

The following are arguments for zmotif.py:
* `-pos_fasta pos_fasta`. FASTA file containing positive sequences (required).
* `-bg bedGraph`. BedGraph file containing regions and signal / weights.
* `-bed bed`. Bed file containing regions of interest.
* `-chrom_sizes chrom_sizes`. Chromosome sizes file.
* `-g genome`. GENOME FASTA file
* `-flank flank`. Length of adjacent regions to draw negative regions (used as "-b" flag in bedtools flank).
* `-motif_db motif_db`. Motif database in MEME format to compare discovered motifs.
* `-neg_fasta neg_fasta`. FASTA file containing negative sequences.
* `-o output_prefix`. Prefix of output files.
* `-e epochs`. Number of training epochs.
* `-n n_motifs`. Max number of motifs to discover (equal to number of convolution kernels).
* `-seed seed_motif`. Motif to seed convolution kernel in MEME format.
* `-seed_db seed_database`. Motif database in MEME format to seed convolution kernels.
* `-b batch_size`. Batch size.
* `-noise gaussian noise`. Gaussian noise.
