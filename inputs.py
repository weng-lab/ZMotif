import argparse

def get_args():
    parser = argparse.ArgumentParser(description='ZMotif')
    parser.add_argument('-bg', '--bedgraph', help='Bedgraph file', type=str, required=True, default=None)
    parser.add_argument('-g', '--genome', help='Genome FASTA file', type=str, required=True, default=None)
    args = parser.parse_args()
    return args