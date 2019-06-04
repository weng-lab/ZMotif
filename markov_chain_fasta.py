from pyfaidx import Fasta
from collections import defaultdict, Counter
import random
import numpy as np

class  MarkovFasta:
    def __init__(self, fasta_file, k = 1):
        self.fasta_file = fasta_file
        self.fasta = Fasta(fasta_file, sequence_always_upper=True, as_raw=True)
        self.k = k
        self.model = defaultdict(Counter)
        self.kmer_counts = Counter()
        self.uniqe_kmers = []
        self.kmer_freqs = []
        self._construct_markov_model()

    def _count_kmers(self):
        print("Counting {}-mers in {}".format(self.k, self.fasta_file))
        for seq in self.fasta:
            temp_seq = seq[:]
            for i in range(len(temp_seq) - self.k):
                self.kmer_counts[temp_seq[i:i+self.k]] += 1
        #print(self.kmer_counts)
        #print(list(self.kmer_counts))
        print("There are {} total {}-mers".format(sum(self.kmer_counts.values()), self.k))
        self.total_kmers = sum(self.kmer_counts.values())
        self.unique_kmers = list(self.kmer_counts)
        self.kmer_freqs = [self.kmer_counts[kmer] / self.total_kmers for kmer in self.unique_kmers]
        
    def _construct_markov_model(self):
        self._count_kmers()
        print("Learning Markov model of order " + str(self.k))
        for seq in self.fasta:
            temp_seq = seq[:]
            for i in range(len(temp_seq) - self.k):
                state = temp_seq[i:i + self.k]
                next = temp_seq[i + self.k]
                self.model[state][next] += 1

        #print(self.model)
        self.states = list(self.model.keys())
        #print(self.model['A'])
        #print([self.model['A'][next_state] for next_state in self.states])
        #print(np.random.choice(self.states, 
        #                       p = [self.model['A'][next_state] for next_state in self.states]/np.sum([self.model['A'][next_state] for next_state in self.states]))) 
        
    def _next_state(self, current_state):
        return(np.random.choice(self.states, 
                               p = [self.model[current_state][next_state] for next_state in self.states]/np.sum([self.model[current_state][next_state] for next_state in self.states])))
    
    def sample(self, seq_len):
        #state = random.choice(list(self.model))
        #print("state:", state, sep=" ")
        current_state = np.random.choice(self.unique_kmers, 1, p = self.kmer_freqs)[0]
        out = list(current_state)
        for j in range(seq_len - self.k):
            out.extend(self._next_state(current_state))
            current_state = current_state[1:] + out[-1]
        return "".join(out)
        
# markov_model = MarkovFasta("371.pos.fasta")
# print(markov_model.sample(100))
# print(markov_model.sample(100))

