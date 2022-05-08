import nltk
import numpy as np
import re
from elision_precision.elision_precision import needleman_wunsch
from itertools import combinations
from collections import defaultdict

"""
Example
-------
>>> from phonetic_distance import phonetic_distance
>>> aligner = phonetic_distance.needleman_wunsch.Needleman_Wunsch()
>>> WPSM = phonetic_distance.WPSM_Matrix()
>>> metrics = phonetic_distance.Phonetic_Distance(WPSM.logodds_mtx, WPSM.unique_phonemes)
Get the pronunciations of a word. You can manually input these, or just use the lookup stored in 
the WPSM_Matrix class
>>> p1=WPSM.arpabet_stressless['tomato'][0]
>>> p1
['T', 'AH', 'M', 'EY', 'T', 'OW']
>>> p2=WPSM.arpabet_stressless['tomato'][1]
>>> p2
['T', 'AH', 'M', 'AA', 'T', 'OW']
Align the sequences (unnecessary in this case)
>>> aligned = aligner.align_sequences(p1, p2)
Get the MIR diffence between the zero element alignment between the two
>>> metrics.get_MIR(aligned[0][0], aligned[0][1])
0.8254594147120822

"""
class Phonetic_Distance:
    """
    Class for the calculation of the phonetic distance between sequences of phonemes
    represented by ARPABET.

    Parameters
    ----------
    logodds_mtx : np.array
        The WPSM matrix 
    unique_phonemes : list of strings
        The unique phonemes whose substitutions are represented by the WPSM matrix
    """
    def __init__(self, logodds_mtx, unique_phonemes):
        self.logodds_mtx = logodds_mtx
        self.unique_phonemes = unique_phonemes
    def get_score_list(self, w1, w2):
        """ Calculates the similarity score the sounds in pronunciations w1 and w1.

        w1 and w2 must be aligned, so that they are the same length.

        Parameters
        ----------
        w1 : list of str
            A pronunciation of a word. Each str must be a stressless phoneme in ARPABET
        w2 : list of str
            A pronunciation of a word. Each str must be a stressless phoneme in ARPABET

        Returns
        -------
        List of floats
            The similarity for each sound in w1 and w2
        """
        scores=[]
        for i in range(len(w1)):
            idx1 = self.unique_phonemes.index(w1[i])
            idx2 = self.unique_phonemes.index(w2[i])
            scores.append(self.logodds_mtx[idx1, idx2])
        return scores
    def get_score(self, w1, w2):
        """ Calculates the similarity score between the pronunciations w1 and w1.

        w1 and w2 must be aligned, so that they are the same length.

        Parameters
        ----------
        w1 : list of str
            A pronunciation of a word. Each str must be a stressless phoneme in ARPABET
        w2 : list of str
            A pronunciation of a word. Each str must be a stressless phoneme in ARPABET

        Returns
        -------
        List of floats
            The similarity between w1 and w2.
        """
        scores=[]
        for i in range(len(w1)):
            idx1 = self.unique_phonemes.index(w1[i])
            idx2 = self.unique_phonemes.index(w2[i])
            scores.append(self.logodds_mtx[idx1, idx2])
        return np.mean(scores)
    def get_MIR(self, w1, w2):
        """ Calculates the MIR (Mean Identity Ratio) score between the pronunciations w1 and w1.

        w1 and w2 must be aligned, so that they are the same length.

        Parameters
        ----------
        w1 : list of str
            A pronunciation of a word. Each str must be a stressless phoneme in ARPABET
        w2 : list of str
            A pronunciation of a word. Each str must be a stressless phoneme in ARPABET

        Returns
        -------
        List of floats
            The MIR between w1 and w2.
        """
        score0 = self.get_score(w1, w1)
        score1 = self.get_score(w1,w2)
        return score1/score0

class WPSM_Matrix:
    """
    Calculates the Weighted Phoneme Substitution Matrix (WPSM).

    This is based on the paper by Hixon et al (1). The computation differs somewhat from
    Hixon et al., giving a somewhat different matrix.

    (1) Hixon, Ben, Eric Schneider, and Susan L. Epstein. "Phonemic similarity metrics
    to compare pronunciation methods."Twelfth Annual Conference of the
    International Speech Communication Association. 2011.

    BLOSUM Computation: http://www.cs.cmu.edu/~durand/03-711/2010/Lectures/blosum10.pdf
    """
    def __init__(self):
        self.arpabet_stressless = self._preprocess_cmu()
        # get those with multiple pronunciations
        self.arpabet_multiple = {key:val for key,val in self.arpabet_stressless.items() if len(val)>1}
        self.unique_phonemes = self._get_phonemes(self.arpabet_multiple)
        self.all_pairs = self._align_multiple_pronunciations(self.arpabet_multiple)
        self.observed_frequency = self._generate_substitution_matrix_fast(self.unique_phonemes, self.all_pairs)
        self.Exx, self.Exy = self._generate_expected_matrix(self.unique_phonemes, self.all_pairs)
        self.logodds_mtx = self._generate_logodds_mtx(self.unique_phonemes, self.observed_frequency, 
                                                     self.Exx, self.Exy)
    def _preprocess_cmu(self):
        """ Preprocesses the CMU_Dict to remove non-alpha entries, and to remove stresses
        """
        arpabet = nltk.corpus.cmudict.dict()
        # get only the words with letter headwords
        arpabet_alpha = {key:val for key,val in arpabet.items() if key.isalpha()}
        # remove all stresses
        arpabet_stressless={key:[remove_stress_arpabet(v) for v in val] for key,val in arpabet_alpha.items()}
        return arpabet_stressless

    def _get_phonemes(self, arpabet):
        """ Create a list of the unique phonemes in cmudict
        """
        # extract phonemes
        all_phonemes=[v for key in arpabet.keys() for w in arpabet[key] for v in w]
        # get the unique, in-order phonemes
        unique_phonemes=sorted(list(set([v for key in arpabet.keys() for w in arpabet[key] for v in w])))
        # add the indel "phoneme"
        unique_phonemes.append('-')
        return unique_phonemes
    def _align_multiple_pronunciations(self, arpabet):
        """Align all multiple pronunciations in arpabet
        """
        print("Generating phonetic alignments (this may take a minute)")
        aligner = needleman_wunsch.Needleman_Wunsch()
        all_pairs = defaultdict(list)
        # using my needleman wunsch
        # NLTK has its own, but I'm having trouble adapting it to my use case
        for word,pronun in arpabet.items():
            for pairs in list(combinations(pronun, 2)):
                for alignment in aligner.align_sequences(pairs[0], pairs[1]):
                    all_pairs[word].append(alignment)
        return all_pairs
    def _generate_substitution_matrix_fast(self, unique_phonemes, all_pairs):
        """ Creates the matrix of substitutions
        """
        # Faster implementation, but less readable and harder to adapt to other uses
        ffrequency_matrices = []
        nbs = []
        for key,val in all_pairs.items():
            for pair in val:
                kb = 2
                kbi=1
                nb = len(pair[0])
                nbs.append(nb)
                Cb = 2
                frequency_matrix = np.zeros((len(unique_phonemes), len(unique_phonemes)))
                p0=pair[0]
                p1=pair[1]
                idxsp0=[unique_phonemes.index(v) for v in p0]
                idxsp1=[unique_phonemes.index(v) for v in p1]
                for l in range(len(idxsp0)):
                    idx1 = idxsp0[l]
                    idx2 = idxsp1[l]
                    frequency_matrix[idx1, idx2] +=1
                    frequency_matrix[idx2, idx1] +=1
                ffrequency_matrices.append(frequency_matrix)
        sum_freq=np.zeros_like(frequency_matrix)
        for M in ffrequency_matrices:
            sum_freq += M
        # Using needleman wunsch, it is impossible for indel to substitute itself.
        # Therefore, we will set is frequency to be average
        sum_freq[-1,-1] = np.mean(sum_freq.diagonal())
        return sum_freq/sum(nbs)
    def generate_subsitution_matrix(self, unique_phonemes, all_pairs):
        """ Generates the matrix of substitutions (slower implementation)
        """
        print("Generating substitution matrix (this may take a few minutes")
        # first pass: only take one pair from each word
        frequency_matrices = []
        nbs = []
        for key,val in all_pairs.items():
            for pair in val:
                kb = 2
                kbi=1
                nb = len(pair[0])
                nbs.append(nb)
                Cb = 2
                frequency_matrix = np.zeros((len(unique_phonemes), len(unique_phonemes)))
                for idx1, ph1 in enumerate(unique_phonemes):
                    for idx2, ph2 in enumerate(unique_phonemes):
                        for l in range(nb):
                            substitution_count = (pair[0][l]==ph1)*(pair[1][l]==ph2)+(pair[1][l]==ph1)*(pair[0][l]==ph2)
                            frequency_matrix[idx1, idx2] += substitution_count/kbi/kbi
                frequency_matrices.append(frequency_matrix)
        sum_freq=np.zeros_like(frequency_matrix)
        for M in frequency_matrices:
            sum_freq += M
        return sum_freq/sum(nbs)
    
    def _generate_expected_matrix(self, unique_phonemes, all_pairs):
        """ Generate the matrix of the expected frequencies of phonemes
        """
        expected_matrices = []
        nbs = []
        Cbs = []
        for key,val in all_pairs.items():
            for pair in val:
                kb = 2
                nb = len(pair[0])
                kbi=1
                nbs.append(nb)
                Cb = 2
                Cbs.append(Cb)
                frequency_matrix = np.zeros(len(unique_phonemes))
                for idx1, ph1 in enumerate(unique_phonemes):
                    for i in range(2):
                        for l in range(nb):
                            substitution_count = (pair[i][l]==ph1)
                            frequency_matrix[idx1] += substitution_count/kbi
                expected_matrices.append(frequency_matrix)
        sum_exp=np.zeros_like(frequency_matrix)
        for M in expected_matrices:
            sum_exp += M
        expected_freq=sum_exp/sum(np.array(nbs)*np.array(Cbs))
        expected_freq=expected_freq
        px = expected_freq
        py = np.transpose(expected_freq)
        pxy = np.outer(px, py)
        pyx = np.outer(py, px)
        Exy = pxy+pyx
        Exx = px**2
        return Exx, Exy

    def _generate_logodds_mtx(self, unique_phonemes, observed_freq, Exx, Exy):
        mtx_final = np.zeros_like(Exy)
        for i, ph1 in enumerate(unique_phonemes):
            for j, ph2 in enumerate(unique_phonemes):
                if i==j:
                    mtx_final[i,j] = observed_freq[i,j]/Exx[i]
                else:
                    mtx_final[i,j] = observed_freq[i,j]/Exy[i,j]
        # Force the zero values to be the smallest nonzero value (to avoid -infs)
        mtx_final[mtx_final==0] = np.min(mtx_final[mtx_final>0])
        return np.log(mtx_final)


def get_sorted_phonemes():
    """ This is a convenience method, which just groups phonemes based on some basic characteristics.
    """
    vowels = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"]
    labial_consonants = ["B", "P", "F", "V"]
    dental = ["D", "T", "DH", "TH", "S", "Z"]
    liquids = ["W", "L", 'R', 'Y']
    palatals = ["JH", "CH","SH", "ZH", "G", "K", 'HH']
    nasals = ["M", "N", "NG"]
    deleted = ["-"]
    sorted_phonemes = vowels+labial_consonants+nasals+dental+liquids+palatals+deleted
    return sorted_phonemes

def remove_stress_arpabet(phon_list):
    """ Removes the stress markers in a sequence of phonemes in CMU Dict

    Parameters
    ----------
    phon_list : list 
        List whose entries are arpabet phonemes (with numerical stress)
    Returns
    -------
    list
        List whose entries are the arpabet phonemes (without numerical stress)
    """
    return [re.sub(r'[^a-zA-Z]', '', w) for w in phon_list]
