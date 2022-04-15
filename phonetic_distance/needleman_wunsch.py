import numpy as np

class node:
    def __init__(self, pos, parent):
        self.parent=parent
        self.children = []
        self.pos = pos

class Needleman_Wunsch():
    """ Class to implement the Needleman Wunsch algorith to align two character sequences

    Example
    -------
    >>> from phonetic_distance import needleman_wunsch
    >>> seq1 = 'GCATGCG'
    >>> seq2 = 'GATTACA'
    >>> nm = needleman_wunsch.Needleman_Wunsch()
    >>> result = nm.align(seq1, seq2)
    >>> result
    ... [('GCATG-CG', 'G-ATTACA'), ('GCAT-GCG', 'G-ATTACA'), ('GCA-TGCG', 'G-ATTACA')]
    """
    def __init__(self):
        self.score_mismatch = -1
        self.score_indel = -1
        self.score_match = 1

    def align_sequences(self, seq1, seq2):
        """ Function to align two sequences

        Parameters
        ----------
        seq1 : string
        seq2 : string

        Returns
        -------
        List of tuples
            Each entry is a possible alignment of seq1 and seq2. 
        """
        source_mtx, score_mtx = self._compute_sequence_matrix(seq1, seq2)
        seqs = self._get_all_sequences(score_mtx, source_mtx)
        pairs = []
        for seq in seqs:
            r, c = self._get_match(seq, seq1, seq2)
            #r = "".join(r)
            #c = "".join(c)
            pairs.append((c,r))
        return pairs

    def _compute_sequence_matrix(self, seq1, seq2):
        source_mtx = dict()
        score_mtx = np.zeros((1+len(seq2), 1+len(seq1)))
        score_mtx[0, :] = -1*np.arange(len(seq1)+1)
        score_mtx[:, 0] = -1*np.arange(len(seq2)+1)
        index_directions = [(-1,0), (0,-1), (-1,-1)]
        # Initialize sources along each axis
        for i in range(1,score_mtx.shape[0]):
            source_mtx[i,0]=[(-1,0)]
        for i in range(1,score_mtx.shape[1]):
            source_mtx[0,i]=[(0,-1)]

        # Compute the actual matrix and source
        for i in range(1,score_mtx.shape[0]):
            for j in range(1,score_mtx.shape[1]):

                score_top = score_mtx[i-1, j] + self.score_indel
                score_left = score_mtx[i, j-1] + self.score_indel
                if seq1[j-1]==seq2[i-1]:
                    score_diag = score_mtx[i-1, j-1]+self.score_match
                else:
                    score_diag = score_mtx[i-1, j-1]+self.score_mismatch
                scores = [score_top, score_left, score_diag]
                best_score_dex = (np.argwhere(np.array(scores)==max(scores))).flatten()

                best_score_loc = [index_directions[v] for v in best_score_dex]
                best_score = np.max(scores)
                source_mtx[i,j] = best_score_loc
                score_mtx[i,j] = best_score
                
        return source_mtx, score_mtx

    # get all paths
    def _get_next_steps(self, parent, source_mtx):
        i,j = parent.pos
        if i==0 and j==0:
            return
        next_step = source_mtx[i,j]
        for v in next_step:
            parent.children.append(node((i+v[0], j+v[1]), parent))
        for child in parent.children:
            self._get_next_steps(child, source_mtx)
            
    # get all leaf nodes
    def _get_leaf(self, node, leaflist):
        if node.children==[]:
            leaflist.append(node)
        else:
            for v in node.children:
                self._get_leaf(v, leaflist)

    def _get_all_sequences(self, score_mtx, source_mtx):
        bottom_node = node((score_mtx.shape[0]-1, score_mtx.shape[1]-1), None)

        self._get_next_steps(bottom_node, source_mtx)

        leaflist=[]
        self._get_leaf(bottom_node, leaflist )

        seqs = []
        for leaf in leaflist:
            this_seq=[]
            thisnode = leaf
            while thisnode is not None:
                this_seq.append(thisnode.pos)
                thisnode = thisnode.parent
            seqs.append(this_seq)
        return seqs

    def _get_match(self, this_seq, seq1, seq2):
        row_word = []
        column_word = []
        for i in range(1, len(this_seq)):
            this_source = this_seq[i][0]-this_seq[i-1][0], this_seq[i][1]-this_seq[i-1][1]
            if this_source==(1,0):
                row_word.append('-')
                column_word.append(seq2[this_seq[i][0]-1])
            elif this_source==(0,1):
                row_word.append(seq1[this_seq[i][1]-1])
                column_word.append('-')
            else:
                row_word.append(seq1[this_seq[i][1]-1])
                column_word.append(seq2[this_seq[i][0]-1])
        return column_word, row_word
