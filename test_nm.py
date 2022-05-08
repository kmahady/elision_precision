import needleman_wunsch
import pytest


@pytest.mark.parametrize(
    "test_input,expected",
    [(('GATTACA', "GATT", 1), "GATT---"),
     (('GATT', "GATTACA", 0), "GATT---"),
     (('GATTACA', "GTTACA", 1), "G-TTACA"),
     (('GTTACA', "GATTACA", 0), "G-TTACA"),
     (('dey', "they", 0), "-dey"),
     (('y', "they", 0), "---y"),
     (('t', "they", 0), "t---"),
     ],
)
def test_eval(test_input, expected):
    nm = needleman_wunsch.Needleman_Wunsch()
    output = nm.align_sequences(test_input[0], test_input[1])
    assert [v for v in expected] in [v[test_input[2]] for v in output]

def test_wikipedia_example():
    seq1 = "GCATGCG"
    seq2 = "GATTACA"
    nm = needleman_wunsch.Needleman_Wunsch()
    results = nm.align_sequences(seq1, seq2)
    expected = ([v for v in "GCATG-CG"], [v for v in "G-ATTACA"])
    assert expected in results
