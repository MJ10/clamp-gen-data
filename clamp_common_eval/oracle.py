from typing import Sequence

class Oracle:
    def __call__(self, sequence: str) -> float:
        """Returns a scalar corresponding to the oracle prediction for `sequence`

        sequence: a str of the amino acids, e.g "AYRIPKSRHPWTCPRR"
        """
        raise NotImplementedError()

    def evaluate_many(self, sequences: Sequence[str]) -> Sequence[float]:
        """Evaluates the oracle for many sequences. Call this if possible to
        take advantage of batching. See __call__"""
        raise NotImplementedError()

    def to(self, device):
        """If applicable, sends this oracle to `device`"""
        pass
