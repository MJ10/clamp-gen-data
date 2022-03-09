from typing import Sequence, Tuple, Optional



class PeptideDataset:
    def get_dataframe(self):
        """Returns this dataset as a pandas dataframe"""
        raise NotImplementedError()

    def sample(self, n: int, r: Optional[float] = 0.5) -> Tuple[Sequence[str], Sequence[float]]:
        """Returns (xs, ys) at random from this dataset's train fold

        xs is a list of n str
        ys is a list of n float labels
        r is the ratio of positive samples (if applicable)

        """
        raise NotImplementedError()

    def test_examples(self) -> Tuple[Sequence[str], Sequence[float]]:
        """Returns all the (xs, ys) from this dataset's test fold"""
        raise NotImplementedError()
