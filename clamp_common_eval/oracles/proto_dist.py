import sys
import os
import numpy as np
from collections import defaultdict

import multiprocessing as mp
import warnings
from Bio.Blast.Applications import NcbiblastpCommandline
from io import StringIO
from Bio.Blast import NCBIXML
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio import pairwise2
from Bio.Application import ApplicationError
import random
import string
from tqdm import tqdm

from polyleven import levenshtein
import time
from Bio.Application import _Option, AbstractCommandline, _Switch
sys.path.append("..")
from ..oracle import Oracle

MEDOIDS_PATH = {
    "D1_target": os.path.join(os.path.split(__file__)[0], '../../data/dataset', 'pos_D1_target_medoids.txt'),
    "D1_cluster": os.path.join(os.path.split(__file__)[0], '../../data/dataset', 'pos_D1_cluster_medoids.txt'),
    "D1_title": os.path.join(os.path.split(__file__)[0], '../../data/dataset', 'pos_D1_title_medoids.txt'),
    "D2_target": os.path.join(os.path.split(__file__)[0], '../../data/dataset', 'pos_D1_target_medoids.txt'),
    "D2_cluster": os.path.join(os.path.split(__file__)[0], '../../data/dataset', 'pos_D1_cluster_medoids.txt'),
    "D2_title": os.path.join(os.path.split(__file__)[0], '../../data/dataset', 'pos_D1_title_medoids.txt')
}

class DiamondCommandline(AbstractCommandline):
    """Base Commandline object for (new) Diamond wrappers (PRIVATE).
    This is provided for subclassing, it deals with shared options
    common to all the diamond makedb and blastp tools.
    """

    def __init__(self, cmd="diamond", **kwargs):
        assert cmd is not None
        self.parameters= [
            # Core:
            _Switch(
                ["help", "h"],
                "Print USAGE, DESCRIPTION and ARGUMENTS description; "
                "ignore other arguments.",
            ),
            _Switch(
                ["blastp", "bp"],
                "use blastp.",
            ),
            _Switch(
                ["makedb", "createdb"],
                "use makedb .",
            ),
            _Switch(
                ["--sensitive", "sensitive"],
                "Use sensitive mode.",
            ),
            _Switch(
                ["--more-sensitive", "more_sensitive"],
                " Use more sensitive mode.",
            ),
            _Switch(
                ["--ignore-warnings", "ignore_warnings"],
                "Ignore bad sequences.",
            ),
            # Output configuration options
            _Option(
                ["--query", "query"],
                "The sequence to search with.",
                filename=True,
                equate=False,
            ),  # Should this be required?
            _Option(
                ["--in", "infile"],
                "input fasta for db.",
                filename=True,
                equate=False,
            ),  # Should this be required?
            _Option(
                ["--out", "out"],
                "Output file for alignment.",
                filename=True,
                equate=False,
            ),
            _Option(
                ["--db", "db"],
                "Output file for alignment.",
                filename=True,
                equate=False,
            ),
            # Formatting options:
            _Option(
                ["-f", "outfmt"],
                "Alignment view.  Typically an integer 0-14 but for some "
                "formats can be named columns like 'BLAST tabular.  "
                "Use 5 for XML output. ",
                filename=False,  # to ensure spaced inputs are quoted
                equate=False,
            ),
            _Option(
                ["-p", "threads"],
                "Number of threads",
                filename=False,
                equate=False,
            ),
        ]
        AbstractCommandline.__init__(self, cmd, **kwargs)

    def _validate_incompatibilities(self, incompatibles):
        """Validate parameters for incompatibilities (PRIVATE).
        Used by the _validate method.
        """
        for a in incompatibles:
            if self._get_parameter(a):
                for b in incompatibles[a]:
                    if self._get_parameter(b):
                        raise ValueError("Options %s and %s are incompatible." % (a, b))

class MedioidDistanceOracle(Oracle):
    """
    Random Forest Classifier trained on T5 features
    Methods
    ----------
    __call__() -> list
        get the list of the prediction_proba

    evaluate_many() -> list
        get the list of the prediction_proba
    """

    __data_split__ = None

    def __init__(self, split="D1_target", dist_fn=None, norm_constant=35,
                 #base_path=os.environ['SLURM_TMPDIR'],
                 base_path=f"/dev/shm/{os.environ['USER']}/clamp/",
                 exp_constant=6):
        self.dist_fn = dist_fn
        self.medoids = self._load_medoids(MEDOIDS_PATH[split])
        self.norm_constant = norm_constant
        self.base_path = base_path
        self.exp_constant = exp_constant
        self.batch_calls = False

    def _load_medoids(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()
            medoids = [line.strip() for line in lines]
        if self.dist_fn != "edit":
            self._save_medoids(medoids)
        return medoids

    def _save_medoids(self, medoids, base_path=f"/dev/shm/{os.environ['USER']}/clamp/"):
        recs = []
        os.makedirs(base_path, exist_ok=True)
        for i, medoid in enumerate(medoids):
            recs.append(SeqRecord(Seq(medoid),
                            id=str(i)))
        seq2_path = os.path.join(base_path, "seq2.fasta")
        SeqIO.write(recs, seq2_path, "fasta")
        db_name = "seq2"

        DiamondCommandline(createdb=True, infile=seq2_path,
                           db=os.path.join(base_path, db_name))()

    def _dist_fn(self, seq, base_path=f"/dev/shm/{os.environ['USER']}/clamp/"):
        seq1 = SeqRecord(Seq(seq),
                    id="seq1")
        seq_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(10))
        seq1_path = os.path.join(base_path, seq_id + "seq1.fasta")
        SeqIO.write(seq1, seq1_path, "fasta")
        start = time.time()
        try:
            output = DiamondCommandline(bp=True, query=seq1_path,
                                        db=os.path.join(base_path, "seq2"),
                                        outfmt="6 bitscore", threads=1)()[0]
        except ApplicationError:
            output = None
        end = time.time()
        os.remove(seq1_path)
        print(" time use ", end - start, output)
        try:
            #dist = float(str(output).split('\t')[10])
            dist = 1 / (float(output) + 1e-3)
        except:
            dist = 1e6
        return dist

    def _edit_dist(self, seq, base_path=None):
        dists = [levenshtein(seq, med) / max(len(seq), len(med)) for med in self.medoids]
        return np.min(dists)

    def _dist_fn_batch(self, seqs):
        #+1 in mode
        recs = [SeqRecord(Seq(s), id=str(i)) for i, s in enumerate(seqs)]
        seq_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for i in range(10))
        seq1_path = os.path.join(self.base_path, seq_id + "seq1.fasta")
        SeqIO.write(recs, seq1_path, "fasta")
        output = DiamondCommandline(bp=True, query=seq1_path,
                                    db=os.path.join(self.base_path, "seq2"),
                                    outfmt="6 bitscore qseqid sseqid", threads=4,
                                    ignore_warnings=True)()[0]
        # Each protein might have multiple hits, let's track all of them
        hits = defaultdict(list)
        for i in output.splitlines():
            bitscore, query, subject = map(eval,i.split('\t'))
            hits[query].append((bitscore, subject))

        # Then for each sequence in seqs, find the best hit (and its
        # subject/target sequence/mode). Return a bitscore of 1 and a
        # mode of None if there are no hits for i.
        scores, modes = zip(*[sorted(hits[i] or [(1, None)])[-1] for i in range(len(seqs))])
        # Normalize. I read somewhere on the internet that bitscore of above 50 is
        # significant, should maybe check if this makes sense.
        scores = np.float32(scores) / 50
        # modes + 1 because mode 0 or None is reserved to be compatible with the
        # real oracle setup
        modes = [i+1 if i is not None else i for i in modes]
        return scores, modes

    def __call__(self, s, eval_uncertainty = False):
        if eval_uncertainty:
            warnings.warn('Uncertainty evaluation is not support for RandomForest Classifier', UserWarning)
        # TODO: Ideally this dist_fn should be thread_safe so we can compute these distances in parallel.
        # distances = [self.dist_fn(s, s_) for s_ in self.medoids]
        # exp_arg = min(distances) / self.norm_constant
        #
        # return np.exp(-exp_arg)
        return np.exp(-1 * self._edit_dist(s) / self.norm_constant)

    def evaluate_many(self, sequences, eval_uncertainty = False, progress=False):
        if eval_uncertainty:
            warnings.warn('Uncertainty evaluation is not support for RandomForest Classifier', UserWarning)
        if self.batch_calls:
            scores, modes = self._dist_fn_batch(sequences)
            scores = np.exp(np.log(scores) / self.norm_constant)
            return scores, modes
        dists = []
        for sequence in tqdm(sequences, disable=not progress, leave=False):
            dists.append(self.__call__(sequence))
        return np.array(dists)
