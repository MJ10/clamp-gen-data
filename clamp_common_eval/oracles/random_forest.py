import sys
import os
import numpy as np
import pickle
import gzip
import torch
from . import CTDD_fast, PortTrans_fast, PortTrans_loader

import multiprocessing as mp
import warnings
sys.path.append("..")
from ..oracle import Oracle

class RandomForestOracle(Oracle):
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

    def __init__(self, source, feature, path=None):

        root = os.path.split(__file__)[0]
        if feature == "T5":
            if source=="D2_target":
                path = os.path.join(root, '../../data/oracles/D2_Target_NegR1.43_RFC500_T5.pkl')
            elif source=="D1_target":
                path = os.path.join(root, '../../data/oracles/D1_Target_NegR1.43_RFC500_T5.pkl')
            else:
                raise NotImplementedError("oracle not defined")
        elif feature == "AlBert":
            if source=="D2_target":
                path = os.path.join(root, '../../data/oracles/D2_Target_NegR1.43_RFC500_AlBert.pkl')
            elif source=="D1_target":
                path = os.path.join(root, '../../data/oracles/D1_Target_NegR1.43_RFC500_AlBert.pkl')
            else:
                raise NotImplementedError("oracle not defined")
        else:
            raise NotImplementedError("feature model not defined")
        
        self.device =torch.device('cpu')
        self.tokenizer, self.feature_model = PortTrans_loader(feature)
        
        # default classifier is sklearn.ensemble.RandomForestClassifier(n_estimators=500)
        with open( path, 'rb') as f:
            self.classifier = pickle.load(f)

    def __call__(self, s, eval_uncertainty = False):
        if eval_uncertainty:
            warnings.warn('Uncertainty evaluation is not support for RandomForest Classifier', UserWarning)
        # features = np.float32(CTDD_fast(s)).reshape((1, -1))
        features = np.float32(
                PortTrans_fast(s, self.tokenizer, self.feature_model, self.device)
                ).reshape((1, -1))


        return self.classifier.predict_proba(features)[:, 1]

    def evaluate_many(self, sequences, eval_uncertainty = False):
        if eval_uncertainty:
            warnings.warn('Uncertainty evaluation is not support for RandomForest Classifier', UserWarning)
        # features = np.float32([CTDD_fast(s) for s in sequences])
        features = np.float32(PortTrans_fast(sequences, self.tokenizer, self.feature_model, self.device))
        return self.classifier.predict_proba(features)[:, 1]
    
    def to(self, device):
        """If applicable, sends this oracle to `device`"""
        self.device = device
        self.feature_model.to(device)

        if not device==torch.device('cpu'):
            self.feature_model = self.feature_model.half()
