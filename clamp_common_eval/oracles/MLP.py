import torch.nn as nn
import torch.nn.functional as F
import os
import torch
import sys
sys.path.append("..")
sys.path.append("../..")
from ..oracle import Oracle
from . import CTDD_fast, PortTrans_fast, PortTrans_loader
import numpy as np
from torch.utils import data


class CTDDDataset(data.Dataset):
    def __init__(self, X):
        self.data = X.astype(np.float32)

    def __getitem__(self, idx):
        return self.data[idx, :]

    def __len__(self):
        return self.data.shape[0]


class MLPOracle(Oracle):
    def __init__(self, source, feature):
        root = os.path.join(os.path.split(__file__)[0], "../")
        # For ProtTrans
        if feature == "T5":
            self.model = MLP(1024, 2, 1024, dropout_rate=0.5)
        elif feature == "AlBert":
            self.model = MLP(4096, 2, 1024, dropout_rate=0.5)
        # For CTDD 
        # self.model = MLP(195, 2, 1024, dropout_rate=0.5)
        if source not in ["D2_target", "D1_target"]:
            raise NotImplementedError("oracle not defined")
        self.best_model_path = os.path.join(
            root, "../data/oracles/" + source +"_MLP_best_Layer_1024_{}.pt".format(feature)
        )
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.device = torch.device("cpu")
        self.tokenizer, self.feature_model = PortTrans_loader(feature)

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.feature_model.to(device)

        if not device==torch.device('cpu'):
            self.feature_model = self.feature_model.half()

    # Input :
    #   s : given single sequence feature N * 195
    #	eval_uncertainty : if evaluate MLP classifere uncertainty
    #	T : MC-sampling times
    # Ouput : a dictionary with keys
    #   confidence : softmax probability with class
    #   prediction : prediction class
    #   entrophy : uncertainty of prediction results, small value refers to certain output, large value refers to uncertain outputs

    def __call__(self, s, eval_uncertainty=True, T=1000):
        # entropy as classification uncertainty:
        # https://github.com/ShellingFord221/My-implementation-of-What-Uncertainties-Do-We-Need-in-Bayesian-Deep-Learning-for-Computer-Vision/blob/master/classification_epistemic.py

        # features = np.float32(CTDD_fast(s)).reshape((1, -1))
        features = np.float32(
                PortTrans_fast(s, self.tokenizer, self.feature_model, self.device)
                ).reshape((1, -1))
        test_dataset = CTDDDataset(features)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
        if not eval_uncertainty:
            outputs = []
            predtions = []
            for X in test_loader:
                X = X.to(self.device)
                output = self.model(X)
                pred = output.data.max(dim=1, keepdim=False)[
                    1
                ]  # get the index of the max log-probability
                predtions.append(pred.cpu().numpy())
                probs = F.softmax(output, dim=1).data.cpu()
                outputs.append(probs.detach().numpy())
            out = np.concatenate(outputs)
            preds = np.concatenate(predtions)
            return {"confidence": out, "prediction": preds}
        else:
            outputs = []
            predtions = []
            entropies = []
            for X in test_loader:
                X = X.to(self.device)
                prob_MC = []
                for _ in range(T):
                    output = self.model(X)
                    probs = F.softmax(output, dim=1).data.cpu().numpy()
                    prob_MC.append(probs)
                prob = np.mean(prob_MC, 0)
                pred = np.argmax(prob, -1)
                ent = entropy(prob)
                outputs.append(prob)
                predtions.append(pred)
                entropies.append(ent)
            out = np.concatenate(outputs)
            preds = np.concatenate(predtions)
            ents = np.concatenate(entropies)
            return {"confidence": out, "prediction": preds, "entropy": ents}

    def evaluate_many(self, sequences, eval_uncertainty=True, T=1000):
        # For CTDD
        # features = np.float32([CTDD_fast(s) for s in sequences])
        # For ProtTrans 
        features = np.float32(PortTrans_fast(sequences, self.tokenizer, self.feature_model, self.device))
        test_dataset = CTDDDataset(features)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
        if not eval_uncertainty:
            outputs = []
            predtions = []
            for X in test_loader:
                X = X.to(self.device)
                output = self.model(X)
                pred = output.data.max(dim=1, keepdim=False)[
                    1
                ]  # get the index of the max log-probability
                predtions.append(pred.cpu().numpy())
                probs = F.softmax(output, dim=1).data.cpu()
                outputs.append(probs.detach().numpy())
            out = np.concatenate(outputs)
            preds = np.concatenate(predtions)
            return {"confidence": out, "prediction": preds}
        else:
            outputs = []
            predtions = []
            entropies = []
            for X in test_loader:
                X = X.to(self.device)
                prob_MC = []
                for _ in range(T):
                    output = self.model(X)
                    probs = F.softmax(output, dim=1).data.cpu().numpy()
                    prob_MC.append(probs)
                prob = np.mean(prob_MC, 0)
                pred = np.argmax(prob, -1)
                ent = entropy(prob)
                outputs.append(prob)
                predtions.append(pred)
                entropies.append(ent)
            out = np.concatenate(outputs)
            preds = np.concatenate(predtions)
            ents = np.concatenate(entropies)
            return {"confidence": out, "prediction": preds, "entropy": ents}


def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)


def entropy(p):
    """
    p = np.array([p1, p2, p3, ...]), where sum(p_i) = 1
    """
    p = np.array(p)
    return np.sum(-np.log(p) * p, axis=-1)


class MLP(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.input_dim = num_inputs
        self.output_dim = num_outputs
        self.dropout_rate = dropout_rate

        self.fc1 = nn.Linear(self.input_dim, num_hiddens)
        self.fc2 = nn.Linear(num_hiddens, num_hiddens)
        self.fc4 = nn.Linear(num_hiddens, self.output_dim)
        self.act = nn.ReLU(inplace=True)  # for saving gpu memory

    def forward(self, x, sample=True):
        mask = self.training or sample
        assert x.size(1) == self.input_dim, "Wrong input dim"
        # -----------
        x = self.fc1(x)
        x = MC_dropout(x, p=self.dropout_rate, mask=mask)
        x = self.act(x)
        # -----------
        x = self.fc2(x)
        x = MC_dropout(x, p=self.dropout_rate, mask=mask)
        x = self.act(x)

        y = self.fc4(x)

        return y
