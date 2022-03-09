''' AMP sequence Oracle training

This script allows the user to extract the AMP features from the sequences, and return the numpy.array features for the classifier.

avaliable extraction method [ 'T5','CTDD' ]

'''

from .CTDD import CTDD_array
from .PortTrans import PortTrans_array
import pickle
import numpy as np
import torch

def get_feature(sequences:list, method:str, label:int, device:torch.device, outfile=None) -> list: 
    ''' Extract the features from the protein sequences.
    Parameters
    ----------
    sequences : str, list of str
        a list of str represent the amino acids
    method: str, choice =  [ 'T5','CTDD' ]
        feature extraction method 
    
    label: int, choice =  [0,1]
        the label of the sequences,[0: nonAMP, 1:AMP]
    
    device: torch.device
        the device used to extract features
    
    outfile: str [optional]
        the path to save the features for offline features
    
    Returns
    ----------
    list
        a list of numpy.array, [features, labels]
        features.shape = [num, feature_dim]
        labels.shape = [num]
    '''

    model_name = method
    assert method in ['AlBert','T5','CTDD'], "only support T5, AlBert and CTDD features "

    if method in ['AlBert','T5']:
        method = 'PortTrans'

    if isinstance(sequences,str):
        sequences = [sequences]

    kw ={}
    myFun = method + '_array' + '(sequences, model_name, device, **kw)'
    print('Descriptor type: ' + model_name)
    newArray = eval(myFun)
    labelArray = np.array( [label for i in range(len(newArray))])

    out = [newArray, labelArray]


    if outfile is not None:
        f = open(outfile, 'wb')
        pickle.dump(out, f)
        f.close()
        print("dump features in ", outfile)

    return out



# if __name__ == '__main__':
#     seq = ["AAAAAAAAAA","WWWWWWWWWW"]
#     Array, Y  = get_feature(seq, 'PortTrans', 1)
#     print(Array.shape,Array, Y)
