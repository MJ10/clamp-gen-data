import os
import pandas as pd
import numpy as np

from .PeptideDataset import PeptideDataset
from sklearn.model_selection import GroupKFold, KFold
import warnings

class DataFrameDataset(PeptideDataset):
    '''
    A class used to manage the AMP dataset.
    ...
    Attributes
    ----------
    dataset : str
        the path of the dataset
    setting: str
        the basis of the data split
    pos_data: pandas.DataFrame
        the raw information of positive data in D1, D2
    d1_pos, d2_pos, d3_pos: pandas.DataFrame
        the positive data in D1, D2 and D3
    d1_neg, d2_neg: list
        the negative data in D1, D2
    d3_neg: pandas.DataFrame
        the negative data in D3

    Methods
    ----------
    get_dataframe() -> dict
        get the original dataframe d1_pos, d2_pos, d3_pos, d1_neg, d2_neg, d3_neg

    sample(dataset: str, neg_ratio: float) -> dict
        get a dict of AMP and nonAMP for training or evaluation.

    append_D3(sequences : list, label : int)
        add more sequences to D3 for further use, it will remove duplicates from D1, D2 and nonAMP for further use.
    '''
    def __init__(self, dataset, setting):
        
        print("init dataset")
        self.dataset = dataset
        self.setting = setting
        self._seed(1234)
        self.name = ['pos_D1_'+setting+'.csv', 'pos_D2_'+setting+'.csv', 'pos_D3.csv', 'neg_D3.csv']

        if not os.path.isfile(os.path.join(self.dataset,'new_AMP.csv')):
            warnings.warn("raw AMP dataset is missing")
        try:
            self.d1_pos = pd.read_csv(os.path.join(self.dataset, self.name[0]))
            self.d2_pos = pd.read_csv(os.path.join(self.dataset, self.name[1]))
            self.d3_pos = pd.read_csv(os.path.join(self.dataset, self.name[2]))
            self.d3_neg = pd.read_csv(os.path.join(self.dataset, self.name[3]))

        except IOError:
            print(".... reset all the data split and create the empty D3......")
            self.pos_data = pd.read_csv(os.path.join(self.dataset,'new_AMP.csv'))
            self.pos_data.drop_duplicates(subset='sequence', keep='first',inplace=True)
            self.pos_data.to_csv(os.path.join(self.dataset,'new_AMP.csv'), index=False)
            assert self.pos_data.shape[0] == 6438 , "something missing in 'new_AMP.csv' "
            self._split_pos(setting)
        
        self.non_amp = [
            i for i in
            open(os.path.join(self.dataset,'all_nonAMP.fasta'),'r').read().splitlines()
            if i != '>' and len(i) <= 60]
        # self.non_amp =  [i for i in self.non_amp if i[0]!='M']
        print(".... number of raw non amps ......",len(self.non_amp))

        # drop duplicates 
        self.raw_pos = self.d1_pos['sequence'].tolist()+self.d2_pos['sequence'].tolist()
        assert len(self.raw_pos) == len(list(set(self.raw_pos))), "some duplicates in D1 and D2, please delete them(pos_D1.csv, pos_d2.csv) and init the DataFrameDataset again"
        self._remove_duplicates_forD3()
        self._remove_duplicates_forNonAMP()
        self._split_neg()

        self._log_num()

        # fasta_file = 'nonAMP.fasta'
        # with open(fasta_file,'w') as f:
        #     for index, seq in enumerate(self.non_amp):
        #         f.write('>nonAMP_'+ str(index) + '\n')
        #         f.write(seq+ '\n')
        
        # fasta_file = 'AMP.fasta'
        # with open(fasta_file,'w') as f:
        #     for index, seq in enumerate(self.raw_pos):
        #         f.write('>AMP_'+ str(index) + '\n')
        #         f.write(seq+ '\n')
        

        # import pdb; pdb.set_trace()
        # self.d1_pos = self.d1_pos.loc[self.d1_pos['group']==11] ####### Candidaalbicans:11
        
        

        # import pdb; pdb.set_trace()
    
    def get_dataframe(self) -> dict:
        '''Get the raw 'pd.DataFrame' for easy manipulate.
        Returns
        -------
        dict
            a dict containing 6 keys including "D1_pos, D2_pos, D3_pos, D1_neg, D2_neg, D3_neg"
            each value is the raw dataset ( dataframe or list)
        '''
        return {'D1_pos':self.d1_pos, 'D2_pos':self.d2_pos, 'D1_neg':self.d1_neg, 'D2_neg': self.d2_neg, 'D3_pos':self.d3_pos, 'D3_neg':self.d3_neg }

    def sample(self, dataset: str, neg_ratio: float) -> dict:
        ''' Get the pos-neg pair dataset for training or evaluation
        Parameters
        ----------
        dataset : str
            choose the dataset you want to sample from. make a choice in ['D1', 'D1-177','D2', 'D3']
        neg_ratio : float
            the ratio of sampled negative data. ratio = num_pos/num_neg, if neg_ratio<0, return all nonAMPs
        
        Returns
        ----------
        dict
            a dict containing 2 keys including "AMP , nonAMP"
            each value is a list of sequence
        '''
        if dataset=="D1":
            pos = self.d1_pos['sequence'].tolist()
            neg = self.d1_neg
        elif dataset=="D1-177":
            pos = self.d1_pos[self.d1_pos['group']==11]['sequence'].tolist()
            neg = self.d1_neg
        elif dataset=="D2":
            pos = self.d2_pos['sequence'].tolist()
            neg = self.d2_neg
        elif dataset=="D3":
            pos = self.d3_pos['sequence'].tolist() +  self.d2_pos['sequence'].tolist()
            neg = self.d3_neg['sequence'].tolist() +  self.d2_neg
        elif dataset=="D":
            pos = self.d1_pos['sequence'].tolist() +  self.d2_pos['sequence'].tolist()
            neg = self.d1_neg + self.d2_neg

        r = neg_ratio
        n = int(np.ceil(len(pos) * r)) 
        neg_max = len(neg)

        if neg_ratio < 0 or n >=neg_max:
            neg_sampled = neg
        else:
            neg_sampled = [neg[i] for i in self.sampling_rng.randint(0, neg_max, n )]
        
        return {"AMP":pos, "nonAMP": neg_sampled}


    def append_D3(self, sequences : list, label : int):
        ''' Add some new sequences to D3 (positive or negetiave)
        Parameters
        ----------
        sequences : list
            a list of some str represent the new sequences
        label : int ,choice=[0,1]
            the label of these new sequences, [0: nonAMP, 1:AMP]

        At the same time, it will remove duplicates from D1, D2 and nonAMP and re-split nonAMPs for further use.
        '''
        if label:
            new = pd.DataFrame(columns = ['sequence','group']) 
            new['sequence'] = sequences
            new['group'] = [-1 for i in range(len(sequences))]

            self.d3_pos = self.d3_pos.append(new, ignore_index = True)
            self.d3_pos.drop_duplicates(subset='sequence', keep='first',inplace=True)
            self.d3_pos.to_csv(os.path.join(self.dataset,self.name[2]), index=False)
            
        else:
            new = pd.DataFrame(columns = ['sequence','group']) 
            new['sequence'] = sequences
            new['group'] = [-10 for i in range(len(sequences))]

            self.d3_neg = self.d3_neg.append(new, ignore_index = True)
            self.d3_neg.drop_duplicates(subset='sequence', keep='first',inplace=True)
            self.d3_neg.to_csv(os.path.join(self.dataset,self.name[3]), index=False)

        self._remove_duplicates_forD3()
        self._remove_duplicates_forNonAMP()
        self._log_num()
        

    def _split_pos(self, setting):
        '''
        the initialization of the positive D1 and D2 from raw new_AMP.csv dataset and create empty file for D3
        '''
        if setting == 'target':
            groups = np.array(self.pos_data.target_id.tolist())
            d1, d2 = next(GroupKFold(2).split(np.arange(len(self.pos_data)),groups=groups))
            ##  'Candidaalbicans' in D1 and Escherichiacoli in D2
            assert 11 in groups[d1].tolist() and 2 in groups[d2].tolist()  ## Escherichiacoli:2,   Candidaalbicans:11
        elif setting == 'title':
            groups = np.array(self.pos_data.citation_id.tolist())
            d1, d2 = next(GroupKFold(2).split(np.arange(len(self.pos_data)),groups=groups))
        elif setting == 'cluster':
            groups = np.array(self.pos_data.cluster_id.tolist())
            d1, d2 = next(GroupKFold(2).split(np.arange(len(self.pos_data)),groups=groups))
        
        d1_pos = self.pos_data.loc[d1,['sequence']]
        d2_pos = self.pos_data.loc[d2,['sequence']]
        d1_pos["group"] = groups[d1]
        d2_pos["group"] = groups[d2]
        d3_pos = pd.DataFrame(columns = ['sequence','group']) 

        d1_pos.reset_index(inplace=True,drop=True)
        d2_pos.reset_index(inplace=True,drop=True)
        d3_pos.reset_index(inplace=True,drop=True)

        d1_pos.to_csv(os.path.join(self.dataset, self.name[0]), index=False)
        d2_pos.to_csv(os.path.join(self.dataset, self.name[1]), index=False)
        d3_pos.to_csv(os.path.join(self.dataset, self.name[2]), index=False)

        self.d1_pos = d1_pos
        self.d2_pos = d2_pos
        self.d3_pos = d3_pos

        self.d3_neg = pd.DataFrame(columns = ['sequence','group'])
        self.d3_pos.to_csv(os.path.join(self.dataset, self.name[3]), index=False)

    def _split_neg(self):
        '''
        the initialization of the negative D1 and D2 from raw all_nonAMP.csv dataset
        '''
        num_neg = len(self.non_amp)
        neg1 = int( ( num_neg  * self.d1_pos.shape[0])/(self.d1_pos.shape[0] + self.d2_pos.shape[0]) )

        self.d1_neg = self.non_amp[:neg1]
        self.d2_neg = self.non_amp[neg1:]

        

        # import pdb; pdb.set_trace()
        # self.d2_neg = [i for i in self.non_amp if i[0]!='M']
        # self.d1_neg = [i for i in self.non_amp if i[0]=='M']
        # print("neg:neg:", len(self.d1_neg), len(self.d2_neg))
    
    def _remove_duplicates_forD3(self):
        '''
        Remove the duplicates from d3_pos and d3_neg. with the priority:
            (D1_pos, D2_pos) > D3_pos > (D1_neg, D2_neg) > D3_neg
        '''
        re_index=[]
        for i, seq in enumerate(self.d3_pos['sequence'].tolist()):
            if seq in self.raw_pos:
                re_index.append(i)
        if len(re_index)>0: 
            self.d3_pos = self.d3_pos.drop(re_index)
            print("warning!!!!  remove ",len(re_index), " repetitive seq in positive D3 and rewrite the 'pos_D3.csv'")
            self.d3_pos.to_csv(os.path.join(self.dataset,self.name[2]), index=False)
        
        re_index=[]
        for i, seq in enumerate(self.d3_neg['sequence'].tolist()):
            if seq in self.raw_pos or seq in self.non_amp or seq in self.d3_pos['sequence'].tolist():
                re_index.append(i)
        if len(re_index)>0: 
            self.d3_neg = self.d3_neg.drop(re_index)
            print("warning!!!!  remove ",len(re_index), " repetitive seq in negative D3 and rewrite the 'neg_D3.csv'")
            self.d3_neg.to_csv(os.path.join(self.dataset,self.name[3]), index=False)
    
    def _remove_duplicates_forNonAMP(self):
        '''
        Remove the duplicates from nonAMPs. with the priority:
            (D1_pos, D2_pos) > D3_pos > (D1_neg, D2_neg) > D3_neg
        '''
        pos = self.raw_pos +self.d3_pos['sequence'].tolist()
        self.non_amp = list(set(self.non_amp))
        self.non_amp = list(filter(lambda x: not x in pos, self.non_amp))
        self.non_amp.sort()
        self.non_amp = [self.non_amp[i] for i in self.sampling_rng.permutation(len(self.non_amp))]
        
    def _seed(self, seed):
        '''
        Ranfom sampling seed.
        '''
        self.sampling_rng = np.random.RandomState(seed)

    def _log_num(self):
        '''
        Print the number of the dataset, D1, D2 and D3 with(positive:negative).
        '''

        d1 = (self.d1_pos.shape[0], len(self.d1_neg))
        d2 = (self.d2_pos.shape[0], len(self.d2_neg))
        d3 = (self.d3_pos.shape[0], self.d3_neg.shape[0])

        print("number of the data (P:N) under", self.setting, "setting: D1:",d1, ";  D2:",d2, "; D3:",d3, )






class TargetSplit(DataFrameDataset):
    def __init__(self, dataset =os.path.join(os.path.split(__file__)[0], '../../data/dataset')):
        super().__init__(dataset, setting = "target")


class TitleSplit(DataFrameDataset):
    def __init__(self, dataset =os.path.join(os.path.split(__file__)[0], '../../data/dataset')):
        super().__init__(dataset, setting = "title")



class ClusterSplit(DataFrameDataset):
    def __init__(self, dataset =os.path.join(os.path.split(__file__)[0], '../../data/dataset')):
        super().__init__(dataset, setting = "cluster")