
# from clamp_common_eval.oracles.random_forest import RandomForestOracle
# from clamp_common_eval.oracles.small_transformer import SmallTransformerOracle

import sys 
sys.path.append("./clamp_common_eval")


from .frame_dataset.data_split import TargetSplit, TitleSplit, ClusterSplit
from .oracles.MLP import MLPOracle
from .oracles.random_forest import RandomForestOracle
from .oracles.proto_dist import MedioidDistanceOracle




import sys,os
r_1 = os.path.abspath(__file__)
r_2 = os.path.split(r_1)[0]
sys.path.append(r_2)
sys.path.append(os.path.split(r_2)[0])
# print(sys.path)
from frame_dataset.data_split import TargetSplit, TitleSplit, DataFrameDataset
# from .oracles.MLP import MLPOracle
from .oracles.random_forest import RandomForestOracle

from .oracle import Oracle


def get_test_oracle(source="D2_target", model="RandomForest", feature="AlBert", **kwargs) -> Oracle:
    """ Get the default test oracle.
    Parameters
    ----------
    source: str , choice = ["D1_target", "D2_target"]
        train_data of the default oracle; 
    model : str , choice =["RandomForest", "MLP"]
        type of classification model of the default oracle
    feature: str , choice = ["T5", "AlBert"]
        which model to extract features
    
    Returns
    ----------
    Oracle
        an Oracle that we can use to make predictions,
        e.g score = oracle("AAAAAA")
    """
    assert source in ["D1_target", "D2_target", "D1_title", "D2_title"]
    if model == "RandomForest":
        return RandomForestOracle(source, feature)
    elif model == "MLP":
        return MLPOracle(source, feature)
    elif model == "MedoidDist":
        return MedioidDistanceOracle(source, kwargs["dist_fn"], kwargs["norm_constant"])


def get_default_data_splits(setting='Target') -> DataFrameDataset:
    """ Get the data_splits
    Parameters
    ----------
    setting:  str, choice = ['Target', 'Title', 'Cluster']
        use the str as group_id to split the initialization dataset.

    Returns
    ----------
    DataFrameDataset
        a DataFrameDataset for AMP dataset management, more details are in README.md
    """
    if setting== 'Target':
        dataset = TargetSplit()  
    elif setting =='Title':
        dataset = TitleSplit()
    elif setting == 'Cluster':
        dataset = ClusterSplit()
    else:
        raise NotImplementedError()

    return dataset


if __name__ == "__main__":
    data = get_default_data_splits('Target')
    train = data.sample("D1", 2)
    print(len(train["AMP"]), len(train["nonAMP"]))
    print(train["AMP"][0], train["nonAMP"][0])


    # # train = data.sample("D1", 2)
    # # print(len(train["AMP"]),  len(train["nonAMP"]))
    # # print(train["AMP"][0],  train["nonAMP"][0])
    # get_default_data_splits()
