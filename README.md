This is code to use the a subset of data from DBAASP suited for active learning for generating peptides. This is a minimal working version that has been extracted from an internal repository. 
Original commits are lost but the credit goes to Jie Fu, Tianyu Zhang and Moksh Jain.

We use `git-lfs` to track the checkpoints and data.

## Installing
```
pip install -r requirements.txt
pip install -e .
```

## Dataset Split
To get training data for our methods:
```python
from clamp_common_eval.defaults import get_default_data_splits
data = get_default_data_splits(setting='Cluster')
data = get_default_data_splits(setting='Target') # or get_default_data_splits(setting='Title')
train_data = data.sample(dataset = "D1", neg_ratio = 2)     # Get D1 and Neg(1 : 2)
train_data = data.sample(dataset = "D1-177", neg_ratio = 1) # Get C. Albican and 177 Neg
train_data = data.sample(dataset = "D2", neg_ratio = 1)     # Get D2 and Neg(1 : 1)
```
