

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, KFold
import warnings

################ positive
# cluster = [
#     i.split() for i in
#     open('AMP_cluster.tsv','r').read().splitlines()]
# print(len(cluster),cluster[:10])

# cluster_list = [-10 for i in range(len(cluster))]

# clust_label = []
# for label, seq in cluster:
#     # print(label,seq)
#     index = int(seq.split('_')[-1])
#     if not label in clust_label:
#         clust_label.append(label)
#     label = clust_label.index(label)

#     cluster_list[index] = label

#     # import pdb; pdb.set_trace()

# amp = [
#         i for i in
#         open('AMP.fasta','r').read().splitlines()
#         if '>' not in i]

# pos_data = pd.read_csv('all_AMP.csv')['sequence'].tolist()

# cluster_id = [-10 for i in range(len(pos_data))]
# for i, cid in enumerate(cluster_list):
#     index = pos_data.index(amp[i])
#     cluster_id[index] = cid

# csv = pd.read_csv('all_AMP.csv')
# csv["cluster_id"] = cluster_id

# import pdb; pdb.set_trace()

# csv.to_csv('new_AMP.csv', index=False)


################ positive
cluster = [
    i.split() for i in
    open('nonAMP_cluster.tsv','r').read().splitlines()]
print(len(cluster),cluster[:10])

cluster_list = [-10 for i in range(len(cluster))]

clust_label = []
for label, seq in cluster:
    # print(label,seq)
    index = int(seq.split('_')[-1])
    if not label in clust_label:
        clust_label.append(label)
    label = clust_label.index(label)

    cluster_list[index] = label

    # import pdb; pdb.set_trace()

amp = [
        i for i in
        open('nonAMP.fasta','r').read().splitlines()
        if '>' not in i]



cluster_id = [-10 for i in range(len(amp))]
for i, cid in enumerate(cluster_list):
    index = amp.index(amp[i])
    cluster_id[index] = cid



csv = pd.DataFrame(columns = ['sequence','cluster_id']) 
csv['sequence'] = amp
csv['cluster_id'] = cluster_id

csv.to_csv('new_nonAMP.csv', index=False)

import pdb; pdb.set_trace()