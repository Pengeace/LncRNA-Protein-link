import time
import numpy as np
import pandas as pd
from RWR import RWR
from copy import deepcopy
from operator import itemgetter
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

data_fold_num = 0  # the data partition to be tested
gene_lncRNA_path = '../data/Gene-LncRNA.csv'
item_index_path = '../data/item_index.txt'
data_partition_dir = '../data/data-partition/'
full_network_matrix_path = '../data/full-matrix.csv'


def load_train_test_split(path):
    train_test = [[], []]
    f = open(path, 'r')
    flag = 0
    for line in f.readlines():
        line = line.strip()
        if 'Train' in line:
            flag = 0
        elif 'Test' in line:
            flag = 1
        else:
            gene, lncRNA, label = [int(x) for x in line.split()]
            train_test[flag].append([gene, lncRNA, label])
    return train_test[0], train_test[1]


def load_embedding(path):
    head = False
    num = None
    dim = None
    embeddings = {}
    with open(path, 'r') as f:
        for line in f:
            if head:
                num, dim = [int(x) for x in line.strip().split()]
                head = False
            else:
                items = line.strip().split()
                embeddings[int(items[0])] = [float(x) for x in items[1:]]
    return embeddings, num, dim


def read_index_file(path):
    index2name = {}
    name2index = {}
    item_list = []
    with open(path) as f:
        for line in f:

            item = line.strip().split('\t')
            item_list.append(item)
            index2name[int(item[0])] = item[1]
            if item[1] not in name2index:
                name2index[item[1]] = int(item[0])
            else:
                print(item[1])

    return index2name, name2index, item_list


def update_adjacent_list(data_frame, name2index, adj_list):
    columns = list(data_frame.columns)[1:]
    data_frame = data_frame.set_index('Name')
    rows = list(data_frame.index)

    for r in rows:
        item = name2index[r]
        adjs = dict(data_frame.loc[r, :])
        if item not in adj_list:
            adj_list[item] = []
        for key in adjs:
            if adjs[key] != 0:
                adj_list[item].append(name2index[key])

    for c in columns:
        item = name2index[c]
        adjs = dict(data_frame[c])
        if item not in adj_list:
            adj_list[item] = []
        for key in adjs:
            if adjs[key] != 0:
                adj_list[item].append(name2index[key])

    return adj_list


def calc_auc(y_label, y_proba):
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return AUC


index2name, name2index, item_list = read_index_file(item_index_path)
total_item_num = len(item_list)
full_network_matrix = pd.read_csv(full_network_matrix_path)
gene_lncRNA_matrix = pd.read_csv(gene_lncRNA_path)
gl_pos_adj_list = update_adjacent_list(gene_lncRNA_matrix, name2index, {})
full_network_matrix.columns = ['Name'] + list(full_network_matrix.columns[1:])
full_network_matrix = full_network_matrix.set_index('Name')
gene_list = [name2index[x] for x in gene_lncRNA_matrix['Name']]

performances = {}
performances['DeepWalk-RandomForest'] = []
performances['RWR'] = []
print("# Process data fold %d" % data_fold_num)
train, test = load_train_test_split(data_partition_dir + 'partition_fold{}.txt'.format(data_fold_num))
embeddings, node_num, dim = load_embedding(data_partition_dir + "embeddings_full.txt")
X_train = []
y_train = []
X_test = []
y_test = []

# RWR with full network
#####################################
print("Full network data:")
# RWR prediction
print("RWR prediction ...")
W = deepcopy(full_network_matrix.values)
sum_column = W.sum(axis=0)
W = W / sum_column
lncRNA_column_tmp = []
gene_column_tmp = []
y_probas_rwr = []
test_v1 = deepcopy(test)
i = 0
start_rwr = time.time()
for pair in test_v1:
    i += 1
    if i % 500 == 0:
        print("processing test pair %d" % (i))
    gene, lncRNA, label = pair
    lncRNA_interacted_gene_num = len(gl_pos_adj_list[lncRNA])
    seeds = set(gl_pos_adj_list[lncRNA])
    if label:
        lncRNA_interacted_gene_num -= 1
        seeds.remove(gene)

        # preserve the lncRNA and gene columns
        lncRNA_column_tmp = W[:, lncRNA]
        gene_column_tmp = W[:, gene]
        # create the current lncRNA and gene columns
        lncRNA_column_cur = np.array(full_network_matrix[str(lncRNA)])
        gene_column_cur = np.array(full_network_matrix[str(gene)])
        lncRNA_column_cur[gene] = 0
        gene_column_cur[lncRNA] = 0
        lncRNA_column_cur = lncRNA_column_cur / sum(lncRNA_column_cur)
        gene_column_cur = gene_column_cur / sum(gene_column_cur)
        W[:, lncRNA] = lncRNA_column_cur
        W[:, gene] = gene_column_cur

    if len(seeds) == 0:
        test_v1.remove(i)
        continue

    rwr = RWR(W=W, seeds=seeds)
    p = rwr.compute()
    gene_score = []
    for gene_item in gene_list:
        if gene_item not in gl_pos_adj_list[lncRNA]:
            gene_score.append((gene_item, p[gene_item]))
    if label:
        gene_score.append((gene, p[gene]))
    cur_gene_score_pair = (gene, p[gene])
    gene_score = sorted(gene_score, key=itemgetter(1))
    pos = gene_score.index(cur_gene_score_pair)
    y_probas_rwr.append(pos * 1.0 / len(gene_score))

    if label:
        W[:, lncRNA] = lncRNA_column_tmp
        W[:, gene] = gene_column_tmp

end_rwr = time.time()
performances['RWR'].append(calc_auc([x[2] for x in test_v1], y_probas_rwr))
print("Performance of RWR: {}. Time: {}s.".format(performances['RWR'], end_rwr - start_rwr))


# Random forest prediction
print("RF prediction ...")
for pair in train:
    X_train.append(embeddings[pair[0]] + embeddings[pair[1]])
    y_train.append(pair[2])
for pair in test_v1:
    X_test.append(embeddings[pair[0]] + embeddings[pair[1]])
    y_test.append(pair[2])
print("There are {} train samples and {} test samples.".format(len(X_train), len(X_test)))
start_rf = time.time()
rf = RandomForestClassifier(100, n_jobs=-1)
rf.fit(X=X_train, y=y_train)
y_probas_rf = rf.predict_proba(X_test)[:, 1]
performances['DeepWalk-RandomForest'].append(calc_auc(y_test, y_probas_rf))
end_rf = time.time()
print("Performance of RF: {}. Time: {}s.".format(performances['DeepWalk-RandomForest'], end_rf - start_rf))

# ROC DeepWalk-RF
fpr, tpr, thresholds = roc_curve(y_test, y_probas_rf)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC of %s (AUC = %0.5f)' % ("DeepWalk-RF", roc_auc))

# ROC RWR
fpr, tpr, thresholds = roc_curve(y_test, y_probas_rwr)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC of %s (AUC = %0.5f)' % ("RWR", roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Luck', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC and AUC comparison', fontsize=16)
plt.legend(loc="lower right")
plt.savefig('../result/ROC-AUC-1.pdf')
plt.show()
plt.close()

# Performance of RWR: 0.9959854067495623. Time: 144.68404984474182s.
# Performance of RF: [0.99983904448541361]. Time: 26.475212574005127s.
#####################################


# RWR with reduced network
#####################################
print("Reduced network data:")
# RWR prediction
print("RWR prediction ...")
W = deepcopy(full_network_matrix.values)
for gene, lncRNA, label in test:
    if label:
        W[gene, lncRNA] = 0
        W[lncRNA, gene] = 0
sum_column = W.sum(axis=0)
for i in range(len(sum_column)):
    if sum_column[i] == 0:
        sum_column[i] = 1
W = W / sum_column

y_probas_rwr = []
test_v2 = deepcopy(test)
i = 0
start_rwr = time.time()
for pair in test_v2:
    i += 1
    if i % 500 == 0:
        print("processing test pair %d" % (i))
    gene, lncRNA, label = pair
    lncRNA_interacted_gene_num = len(gl_pos_adj_list[lncRNA])
    seeds = set(gl_pos_adj_list[lncRNA])
    if label:
        lncRNA_interacted_gene_num -= 1
        seeds.remove(gene)

    if len(seeds) == 0:
        test_v2.remove(i)
        continue

    rwr = RWR(W=W, seeds=seeds)
    p = rwr.compute()
    gene_score = []
    for gene_item in gene_list:
        if gene_item not in gl_pos_adj_list[lncRNA]:
            gene_score.append((gene_item, p[gene_item]))
    if label:
        gene_score.append((gene, p[gene]))
    cur_gene_score_pair = (gene, p[gene])
    gene_score = sorted(gene_score, key=itemgetter(1))
    pos = gene_score.index(cur_gene_score_pair)
    y_probas_rwr.append(pos * 1.0 / len(gene_score))

end_rwr = time.time()
performances['RWR'].append(calc_auc([x[2] for x in test_v2], y_probas_rwr))
print("Performance of RWR: {}. Time: {}s.".format(performances['RWR'], end_rwr - start_rwr))

# Random forest prediction
X_test = []
y_test = []
print("RF prediction ...")
for pair in test_v2:
    X_test.append(embeddings[pair[0]] + embeddings[pair[1]])
    y_test.append(pair[2])
print("There are {} train samples and {} test samples.".format(len(X_train), len(X_test)))
start_rf = time.time()
rf = RandomForestClassifier(100, n_jobs=-1)
rf.fit(X=X_train, y=y_train)
y_probas_rf = rf.predict_proba(X_test)[:, 1]
performances['DeepWalk-RandomForest'].append(calc_auc(y_test, y_probas_rf))
end_rf = time.time()
print("Performance of RF: {}. Time: {}s.".format(performances['DeepWalk-RandomForest'], end_rf - start_rf))

# ROC DeepWalk-RF
fpr, tpr, thresholds = roc_curve(y_test, y_probas_rf)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC of %s (AUC = %0.5f)' % ("DeepWalk-RF", roc_auc))

# ROC RWR
fpr, tpr, thresholds = roc_curve(y_test, y_probas_rwr)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, lw=1, alpha=0.8, label='ROC of %s (AUC = %0.5f)' % ("RWR", roc_auc))

plt.plot([0, 1], [0, 1], linestyle='--', lw=1, label='Luck', alpha=.8)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC and AUC comparison', fontsize=16)
plt.legend(loc="lower right")
plt.savefig('../result/ROC-AUC-2.pdf')
plt.show()
plt.close()

# Performance of RWR: [0.99750938529338584]. Time: 146.42518210411072s.
# Performance of RF: [0.9997800658657765]. Time: 28.522676467895508s.
#####################################
