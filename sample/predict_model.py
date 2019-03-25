import time
import numpy as np
from math import sqrt
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression

fold = 10
gene_lncRNA_path = '../data/Gene-LncRNA.csv'
item_index_path = '../data/item_index.txt'
data_partition_dir = '../data/data-partition/'


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


def calc_metrics(y_label, y_proba):
    con_matrix = confusion_matrix(y_label, [1 if x >= 0.5 else 0 for x in y_proba])
    TN = float(con_matrix[0][0])
    FP = float(con_matrix[0][1])
    FN = float(con_matrix[1][0])
    TP = float(con_matrix[1][1])
    P = TP + FN
    N = TN + FP
    Sn = TP / P
    Sp = TN / N
    Acc = (TP + TN) / (P + N)
    Pre = (TP) / (TP + FP)
    MCC = 0
    tmp = sqrt((TP + FP) * (TP + FN)) * sqrt((TN + FP) * (TN + FN))
    if tmp != 0:
        MCC = (TP * TN - FP * FN) / tmp
    fpr, tpr, thresholds = roc_curve(y_label, y_proba)
    AUC = auc(fpr, tpr)
    return Acc, Sn, Sp, Pre, MCC, AUC


# index2name, name2index, item_list = read_index_file(item_index_path)
# genes = [x[1] for x in item_list if x[2]=='gene']
# lncRNAs = [x[1] for x in item_list if x[2]=='lncRNA']


# gl = pd.read_csv(gene_lncRNA_path)
# gl = gl.set_index("Name")


# k-fold cross validation

performances = {}
performances['RandomForest'] = []
performances['SVM'] = []
performances['LR'] = []
for cur_fold in range(fold):
    print("# Fold %d" % cur_fold)
    train, test = load_train_test_split(data_partition_dir + 'partition_fold{}.txt'.format(cur_fold))
    embeddings, node_num, dim = load_embedding(data_partition_dir + "embeddings_fold{}.txt".format(cur_fold))
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    for pair in train:
        X_train.append(embeddings[pair[0]] + embeddings[pair[1]])
        y_train.append(pair[2])
    for pair in test:
        X_test.append(embeddings[pair[0]] + embeddings[pair[1]])
        y_test.append(pair[2])

    print("There are {} train samples and {} test samples.".format(len(X_train), len(X_test)))

    start_rf = time.time()
    rf = RandomForestClassifier(100, n_jobs=-1)
    rf.fit(X=X_train, y=y_train)
    performances['RandomForest'].append(calc_metrics(y_test, rf.predict_proba(X_test)[:, 1]))
    end_rf = time.time()
    print("Performance of RF: {}. Time: {}s.".format(performances['RandomForest'][cur_fold], end_rf - start_rf))

    start_lr = time.time()
    lr = LogisticRegression(tol=1e-6, max_iter=2000, solver='lbfgs', n_jobs=-1)
    lr.fit(X=X_train, y=y_train)
    end_lr = time.time()
    performances['LR'].append(calc_metrics(y_test, lr.predict_proba(X_test)[:, 1]))
    print("Performance of LR: {}. Time: {}s.".format(performances['LR'][cur_fold], end_lr - start_lr))

    start_svm = time.time()
    # svm = SVC(C=2, probability=True)
    # svm = BaggingClassifier(SVC(C=10, probability=True), max_samples=1.0 / 10, n_estimators=10, n_jobs=-1)
    svm = LinearSVC(C=0.01, tol=1e-6, max_iter=2000)
    svm.fit(X=X_train, y=y_train)
    # performances['SVM'].append(calc_metrics(y_test, svm.predict_proba(X_test)[:, 1]))
    performances['SVM'].append(calc_metrics(y_test, svm.predict(X_test)))
    end_svm = time.time()
    print("Performance of SVM: {}. Time: {}s.".format(performances['SVM'][cur_fold], end_svm - start_svm))

print('Performance in {} fold:'.format(fold))
print('RF:', np.mean(performances['RandomForest'], axis=0))
print('SVM:', np.mean(performances['SVM'], axis=0))
print('LR:', np.mean(performances['LR'], axis=0))

# # Fold 0
# Performance of RF: (0.9940703752828275, 0.9876425855513308, 0.9966611932129174, 0.9916825743114263, 0.9855061034217093, 0.9994716935329702)
# # Fold 1
# Performance of RF: (0.9950066318171179, 0.9896795219989136, 0.9971538040503558, 0.9929155313351499, 0.9877967746071414, 0.9995043937770903)
# # Fold 2
# Performance of RF: (0.9942654287274714, 0.9862846279196089, 0.9974822112753148, 0.9937063893829525, 0.9859783305012688, 0.999433061979654)
# # Fold 3
# Performance of RF: (0.9944212538524558, 0.9869618362080673, 0.9974274767378216, 0.9935739677331146, 0.9863594385041703, 0.9993778713238077)
# # Fold 4
# Performance of RF: (0.9944210362047441, 0.9879125356512292, 0.9970441731895561, 0.9926310043668122, 0.9863616919532862, 0.9993240290033247)
# # Fold 5
# Performance of RF: (0.9938748439450686, 0.984381366290914, 0.997701023591877, 0.994238683127572, 0.9850212575529095, 0.9996228886509289)
# # Fold 6
# Performance of RF: (0.9941869538077404, 0.9879125356512292, 0.9967157479883957, 0.9918189255522225, 0.9857909583606033, 0.9992581216824663)
# # Fold 7
# Performance of RF: (0.9953183520599251, 0.9895423061252206, 0.9976462860583503, 0.9941328967116932, 0.9885563166702503, 0.9994662700189072)
# # Fold 8
# Performance of RF: (0.9947721598002497, 0.988184164063561, 0.9974273359242433, 0.9935818653557286, 0.9872194902677475, 0.9994442575461688)
# # Fold 9
# Performance of RF: (0.9947721598002497, 0.9879125356512292, 0.9975368109912968, 0.9938516190736439, 0.9872189262512062, 0.9994137627257795)
# Performance in 10 fold:
# ('RF:', array([0.99451092, 0.9876414 , 0.99727961, 0.99321335, 0.98658093, 0.99943164]))
