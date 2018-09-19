import pandas as pd


gg = pd.read_csv('../data/Gene-Gene.csv')
dg = pd.read_csv('../data/Disease-Gene.csv')
dl = pd.read_csv('../data/Disease-LncRNA.csv')
gl = pd.read_csv('../data/Gene-LncRNA-source.csv')
threshold = 0.6

# ('Duplicate genes(8):', ['BRCA2', 'PALB2', 'NBN', 'PIK3CA', 'RAD51', 'AKT1', 'CHEK2', 'BRIP1'])
# ('Isolated diseases(2):', [102660L, 155720L])


duplicate_genes = []    # 8 duplicate genes
gene_set1 = list(gg['Name'])
gene_set2 = list(gg.columns[1:])
for (a,b) in zip(gene_set1, gene_set2):
    if a!=b:
        duplicate_genes.append(a)
print('Duplicate genes(%d):' % len(duplicate_genes), duplicate_genes)
if len(duplicate_genes)>0:
    gg.drop_duplicates(inplace=True, keep='first')
    gg.drop(labels=[x+'.1' for x in duplicate_genes],axis=1,inplace=True)
    dg.drop(labels=[x+'.1' for x in duplicate_genes],axis=1,inplace=True)
    gl.drop_duplicates(inplace=True, keep='first')

isolated_diseases = []
dg=dg.set_index('Name')
dl=dl.set_index('Name')
for d in dg.index:
    if sum(list(dg.ix[d,])+list(dl.ix[d,]))==0:
        isolated_diseases.append(d)
print('Isolated diseases(%d):' % len(isolated_diseases),isolated_diseases)
if len(isolated_diseases)>0:
    dg.drop(isolated_diseases, axis=0, inplace=True)
    dl.drop(isolated_diseases, axis=0, inplace=True)


gg=gg.set_index('Name')
gl=gl.set_index('Name')
for r in gl.index:
    for c in gl.columns:
        gl.ix[r,c] = 0 if gl.ix[r,c]<threshold else 1
print(gg.shape)
print(dg.shape)
print(gl.shape)
# (503, 503)
# (99, 503)
# (503, 511)
gg.to_csv('../data/Gene-Gene.csv',)
dg.to_csv('../data/Disease-Gene.csv')
dl.to_csv('../data/Disease-LncRNA.csv')
gl.to_csv('../data/Gene-LncRNA.csv')