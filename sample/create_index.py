import pandas

dg = pandas.read_csv('../data/Disease-Gene.csv')
dl = pandas.read_csv('../data/Disease-LncRNA.csv')
gl = pandas.read_csv('../data/Gene-LncRNA.csv')


genes = list(dg.columns[1:])
lncRNAs = list(dl.columns[1:])
diseases = list(dl['Name'])

print(len(genes),len(lncRNAs),len(diseases))
# (503, 511, 99)

with open('../data/item_index.txt', 'w') as f:
    index = 0
    for gene in genes:
        f.write('%d\t%s\t%s\n' % (index, gene, 'gene'))
        index = index + 1
    for lncRNA in lncRNAs:
        f.write('%d\t%s\t%s\n' % (index, lncRNA, 'lncRNA'))
        index = index + 1
    for disease in diseases:
        f.write('%d\t%s\t%s\n' % (index, disease, 'disease'))
        index = index + 1