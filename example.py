from __future__ import print_function
from CPR import load_data
example = load_data()

print(example.keys())
""" result
>>> ['geneList', 'edgeList', 'data_BRCA', 'label_BRCA',
'data_GSE4922', 'label_GSE4922', 'data_GSE7390', 'label_GSE7390']
"""

geneList = example['geneList']
edgeList = example['edgeList']
data_train = example['data_BRCA']
label_train = example['label_BRCA']
data_test = example['data_GSE4922']
label_test = example['label_GSE4922']


from CPR import CPR

# construct with default values
cpr = CPR()

# set paramters
cpr.setParam(dampingFactor=0.5)

# fit with BRCA dataset
cpr.fit(geneList, edgeList, data_train, label_train, randomState=1)

# validate with GSE4922 dataset
AUC = cpr.validate(geneList, data_test, label_test, randomState=1)


# print accuracy
print("the area under curve=%g" % AUC)
""" result
>>> the area under curve=0.654135
"""

# print the top 10 genes
rankedGenes = cpr.getRankedGenes()
print(rankedGenes[:10])
""" result
>>> the area under curve=0.654135
"""

# print the top 5 biomarkers
biomarkers = cpr.getBiomarkers()
print(biomarkers[:5])
""" result
>>> the area under curve=0.654135
"""

# get the parameter used in model
parameters = cpr.getParam()
print(parameters)
""" result
>>> the area under curve=0.654135
"""
