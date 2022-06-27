CPR is a program to identify prognostic genes (biomarkers) and use them to predict prognosis of cancer patients.

Please refer to included 'manual.pdf'.

For more detail, please refer to Choi, Jonghwan, et al. "Improved prediction of breast cancer outcome by identifying heterogeneous biomarkers." Bioinformatics 33.22 (2017): 3619-3626.

Latest update: 27 June 2022


--------------------------------------------------------------------------------------------
USAGE: 
```
python CPR.py EXPRESSION_FILE CLINICAL_FILE NETWORK_FILE RESULT_FILE [optional parameters]
```
	
--------------------------------------------------------------------------------------------
example:
```
$ python CPR.py ex_EXPRESSION.txt ex_CLINICAL ex_NETWORK.txt ex_RESULT
>>> 0. Arguments
Namespace(CLINICAL_FILE='ex_CLINICAL.txt', EXPRESSION_FILE='ex_EXPRESSION.txt', NETWORK_FILE='ex_NETWORK.txt', RESULT_FILE='ex_RESULT', conditionHubgene=0.02, crossvalidation=False, dampingFactor=0.7, numBiomarkers=70, numClusters=0)
>>> 1. Load data
>>> 2. Preprocess data
	n_samples: 189
	n_genes  : 8819     (common genes in both EXPRESSION and NETWORK)
	n_edges  : 150168   (edges with the common genes)
>>> 3. Conduct CPR
	K-means clustering
	-> n_clusters: 2
		In cluster[0], n_samples:85, n_goods:51, n_poors:34
		In cluster[1], n_samples:104, n_goods:48, n_poors:56
	Modified PageRank
>>> 4. Save results
	ex_RESULT_biomarker.txt
	ex_RESULT_score.txt
	ex_RESULT_subnetwork.txt
$
```
	
--------------------------------------------------------------------------------------------
Note:
The option parameter '-h' shows help message.
```
$ python CPR.py -h
```	
	
--------------------------------------------------------------------------------------------