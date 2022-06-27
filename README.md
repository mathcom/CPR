# Improved prediction of breast cancer outcome by identifying heterogeneous biomarkers

CPR (clustering and pagerank) is a program to identify prognostic genes (biomarkers) and use them to predict prognosis of cancer patients.

Please refer to included 'manual.pdf'.

For more detail, please refer to Choi, Jonghwan, et al. "Improved prediction of breast cancer outcome by identifying heterogeneous biomarkers." Bioinformatics 33.22 (2017): 3619-3626.

- Latest update: 27 June 2022

--------------------------------------------------------------------------------------------
## Installation: 
- We recommend to install via Anaconda (https://www.anaconda.com/)
- After installing Anaconda, please create a conda environment with the following commands:
```
git clone https://github.com/mathcom/CPR.git
cd CPR
conda env create -f environment.yml
```

--------------------------------------------------------------------------------------------
## Usage: 
```
python CPR.py EXPRESSION_FILE CLINICAL_FILE NETWORK_FILE RESULT_FILE [optional parameters]
```
	
--------------------------------------------------------------------------------------------
## Example:
```
mkdir results
python CPR.py data/ex_EXPRESSION.txt data/ex_CLINICAL.txt data/ex_NETWORK.txt results/ex_RESULT
```
```
>>> 0. Arguments
Namespace(EXPRESSION_FILE='data/ex_EXPRESSION.txt', CLINICAL_FILE='data/ex_CLINICAL.txt', NETWORK_FILE='data/ex_NETWORK.txt', RESULT_FILE='results/ex_RESULT', numClusters=0, dampingFactor=0.7, numBiomarkers=70, conditionHubgene=0.02, crossvalidation=False)
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
	results/ex_RESULT_biomarker.txt
	results/ex_RESULT_score.txt
	results/ex_RESULT_subnetwork.txt
$
```
	
--------------------------------------------------------------------------------------------
## Note:
- The option parameter '-h' shows help message.
```
$ python CPR.py -h
```	
	
--------------------------------------------------------------------------------------------