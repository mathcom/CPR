from __future__ import print_function
import numpy as np
import pickle
import sys
from copy import deepcopy
from math import sqrt
from operator import itemgetter
from scipy import interp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve, silhouette_score
from sklearn.model_selection import StratifiedKFold

from CPR_preprocessing import *
from CPR_util import *

def main(argc, argv):
	networkFile = ""
	exprFile_tr = ""
	exprFile_te = ""
	resultFile = ""
	dampingFactor = 0.7   #default
	n_biomarkers  = 70    #default
	isValidateGiven = False
	isResultFileGiven = False
	
	if "-g" not in argv or "-r" not in argv:
		print("")
		print("usage: python CPR.py -g GENE_NETWORK_FILE -r GENE_EXPRESSION_FILE_FOR_TRAIN [-e GENE_EXPRESSION_FILE_FOR_TEST] [-o RESULT_FILE] [optional parameters]")
		print("")
		print("< Required inputs >")
		print("-g  GENE_NETWORK_FILE : Gene interactions")
		print("        Tab-delimited file with one header 'gene1\tgene2'")
		print("-r  GENE_EXPRESSION_FILE_TRAIN : Gene expressions dataset for training")
		print("        Tab-delimited file with three headers 'PATIENT', 'OSEVENT' and 'OSDURATION'")
		print("        For detail annotation, please see manual.pdf")
		print("")
		print("< Optional inputs >")
		print("-e  GENE_EXPRESSION_FILE_TEST : Gene expressions dataset to test")
		print("        Tab-delimited file with three headers 'PATIENT', 'OSEVENT' and 'OSDURATION'")
		print("        If test dataset is given, the fitted model is validated with test dataset")
		print("        If not given, the fitted model is validated with training dataset")
		print("-o  RESULT_FILE : A summary of results are written in the file")
		print("        If RESULT_FILENAME is not given, all results are showed in command lines")
		print("        The summary have 1) accuracy, 2) biomarkers, and 3) subnetwork with biomarkers")
		print("")
		print("< Optional parameters >")
		print("-d  dampingFactor : float, default = 0.7")
		print("        This parameter decides an influence of network information on prediction")
		print("        The value must be between 0.0 and 1.0")
		print("-n  n_biomarkers : int, default = 70")
		print("        This parameter decides number of biomarkers to use in prediction")
		print("")
		exit(1)
	else:
		for i in range(1,argc,2):
			flag = argv[i]
			value = argv[i+1]
			
			if flag == "-g":
				networkFile = value
			elif flag == "-r":
				exprFile_tr = value
			elif flag == "-e":
				exprFile_te = value
				isValidateGiven = True
			elif flag == "-o":
				resultFile = value
				isResultFileGiven = True
			elif flag == "-d":
				dampingFactor = float(value)
			elif flag == "-n":
				n_biomarkers = int(value)
	
	""" 1. Load data """
	print(">>> Now loading data...")
	edgeList, geneset_g = preprocessing_network(read_file(networkFile))
	samples_tr, labels_tr, data_tr, geneList_tr = preprocessing_expression(read_file(exprFile_tr)) 
	if isValidateGiven:
		samples_te, labels_te, data_te, geneList_te = preprocessing_expression(read_file(exprFile_te))
		
	commonGenes = find_commonGenes([geneset_g, set(geneList_tr)])
	
	edgeList = parsing_commonEdge(commonGenes, edgeList)
	
	data_tr, geneList_tr = parsing_commonExpr(commonGenes, data_tr, geneList_tr)
	
	""" 2. construct and fit model """
	print(">>> Constructing model")
	cpr = CPR()
	cpr.setParam(dampingFactor=dampingFactor, n_biomarkers=n_biomarkers)
	print(">>> Fitting the model...")
	cpr.fit(geneList_tr, edgeList, data_tr, labels_tr, randomState=1)
	
	""" 3. validate model """
	print(">>> Now validating classification...")
	if isValidateGiven:
		AUC = cpr.validate(geneList_te, data_te, labels_te, randomState=1)
	else:
		AUC = cpr.validate(geneList_tr, data_tr, labels_tr, randomState=1)
	
	""" 4. find subnetwork """
	subEdgeList = find_subnetwork(cpr.getBiomarkers(), edgeList)
	
	""" 5. summary for result """
	print(">>> Summary")
	if isResultFileGiven:
		fwrite_summary(resultFile, AUC, cpr.getBiomarkers(), subEdgeList)
	else:
		print_summary(AUC, cpr.getBiomarkers(), subEdgeList)
	
	print(">>> Finish")
	

class CPR:
	""" hyper-parameters """
	dampingFactor = .0      # parameter of pageRank	
	n_biomarkers  = 0
	n_clusters    = 0       # the number of clusters in K-means clustering
	n_pc          = 0       # the number of principal components in PCA
	t_degree      = .0      # threshold of degree for selection of biomarkers
	
	""" constant parameters """
	iterations    = 5
	n_trees       = 100
	
	""" private variables """
	biomarkers = list()
	rankedGenes = list()
	networks = list()
	isFitted = False
	
	""" constructor """
	def __init__(self,
				 dampingFactor=0.7,
				 n_biomarkers=70,
				 n_clusters=2,
				 n_pc=2,
				 t_degree=0.02):
		
		self.dampingFactor = float(dampingFactor)
		self.n_biomarkers  = int(n_biomarkers)
		self.n_clusters    = int(n_clusters)
		self.n_pc          = int(n_pc)
		self.t_degree      = float(t_degree)
	
	""" set parameters """
	def setParam(self, dampingFactor=None, n_biomarkers=None, n_clusters=None, n_pc=None, t_degree=None):
		if dampingFactor != None: self.dampingFactor = float(dampingFactor)
		if n_biomarkers != None: self.n_biomarkers = int(n_biomarkers)
		if n_clusters != None: self.n_clusters = int(n_clusters)
		if n_pc != None: self.n_pc = int(n_pc)
		if t_degree != None: self.t_degree = float(t_degree)
		self.isFitted = False
	
	""" get paramters """
	def getParam(self):
		parameters = dict()
		parameters['dampingFactor'] = self.dampingFactor
		parameters['n_biomarkers'] = self.n_biomarkers
		parameters['n_clusters'] = self.n_clusters
		parameters['n_pc'] = self.n_pc
		parameters['t_degree'] = self.t_degree
		return parameters
	
	""" get average-ranked genes """
	def getRankedGenes(self):
		if self.isFitted:
			return self.rankedGenes
		else:
			print("Fitting Error: Please getRankedGenes after fitting with training data")
			exit(1)
		
	""" get biomarkers """
	def getBiomarkers(self):
		if self.isFitted:
			return self.biomarkers
		else:
			print("Fitting Error: Please getRankedGenes after fitting with training data")
			exit(1)
	
	""" validate
	geneList : list() for genes
		ex) [gene1, gene2, ...]
		
	data_test : 2dim np.array() whose columns represent poor prognosis samples for test
	    and rows represent genes ordered in the order on the geneList
		
	label_test : 1dim np.array() for labels of samples
		The order of samples must be mapped with elemenes in data_test
		
	randomState: This parameter is used for scikit-learn functinos
		Default value is set as None.
	"""
	def validate(self, geneList, data_test, label_test, randomState=None):
		if not self.isFitted:
			print("Fitting Error: Please validate after fitting with training data")
			exit(1)
			
		# basic number parameters
		n_genes = len(geneList)
		n_samples = len(data_test)

		# user parameters
		n_trees     = self.n_trees
		
		# fitted information
		biomarkers = self.biomarkers
		
		# 2.4) make gene indices
		# geneToIndex: dict() for (key=geneName, value=index)
		geneToIndex = self._makeGeneIndices(geneList)
		idx_biomarkers = list()
		for biomarker in biomarkers:
			if geneToIndex.has_key(biomarker):
				idx_biomarkers.append(geneToIndex[biomarker])
				
		mat_test = np.array(map(lambda line:line[idx_biomarkers], data_test))
		
		# Spearman ranking
		ranked_mat_test = self._ranking(mat_test, n_samples)
		
		# cross validation
		cv = StratifiedKFold(n_splits=10, random_state=randomState, shuffle=False)
		clf = RandomForestClassifier(n_estimators=n_trees, random_state=randomState)
		
		mean_tpr = .0
		mean_fpr = np.linspace(start=0, stop=1, num=100)
		
		for train, test in cv.split(ranked_mat_test, label_test):		
			probas_ = clf.fit(ranked_mat_test[train], label_test[train]).predict_proba(ranked_mat_test[test])
			
			# compute ROC curve and Area the curve
			fpr, tpr, thresholds = roc_curve(label_test[test], probas_[:,1])
			mean_tpr += interp(mean_fpr, fpr, tpr)
			mean_tpr[0] = .0
		
		mean_tpr /= 10
		mean_tpr[-1] = 1.
		mean_auc = auc(mean_fpr, mean_tpr)
		
		return mean_auc
	
	""" fitting
	geneList : list() for genes
		ex) [gene1, gene2, ...]
	
	edgeList : list() for edges in Network
		ex) [(gene1, gene2), (gene1, gene3), ...]
	
	data_train : 2dim np.array() whose columns represent poor prognosis samples for train
	    and rows represent genes ordered in the order on the geneList
		
	label_train : 1dim np.array() for labels of samples
		The order of samples must be mapped with elemenes in data_train
		
	randomState: This parameter is used for scikit-learn functinos
		Default value is set as None.
	"""
	def fit(self, geneList, edgeList, data_train, label_train, randomState=None):
		# basic number parameters
		n_genes = len(geneList)
		n_edges = len(edgeList)
		n_samples = len(data_train)

		# user parameters
		n_biomarkers= self.n_biomarkers
		n_clusters  = self.n_clusters
		n_pc        = self.n_pc
		n_trees     = self.n_trees
		t_degree    = self.t_degree
		
		# 2.4) make gene indices
		# geneToIndex: dict() for (key=geneName, value=index)
		geneToIndex = self._makeGeneIndices(geneList)
		
		# 2.5) Computing degree of genes in network
		# geneDegrees: dict() for (key=geneName, value=degree in network)
		geneDegrees = self._computeDegree(geneList, edgeList)
		
		# 2.6) calculate degreecut with t_degree
		degreeList = geneDegrees.values()
		degreeList = sorted(degreeList, reverse=True)
		if t_degree == 1:
			degreecut = -1
		else:
			degreecut = degreeList[ int(n_genes * t_degree) ]
		
		"""3. K Means clustering"""
		clusters = list()
		if n_clusters > 1:
			# 3.1) normalizing gene expressions genewise using z-scoring
			zscored_data_train = self._zscoring(data_train.T, n_genes)
			zscored_data_train = zscored_data_train.T    # genewise --> samplewise
			
			# 3.2) PCA to reduce high-dimension
			pca = PCA(n_components=n_pc, random_state=randomState)
			pc_data_train = pca.fit_transform(zscored_data_train)
			
			# 3.3) K Means clustering
			kmeans        = KMeans(n_clusters=n_clusters, random_state=randomState).fit(pc_data_train)
			cluster_train = kmeans.labels_
			
			for i in range(n_clusters):
				clusters.append(dict())
				clusters[i]["label_train"] = label_train[cluster_train == i]
				clusters[i]["data_train"]  = data_train[cluster_train == i]
				
			# 3.4) If some cluster has only one label, then raise cluster error
			for i in range(n_clusters):
				labels = clusters[i]["label_train"]
				if len(labels[labels==1]) == 0 or len(labels[labels==0]) == 0:
					print("Cluster Error: cluster[%d] has only one label" % i)
					exit(1)
					
		elif n_clusters == 1:
			clusters.append(dict())
			clusters[0]["label_train"] = label_train
			clusters[0]["data_train"]  = data_train
			
		
		"""4. PageRank"""
		PRresult = list()
		for i in range(n_clusters):
			tmp_PRresult = self._modifiedPageRank(clusters[i]["data_train"], clusters[i]["label_train"], geneToIndex, edgeList)
			PRresult.append(tmp_PRresult)
		
		"""5. Order genes by avgRank and degree"""
		geneRank = list()
		for i in range(n_clusters):
			geneRank.append(dict())
			for j in range(len(PRresult[i])):
				geneRank[i][geneList[PRresult[i][j]]] = j
		
		PRavgDict = dict()
		for geneName in geneList:
			PRavgDict[geneName] = 0
			for i in range(n_clusters):
				PRavgDict[geneName] -= geneRank[i][geneName]    # Note: To easy sort, we use negative ranking
		
		PRavgList = PRavgDict.items()
		PRavgList = map(lambda elem: (geneToIndex[elem[0]], geneDegrees[elem[0]], elem[1]), PRavgList)
		PRavgList = sorted(PRavgList, key=itemgetter(2,1), reverse=True)
		rankedGeneIndices = map(lambda elem:elem[0], PRavgList)
		self.rankedGenes = deepcopy(map(lambda geneIdx:geneList[geneIdx], rankedGeneIndices))
		
		"""6. collect candidates of biomarker"""
		biomarkers = list()
		for i in range(n_genes):
			geneIdx  = rankedGeneIndices[i]
			geneName = geneList[geneIdx]
			degree   = geneDegrees[geneName]
			if degree > degreecut:
				biomarkers.append(geneList[geneIdx])
				if len(biomarkers) >= n_biomarkers:
					break
		
		self.biomarkers = deepcopy(biomarkers)
		self.isFitted = True		
		
	
	def _ranking(self, X, n_X):
		result = deepcopy(X)
				
		for i in range(n_X):
			vectorLength = len(X[i])
			arr = list()
			
			for j in range(vectorLength):
				arr.append( (j, X[i][j]) )
				
			arr = sorted(arr, key=itemgetter(1))
			arr = map(lambda elem:elem[0], arr)
			
			for j in range(vectorLength):
				 result[i][arr[j]] = j
		
		return result
	
	def _zscoring(self, X, n_X):
		result = deepcopy(X)
		for i in range(n_X):
			mean = X[i].mean()
			std  = X[i].std()
			
			if std > 0.:
				result[i] -= mean
				result[i] /= std
			else:
				result[i] *= 0.
				
		return result
		
		
	def _modifiedPageRank(self, mat, labels, geneToIndex, edgeList):
		"""1. parameters"""
		d          = self.dampingFactor
		iterations = self.iterations
		
		mat_poor = mat[labels == 1].T	# samplewise --> genewise
		mat_good = mat[labels == 0].T   # samplewise --> genewise
		
		n_poor  = len(mat_poor[0])
		n_good  = len(mat_good[0])
		n_genes = len(mat_poor)
		
		# print("n_poor: %d" % n_poor)
		# print("n_good: %d" % n_good)
		# print("n_genes: %d" % n_genes)
		
		initValue = 1. / float(n_genes)
		
		"""2. generate adjacency matrix of weighted network"""
		# 2.1) construct adjacency matrix
		adjMat = np.zeros([n_genes, n_genes]).astype(np.float64)
		
		for edge in edgeList:
			if geneToIndex.has_key(edge[0]) and geneToIndex.has_key(edge[1]):
				x = geneToIndex[edge[0]]
				y = geneToIndex[edge[1]]
				adjMat[x][y] = adjMat[y][x] = 1.
		
		# 2.2) make unweighted network
		un_adjMat = deepcopy(adjMat)
			
		# 2.3) compute t statistics
		TScores = np.zeros(n_genes).astype(np.float64)
		for i in range(n_genes):
			sampleMeans = [mat_poor[i].mean(), mat_good[i].mean()]
			sampleStds  = [mat_poor[i].std(ddof=1), mat_good[i].std(ddof=1)]
		
			# print(sampleMeans)
			# print(sampleStds)
		
			firstDenominator  = sqrt( ((float(n_poor) - 1.)*sampleStds[0]*sampleStds[0] + (float(n_good) - 1.)*sampleStds[1]*sampleStds[1]) / float(n_poor + n_good - 2))
			secondDenominator = sqrt( (1. / float(n_poor)) + (1. / float(n_good)) )
			
			# print(firstDenominator)
			# print(secondDenominator)
			# print("")
				
			tmp_TScore = (sampleMeans[0] - sampleMeans[1]) / (firstDenominator * secondDenominator)
					
			if tmp_TScore < 0.:
				TScores[i] -= tmp_TScore
			else:
				TScores[i] += tmp_TScore
		
		# 2.4) give weights to adjacency matrix
		for x in range(n_genes):
			if TScores[x] > 0:
				adjMat[x] *= TScores[x]
			
		# 2.5) normalize each column to make 'col.sum() = 1'
		adjMat = self._normalize_col(adjMat, n_genes, initValue)
		un_adjMat = self._normalize_col(un_adjMat, n_genes, initValue)
			
		"""3. compute PRScore iteratively"""
		initScore = np.zeros(n_genes) + initValue
		PRScore   = deepcopy(initScore)
		un_PRScore = deepcopy(initScore)
		
		for i in range(iterations):
			PRScore    = (1.-d)*initScore + d*(adjMat.dot(PRScore))
			un_PRScore = (1.-d)*initScore + d*(un_adjMat.dot(un_PRScore))
		
		"""4. sorted genes by PRScore"""
		result = list()
		for i in range(n_genes):
			result.append( (i, PRScore[i] / un_PRScore[i]) )
		
		# reverse=True --> descending order
		result = sorted(result, key=itemgetter(1), reverse=True)
		result = map(lambda elem:elem[0], result)
		
		return result
		

	def _normalize_col(self, mat, n_genes, initValue):	
		mat = mat.T
		for y in range(n_genes):
			sum = mat[y].sum()
			
			if sum > .0:
				mat[y] /= sum
			else:
				mat[y] += initValue
		
		mat = mat.T
		return mat
		
	def _computeDegree(self, geneList, edgeList):
		geneDegrees = dict()
		
		# 1. initialize
		for geneName in geneList:
			geneDegrees[geneName] = 0
			
		# 2. count neighbors
		for edge in edgeList:
			firstGene  = edge[0]
			secondGene = edge[1]
			geneDegrees[firstGene]  += 1
			geneDegrees[secondGene] += 1
			
		return geneDegrees

	def _makeGeneIndices(self, geneList):
		geneToIndex = dict()
		
		n_genes = len(geneList)
		for i in range(n_genes):
			geneName = geneList[i]
			geneToIndex[geneName] = i
		
		return geneToIndex
		
	def _parsingEdgeList(self, commonGenes, edgeList):
		edgeList = list()
		parsedEdgeSet  = set()
		
		for elem in edgeList:
			firstGene  = elem[0]
			secondGene = elem[1]
			if firstGene in commonGenes and secondGene in commonGenes:
				# In undirected graph, (A,B) and (B,A) are equal.
				# To remove redundnacy, we choose (A,B) not (B,A).
				if firstGene > secondGene:
					edge = (secondGene, firstGene)
				else:
					edge = (firstGene, secondGene)
				
				if edge not in parsedEdgeSet:
					edgeList.append(edge)
					parsedEdgeSet.add(edge)
		
		return edgeList
		
	
	def _findCommonGenes(self, geneList, edgeList):
		commonGenes = set()
		for edge in edgeList:
			commonGenes.add(edge[0])
			commonGenes.add(edge[1])
		
		commonGenes = commonGenes.intersection( set(geneList) )
		
		return commonGenes



if __name__=="__main__":
	main(len(sys.argv), sys.argv)