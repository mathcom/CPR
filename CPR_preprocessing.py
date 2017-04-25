from copy import deepcopy
import numpy as np

class NAvalueError(Exception):
	def __init__(self, msg):
		self.msg = msg
		
	def __str__(self):
		return self.msg

def preprocessing_network(data):
	""" data format
	1st row   : header
	remainder : the list of edges
	"""
	edgeList = data[1:]    # remove one header
	edgeSet = set()
	geneset = set()
	for edge in edgeList:
		fgene = edge[0]
		sgene = edge[1]
		
		if fgene != sgene:
			if fgene not in geneset:
				geneset.add(fgene)
			if sgene not in geneset:
				geneset.add(sgene)
		
			elem = (fgene, sgene)
			if fgene > sgene:
				elem = (sgene, fgene)
				
			edgeSet.add(elem)
	
	new_edgeList = list(edgeSet)
	
	return new_edgeList, geneset

def parsing_commonEdge(commonGenes, edgeList):
	new_edgeList = list()
	
	for edge in edgeList:
		fgene = edge[0]
		sgene = edge[1]
		
		if fgene in commonGenes and sgene in commonGenes:
			new_edgeList.append(edge)
		
	return new_edgeList
	
def parsing_commonExpr(commonGenes, data, geneList):
	new_data = list()
	new_geneList = list()
	n_samples = len(data)
		
	ix = map(lambda geneSymbol:geneSymbol in commonGenes, geneList)
	
	for i in range(len(geneList)):
		if ix[i]:
			new_geneList.append(geneList[i])
	
	for i in range(n_samples):
		arr = data[i][ix]
		arr -= arr.mean()
		arr /= arr.std()
		new_data.append(arr)
	
	new_data = np.array(new_data).astype(np.float64)
		
	return new_data, new_geneList
		
def find_commonGenes(genesetList):
	commonGenes = set(genesetList[0])
	
	for i in range(1,len(genesetList)):
		currentset = set(genesetList[i])
		commonGenes = commonGenes.intersection(currentset)
		
	return commonGenes
		
def preprocessing_expression(data):
	""" data format
	1st row   : sample Names
	2nd row   : survival event
	3rd row   : survival time
	remainder : gene expressions
	"""
	# 1. partition
	samples = data[0]
	samples = np.array(samples[1:])
	
	events = data[1]
	try:
		events = np.array(events[1:]).astype(int)   # 1:occured, 0:censored, -1:NA
	except:
		raise NAvalueError("Survival event must have only three values, 1:occured, 0:censored, and -1:NA.")
	
	times = data[2]
	try:
		times = np.array(times[1:]).astype(np.float64)   # -1:NA
	except:
		raise NAvalueError("Survival time must have only numbers.")
	
	data = data[3:]
	
	# 2. decide labels
	""" label rule
	(event==1 and time < 5) ==> 1:poor prognosis
	(event==0 and time > 5) ==> 0:good prognosis
	Otherwise  ==> -1:NA
	"""
	n_samples = len(samples)
	labels = np.zeros(n_samples).astype(int) - 1	
	for i, event, time in zip(range(n_samples), events, times):
		if event == 1 and time < 5:
			labels[i] = 1
		elif event == 0 and time > 5:
			labels[i] = 0
	
	# 3. remove redundant genes
	""" redundant gene rule
	If a gene has two more expression vectors, then
	we compute an average expressions vector of them.
	"""
	geneList = list()
	exprDict = dict()
	redunDict = dict()
	for line in data:
		geneSymbol = line[0]
		expressions = np.array(line[1:]).astype(np.float64)
		
		if exprDict.has_key(geneSymbol):
			exprDict[geneSymbol] += expressions
			redunDict[geneSymbol] += 1
		else:
			geneList.append(geneSymbol)
			exprDict[geneSymbol] = expressions
			redunDict[geneSymbol] = 1
			
	for geneSymbol in geneList:
		redundancy = redunDict[geneSymbol]
		if redundancy > 1:
			exprDict[geneSymbol] /= float(redundancy)
	
	# 3. make new 
	new_samples = samples[labels > -1]
	new_labels = labels[labels > -1]
	new_data = list()
	for geneSymbol in geneList:
		new_data.append(exprDict[geneSymbol][labels > -1])
	new_data = np.array(new_data).T
	
	return new_samples, new_labels, new_data, geneList
	
def preprocessing_expression_test(data):
	""" data format
	1st row   : sample Names
	remainder : gene expressions
	"""
	# 1. partition
	samples = data[0]
	samples = np.array(samples[1:])
		
	data = data[1:]
		
	# 2. remove redundant genes
	""" redundant gene rule
	If a gene has two more expression vectors, then
	we compute an average expressions vector of them.
	"""
	geneList = list()
	exprDict = dict()
	redunDict = dict()
	for line in data:
		geneSymbol = line[0]
		expressions = np.array(line[1:]).astype(np.float64)
		
		if exprDict.has_key(geneSymbol):
			exprDict[geneSymbol] += expressions
			redunDict[geneSymbol] += 1
		else:
			geneList.append(geneSymbol)
			exprDict[geneSymbol] = expressions
			redunDict[geneSymbol] = 1
			
	for geneSymbol in geneList:
		redundancy = redunDict[geneSymbol]
		if redundancy > 1:
			exprDict[geneSymbol] /= float(redundancy)
	
	# 3. make new 
	new_data = list()
	for geneSymbol in geneList:
		new_data.append(exprDict[geneSymbol])
	new_data = np.array(new_data).T
	
	return samples, new_data, geneList