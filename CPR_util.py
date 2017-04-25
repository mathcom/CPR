from __future__ import print_function

class NoFileError(Exception):
	def __init__(self, msg):
		self.msg = msg
		
	def __str__(self):
		return self.msg

def fwrite_summary(resultFile, AUC, biomarkers, subEdgeList):
	n_biomarkers = len(biomarkers)
	n_edges = len(subEdgeList)
	
	fout = open(resultFile, 'w')
	fout.write("=====  RESULT  =================\n")
	fout.write("Accuracy(=AUC): %.3f\n" % AUC)
	fout.write("\n")
	
	fout.write("=====  Biomarkers (%d)  ========\n" % n_biomarkers)
	for i in range(n_biomarkers):
		fout.write("%s" % biomarkers[i])
		if i % 10 == 9:
			fout.write("\n")
		else:
			fout.write("\t")
	
	fout.write("\n")
	fout.write("=====  Subnetwork (%d)  ========\n" % n_edges)
	for edge in subEdgeList:
		fout.write("%s\t%s\n" % edge)
		
	fout.close()
	
def print_summary(AUC, biomarkers, subEdgeList):
	n_biomarkers = len(biomarkers)
	n_edges = len(subEdgeList)
	print("")
	print("=====  RESULT  =================")
	print("Accuracy(=AUC): %.3f" % AUC)
	
	print("")
	print("=====  Biomarkers (%d)  ========" % n_biomarkers)
	for i in range(n_biomarkers):
		print("%s" % biomarkers[i], end="")
		if i % 10 == 9:
			print("")
		else:
			print("\t", end="")
	
	print("")
	print("=====  Subnetwork (%d)  ========" % n_edges)
	for edge in subEdgeList:
		print("%s\t%s" % edge)
	
	
	
def find_subnetwork(biomarkers, edgeList):
	biomarkerSet = set(biomarkers)
	subEdgeList = list()
	for edge in edgeList:
		fgene = edge[0]
		sgene = edge[1]
		if fgene in biomarkerSet and sgene in biomarkerSet:
			subEdgeList.append(edge)
		
	return subEdgeList

		
def read_file(dataFile):
	try:
		fin = open(dataFile)
	except:
		raise NoFileError("No file: %s" % dataFile)
	else:	
		lines = map(lambda line:line.rstrip().split('\t'), fin.readlines())
		fin.close()
	
	return lines
	
	

