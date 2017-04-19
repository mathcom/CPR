import CPR

if __name__=="__main__":
	examples = CPR.load_data()
	
	edgeList = examples['edgeList']
	geneList = examples['geneList']
	data_train = examples['data_BRCA']
	label_train = examples['label_BRCA']
	data_test = examples['data_GSE4922']
	label_test = examples['label_GSE4922']
	
	# 3. generate CPR
	print(">>> fitting...")
	cpr = CPR.CPR()
	cpr.setParam(dampingFactor=0.5, n_biomarkers=70)
	cpr.fit(geneList, edgeList, data_train, label_train, randomState=1)
	
	print("The number of biomarkers: %d" % len(cpr.biomarkers))
	cpr.getParam()
	
	auc = cpr.estimate(geneList, data_test, label_test, randomState=1)
	print("AUC = %g" % auc)
	
	cpr.print_rankedGenes(5)
	
	cpr.print_biomarkers(5)