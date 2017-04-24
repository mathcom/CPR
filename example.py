from __future__ import print_function
import CPR

if __name__=="__main__":
	""" 1. How to load data """
	# A return value of load_data() is 'dictionary'.
	# User can check keys using '.keys()'
	examples = CPR.load_data()
	# print(examples.keys())
	
	edgeList = examples['edgeList']
	geneList = examples['geneList']
	data_train = examples['data_BRCA']
	label_train = examples['label_BRCA']
	data_test = examples['data_GSE4922']
	label_test = examples['label_GSE4922']
	
	""" 2. How to fit classification """
	# If user does not give parameter, the instance is initialized with default values.
	# The default values are described in 'README'.
	cpr = CPR.CPR()
	
	# User can chage parameters. Detail information is written in 'README'.
	cpr.setParam(dampingFactor=0.5, n_biomarkers=70)
	
	# To fit model, user must provide 4 data.
	cpr.fit(geneList, edgeList, data_train, label_train, randomState=1)
	
	""" 3. How to use the fitted model """
	# The fitted model provides only one measure, the area of under curve (AUC).
	auc = cpr.estimate(geneList, data_test, label_test, randomState=1)
	print("AUC = %g" % auc)
	
	""" 4. How to get information for model """
	# prints parameters of a model.
	cpr.getParam()
	
	# prints prioritized genes.
	# If user does not give any parameter, then all genes are printed.
	cpr.print_rankedGenes(5)
	
	# prints biomarkers.
	# If user does not give any parameter, then all biomarkers are printed.
	cpr.print_biomarkers(5)
	
	# User can directly access to variables in class.
	print("The number of biomarkers: %d" % len(cpr.biomarkers))
	