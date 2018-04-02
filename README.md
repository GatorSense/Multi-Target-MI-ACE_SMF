# MTMIACE
Multi Target - Multiple Instance - Adaptive Cosine Estimator

main.m will run MTMIACE.

	You will need to bag your data as follows:

		data.dataBags: bagged data
			* a positive bag should have at least one positive instance in it
			* a negative bag should consist of all negative instances

		data.labels: labels for dataBags
			* the labels should be a row vector with labels corresponding to the 
			* parameters.posLabel and parameters.negLabel where a posLabel corresponds
			* to a positive bag and a negLabel corresponds to a negative bag.
			* The index of the label should match the index of the bag in dataBags
			
The code is still being developed and not all init types may work. This code was revamped from single target to multi target and therefor not all functionality may be working.
Best results are currently seen using init type 4 for multiple target learning and optimize set to 1. 