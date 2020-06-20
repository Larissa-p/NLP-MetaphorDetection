# NLP-MetaphorDetection
The codebase comprises of 2 files:
	1. Model1.py
	2. Model2.py
	3. configuration.json
	
1. Model1.py:

	The input file names (train, test and validation) are maintained in configuration.json
	
	Output will print the following: Precision, Recall, F1-Score, Support
	
	The file ValOut.txt contains the prediction results on the validation dataset.
	Similarly, the file testOut.txt contains the prediction results on the test dataset.
	
	
2. Model2.py:

	The input file names (train, test and validation) are maintained in configuration.json
		
	Output will print the following: Precision, Recall, F1-Score, Support

	The file ValOut.txt contains the prediction results on the validation dataset
	Similarly, the file testOut.txt contains the prediction results on the test dataset
	
	IMPORTANT: The functions train_maxEnt() takes a while to run (~15-17 min). This time is because of the maxent.MaxentClassifier.train() function which trains the model on the features: word, part of speech of that word, length of the word,position of the word , part of speech of previous word, part of speech of next word
	
