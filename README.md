# NLP-MetaphorDetection
The codebase comprises of 2 files:
	1. HMMMode1.py
	2. HMMModel2.py
	
1. HMMMode1.py:

	Change the paths ksTrainPath, ksValPath, ksValOutPath to reflect to files in your system. Please note that ksValOutPath is the file where we output the predcted tags for the val/test file on basis of what ksValPath you run it on.
	
	Output will print the following: Precision, Recall, F1-Score, Support
	
	Just run each cell one by one on jupyter notebook. In case running the python code, run the main() method.
	
2. HMM+Model+2.ipynb:

	Change the paths ksTrainPath, ksValPath, ksValOutPath to reflect to files in your system. Please note that ksValOutPath is the file where we output the predcted tags for the val/test file on basis of what ksValPath you run it on.
	
	Output will print the following: Precision, Recall, F1-Score, Support
	
	IMPORTANT: The functions train_maxEnt() takes a while to run (~15-17 min). This time is because of the maxent.MaxentClassifier.train() function which trains the model on the features: word, part of speech of that word, length of the word,position of the word , part of speech of previous word, part of speech of next word
	
