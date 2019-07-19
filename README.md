# Grammatical Tagging using Hidden Markov model 

- The project aims to build a generic part of speech tagger for any given training data. 
- A Hidden Markov model built from scratch in python has been used to learn the given data sets for part of speech tagging. 
- The Hidden moarkov model designed in this project was tested on two datasets - Italian and Japanese. With an average accuracy of 95%. 

## Files description 
- hmmlearn3.py - Reads a dataset containing sentences with each word tagged with the part of speech and builds a Hidden Markov model for the learnt language and stores this model in the file hmmmodel.txt 
- hmmdecode.py - Reads the model hmmmodel.txt. Given any sentence, it will provide the part of speech tags to every word in the sentence. 
