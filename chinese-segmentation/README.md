# Chinese segmentation
Unlike English, Chinese words are not seperated by spaces in sentences. Word segmentation is important. Here, I implemented a simple and effective approach to Chinese segementation. 
- Extract 4-gram from string data. Break down each 4-gram "abcd" into pattern [ab, b, bc, c, cd]. 
- Encode the 1 and 2 grams in the pattern as one hot vectors using sklearn sparse representation, since the vocabulary is huge. Sum up the five smaller gram representations to form one vector as input sample. A binary value indicating whether there is a space between "b" and "c" is the output of the model.
- Train using logistic regression or other types of classifier.
