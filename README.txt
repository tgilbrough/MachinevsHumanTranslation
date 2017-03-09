Dependencies:
nltk -- http://www.nltk.org/
Stanford parser -- http://nlp.stanford.edu/software/lex-parser.shtml
sklearn -- http://scikit-learn.org/stable/
pandas -- http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html

How to run code:

Simply type:
python predict.py
This will run a SVM classifier using the provided training data along with the preprocessed trees and output text in the predicted file format.
Redirect the output to a file such as "A5.test.predicted" to save the results.

Preprocess trees:
If using new data, run the create_tree.py script that is provided with the code. Change the file name at the top of the script and capture
the output in a file such as "A5.train_trees.labeled". If the trees are save to a different file name, just remember to also change the filenames
within util.py under the data parsing functions.