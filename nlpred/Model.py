import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
from TextPrep import prepareTestTrainData
from TextPrep import preparePredData

class Model:

    def __init__(self, topicQuantity, fit, topicLDAModel, count_vectorizer, confusion_matrix, accuracy, recall, AUC):

        ############# INSTANCE VARIABLES #############

        # 1) "topicQuantity", the number of topics used in that model
        self.topicQuantity = topicQuantity
        # 2) "fit", the fitted model itself
        self.fit = fit
        # 3) "topicLDAModel", the LDA object for the fitted topic model
        self.topicLDAModel = topicLDAModel
        # 4) "count_vectorizer", the CountVectorizer object used for the LDA
        self.count_vectorizer = count_vectorizer
        # 5) "confusion_matrix":
        # Confusion matrix format, (Actual value, predicted value):
        # [[ (A-0,P-0)  (A-0, P-1)]
        # [ (A-1, P-0) (A-1, P-1)]]
        self.confusion_matrix = confusion_matrix
        # 6) "accuracy", the classification rate, i.e. the percent of correct predictions.
        #   i.e., the number of correct predictions made divided by the total number of predictions made.
        self.accuracy = accuracy
        # 7) "recall", a.k.a SENSITIVITY; P(Y=1 | X=1), proportion of true samples correctly identified as true.
        #   i.e., the number of true samples correctly identified divided by the total number of true samples.
        #   The recall is intuitively the ability of the classifier to find all the positive samples.
        self.recall = recall
        # 8) "AUC", a general measure of fit for the model as per the chosen predictive threshold for "true" vs 
        #       "false" from logreg predictive output
        self.AUC = AUC
    
    # Predicts True/False (1/0) for each document (String) in the List listOfDocStrings argument using the model stored in 
    #   this Model object.
    # @arg listOfDocStrings a List of Strings wherein each String contains the text of a document to predict True/False from
    # @return a DataFrame containing two columns:
    #           1)  'paper_text', the original input textual data
    #           2)  'paper_text_processed', the processed (tokenized with punctuation and capitalization removed) textual data
    #           3)  'value', the 1/0 (True/False) prediction for the 'paper_text' entry in the same row
    def predict(self, listOfDocStrings):
        pass

# Creates a Model object for each topic quantity specified in the argument topicQuantityVector.
# @arg topicQuantityVector a vector of the topic quantities to test, e.g. [30,40,50,60,70,80].
#                           Note that the values in this vector must be integers.
# @arg count_data the input to LDA.fit(...)
# @arg count_vectorizer the CoutVectorizer object used int he creation of the Term-Document Matrix
# @arg responseValues the response, Y, for the input data; a vector;
#                       this is 'values' from the DataFrame 'processedCorpusDataFrame' which is in the Dictionary returned
#                       from the prepareTestTrainData() method in TextPrep.py
# @arg max_iterations the maximum number of iterations for the Stochastic Average Gradient Descent ('saga') solver 
#                       to go through for the elastic net CV.  Default is 4900.  Lower numbers drastically decrease 
#                       runtime for large datasets, but may create inferior models.
# @return a list of Model objects, one for each fitted model based on the topic quantities provided in the argument topicQuantityVector.
def modelWithNTopics(topicQuantityVector, count_data, count_vectorizer, responseValues, max_iterations = 4900):
    
    fitList = []
    for topicQuantity in topicQuantityVector:
        number_topics = topicQuantity
        
        # Create and fit the LDA model
        lda = LDA(n_components=number_topics)
        lda.fit(count_data)
        
        # Create a topic count matrix for the topic distribution of each document
        docTopics = lda.transform(count_data)  
        # docTopics is a numpy.ndarray object
        
        #Split into test and train sets
        xtrain, xtest, ytrain, ytest = train_test_split(docTopics, responseValues, test_size = 0.25, random_state = 0) 
        
        # Do the cross-validated logistic regression with Elastic Net regularization
        model = LogisticRegressionCV(penalty = 'elasticnet', solver = 'saga', cv = 10, max_iter = max_iterations, l1_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) 
        fit = model.fit(xtrain, ytrain)
        
        # test the model
        y_pred = fit.predict(xtest)

        # Confusion matrix format, (Actual value, predicted value):
        # [[ (A-0,P-0)  (A-0, P-1)]
        # [ (A-1, P-0) (A-1, P-1)]]
        cm = confusion_matrix(ytest, y_pred) 
        
        # Accuracy: The classification rate, i.e. the percent of correct predictions.
        # i.e.,  the number of correct predictions made divided by the total number of predictions made.
        acc = accuracy_score(ytest, y_pred)
        
        # Recall: a.k.a SENSITIVITY; P(Y=1 | X=1), proportion of true samples correctly identified as true.
        #   i.e., the number of true samples correctly identified divided by the total number of true samples.
        #   The recall is intuitively the ability of the classifier to find all the positive samples.
        rec = recall_score(ytest, y_pred)
        
        # AUC: 
        y_pred_proba = fit.predict_proba(xtest)[::,1]
        auc = roc_auc_score(ytest, y_pred_proba)
        
        # Create the dictionary to add to the list
        # tempDict = {
        #     "topicQuantity" : number_topics,
        #     "fit" : fit,
        #     "topicLDAModel" : lda,
        #     "count_vectorizer" : count_vectorizer,
        #     "confusionMatrix" : cm,
        #     "accuracy" : acc,
        #     "recall" : rec,
        #     "AUC" : auc
        # }
        # Add the dictionary to the list
        #fitList.append(tempDict)

        # Create the Model object to add to the list
        tempModel = Model(number_topics, fit, lda, count_vectorizer, cm, acc, rec, auc)

        # Add the Model object to the list
        fitList.append(tempModel)

    return fitList

# Test Code
# preppedData = prepareTestTrainData('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', isRedditData=True)
# count_d = preppedData['count_data']
# count_v = preppedData['count_vectorizer']
# yVals = preppedData['processedCorpusDataFrame']["value"].tolist()
# modelList = modelWithNTopics([10,20], count_d, count_v, yVals)