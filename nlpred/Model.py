import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle
from typing import List
import multiprocessing
import tqdm
from TextPrep import prepareTestTrainData
from TextPrep import preparePredData

class Model:

    def __init__(
        self, topicQuantity: int, fit: LogisticRegressionCV, topicLDAModel: LatentDirichletAllocation, 
        count_vectorizer: CountVectorizer, confusion_matrix, accuracy: float, recall: float, AUC: float):

        ############# INSTANCE VARIABLES #############

        # 1) "topicQuantity", the number of topics used in .this model
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
    
    ## Predicts True/False (1/0) for each document (String) in the List listOfDocStrings argument using the model stored in 
    #   this Model object.
    #
    # @arg listOfDocStrings a List of Strings wherein each String contains the text of a document to predict True/False from
    #
    # @return a DataFrame containing three columns:
    #           1)  'paper_text', the original input textual data
    #           2)  'paper_text_processed', the processed (tokenized with punctuation and capitalization removed) textual data
    #           3)  'value', the 1/0 (True/False) prediction for the 'paper_text' entry in the same row
    def predict(self, listOfDocStrings: List[str]):
        predDict = preparePredData(listOfDocStrings)
        # Create the Document Topic Matrix
        docTopMat = self.topicLDAModel.transform(predDict['count_data'])
        # Predict for each provided document
        y_pred = self.fit.predict(docTopMat)
        # Add the predictions to predDict
        predDict['processedCorpusDataFrame']['value'] = y_pred
        return predDict['processedCorpusDataFrame']
    
    ## Prints to stdout a summary of the model.
    def print(self):
        print("Topic Quantity : ", self.topicQuantity)
        print("Confusion Matrix : \n", self.confusion_matrix)
        print("Accuracy : ", self.accuracy)
        print("Recall : ", self.recall)
        print("AUC : ", self.AUC)

    ## Permits the standard print() function to print a summary of the model when called with a Model object as an argument.
    def __str__(self):
        tbr = (
            f"Topic Quantity : {self.topicQuantity}\n"
            f"Confusion Matrix : \n{self.confusion_matrix}\n"
            f"Accuracy : {self.accuracy}\n"
            f"Recall : {self.recall}\n"
            f"AUC : {self.AUC}"
        )
        return tbr
    
    ## Saves the model as a .pkl file
    #
    # @arg fileName the desired filename, including the .pkl extension.  e.g., "my_model.pkl"
    #
    # @postState the model object will be saved as fileName in the current working directory.
    def save(self, fileName: str = "my_model.pkl"):
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)

## Loads a model saved as a .pkl file
#
# @arg fileName the file name of the saved model, including the .pkl extension.  e.g., "my_model.pkl"
#
# @return a Model object of the model saved in the specified .pkl file.
def loadModel(fileName: str) -> Model:
    with open(fileName, 'rb') as file:
        tempModel = pickle.load(file)
    return tempModel

## Predicts True/False (1/0) for each document (String) in the List listOfDocStrings argument using the model stored in 
#   the indicated file location.
#
# @arg fileName the file name of the saved model, including the .pkl extension.  e.g., "my_model.pkl"
# @arg listOfDocStrings a List of Strings wherein each String contains the text of a document to predict True/False from
#
# @return a DataFrame containing three columns:
#           1)  'paper_text', the original input textual data
#           2)  'paper_text_processed', the processed (tokenized with punctuation and capitalization removed) textual data
#           3)  'value', the 1/0 (True/False) prediction for the 'paper_text' entry in the same row
def predictFromFile(fileName: str, listOfDocStrings: List[str]):
    tempModel = loadModel(fileName)
    tbr = tempModel.predict(listOfDocStrings)
    return tbr

## Creates a Model object for each topic quantity specified in the argument topicQuantityVector.
# Python's implementation of multithreading, mutiprocessing, is used in this method to vastly decrease runtime.
# Multiprocessing is used instead of Python's "multithreading" since this method is entirely CPU bound and involves no file/internet I/O.
#
# @arg topicQuantityVector a vector of the topic quantities to test, e.g. [30,40,50,60,70,80].
#                           Note that the values in this vector must be integers.
#                           Note that if only a single quantity is desired, it must be passed as
#                               a list with one element, e.g. topicQuantityVector = [10].
# @arg count_data the input to LDA.fit(...)
# @arg count_vectorizer the CoutVectorizer object used int he creation of the Term-Document Matrix
# @arg responseValues the response, Y, for the input data; a vector;
#                       this is 'values' from the DataFrame 'processedCorpusDataFrame' which is in the Dictionary returned
#                       from the prepareTestTrainData() method in TextPrep.py
# @arg max_iterations the maximum number of iterations for the Stochastic Average Gradient Descent ('saga') solver 
#                       to go through for the elastic net CV.  Default is 4900.  Lower numbers drastically decrease 
#                       runtime for large datasets, but may create inferior models.
#
# @return a list of Model objects, one for each fitted model based on the topic quantities provided in the argument topicQuantityVector.
def modelMultiProc(topicQuantityVector: List[int], count_data, count_vectorizer: CountVectorizer, responseValues: List[bool], max_iterations: int = 4900):
    # Check for the number of simultaneous threads that can be supported by current hardware and OS
    numProcesses = multiprocessing.cpu_count()
    # Don't allocate extra threads if there are less quantities (models to create) than available threads
    if(numProcesses > len(topicQuantityVector)):
        numProcesses = len(topicQuantityVector)
    # Break the list of topic quantities into numProcesses parts
    topicQuantityChunks = np.array_split(topicQuantityVector, numProcesses)
    # Note that topicQuantityChunks is a List of Lists of topic quantities to be used.
    # topicQuantityChunks[i] is a list of the topic quantities to be used by thread i, for i an element of [0,numProcesses).
    
    tbrList = []
    # Instantialize a Manager object
    #manager = multiprocessing.Manager()
    if __name__ == "__main__":
        with multiprocessing.Manager() as manager:
            # Instantialize a Lock object (conntainer for operating system's Semaphore object)
            lock = manager.Lock()
            # Create an array to hold the threads
            threads = []
            # Create a proxy list to be altered by the threads
            managerList = manager.list([])
            # Generate threads
            for i in range(numProcesses):
                thread = multiprocessing.Process(target=_modelMultiProcHelper, args=(topicQuantityChunks[i], count_data, count_vectorizer, responseValues, managerList, lock, max_iterations))
                threads.append(thread)
                if __name__ == "__main__":
                    print(f"Thread {i} started...")
                    thread.start()
            # Allow all threads to complete
            for thread in threads:
                if __name__ == "__main__":
                    thread.join()
            tbrList = managerList.__deepcopy__({})
        # Return a List of Model objects generated by the threads
        #return managerList
        return tbrList
    else:
        print("__name__ != '__main__'")

## Helper method for modelMultiProc().
# Calls modelWithNTopics(), then appends the returned list of generated models to the managerList argument.  Potential race conditions are accounted for with locks.
#
# @arg managerList the proxy list generated from multiprocessing.Manager.list() in the modelMultiProc() method
# @arg lock the multiprocessing.Lock() object instantiated in the modelMultiProc() method.  This is a Semaphore object from the operating system.
# @see modelWithNTopics method for further argument details.
#
# @poststate the proxy list "managerList" is appended with the models generated for the topic quantities listed in the argument topicQuantityVector.
def _modelMultiProcHelper(topicQuantityVector: List[int], count_data, count_vectorizer: CountVectorizer, responseValues: List[bool], managerList: List[Model], lock: multiprocessing.Lock, max_iterations: int = 4900):
    fitList = modelWithNTopics(topicQuantityVector, count_data, count_vectorizer, responseValues, max_iterations)
    lock.acquire()
    for fit in fitList:
        managerList.append(fit)
    lock.release()
    

## Creates a Model object for each topic quantity specified in the argument topicQuantityVector.
#
# @arg topicQuantityVector a vector of the topic quantities to test, e.g. [30,40,50,60,70,80].
#                           Note that the values in this vector must be integers.
#                           Note that if only a single quantity is desired, it must be passed as
#                               a list with one element, e.g. topicQuantityVector = [10].
# @arg count_data the input to LDA.fit(...)
# @arg count_vectorizer the CoutVectorizer object used int he creation of the Term-Document Matrix
# @arg responseValues the response, Y, for the input data; a vector;
#                       this is 'values' from the DataFrame 'processedCorpusDataFrame' which is in the Dictionary returned
#                       from the prepareTestTrainData() method in TextPrep.py
# @arg max_iterations the maximum number of iterations for the Stochastic Average Gradient Descent ('saga') solver 
#                       to go through for the elastic net CV.  Default is 4900.  Lower numbers drastically decrease 
#                       runtime for large datasets, but may create inferior models.
#
# @return a list of Model objects, one for each fitted model based on the topic quantities provided in the argument topicQuantityVector.
def modelWithNTopics(topicQuantityVector: List[int], count_data, count_vectorizer: CountVectorizer, responseValues: List[bool], max_iterations: int = 4900):
    
    fitList = []
    for topicQuantity in topicQuantityVector:
        number_topics = topicQuantity
        
        # Create and fit the LDA model
        lda = LatentDirichletAllocation(n_components=number_topics)
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

        # Create the Model object to add to the list
        tempModel = Model(number_topics, fit, lda, count_vectorizer, cm, acc, rec, auc)

        # Add the Model object to the list
        fitList.append(tempModel)

    return fitList

# Test Code

##### modelWithNTopics() method #####
# preppedData = prepareTestTrainData('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', isRedditData=True)
# count_d = preppedData['count_data']
# count_v = preppedData['count_vectorizer']
# yVals = preppedData['processedCorpusDataFrame']["value"].tolist()
# modelList = modelWithNTopics([10], count_d, count_v, yVals)

##### File I/O #####
# pkl_filename = "test_model.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(modelList[0], file)
    
# with open("optimal_reddit_suicide_model.pkl", 'rb') as file:
#     test_model = pickle.load(file)

##### Model.predict() #####
# from TextPrep import _getSubmissionStringArray
# testDocsToTest1 = _getSubmissionStringArray('SuicideWatchRedditJSON.json')
# testDocsToTest2 = _getSubmissionStringArray('AllRedditJSON.json')
# testDocStringList = testDocsToTest1 + testDocsToTest2
# testModel = loadModel("optimal_reddit_suicide_model.pkl")
# predDict = testModel.predict(testDocStringList)

##### modelMultiProc() method #####
preppedData = prepareTestTrainData('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', isRedditData=True)
count_d = preppedData['count_data']
count_v = preppedData['count_vectorizer']
yVals = preppedData['processedCorpusDataFrame']["value"].tolist()
modelList = modelMultiProc([10,15,20,25], count_d, count_v, yVals)
