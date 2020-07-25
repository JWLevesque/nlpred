import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd
import pickle
from typing import List
import multiprocessing
import tqdm
from uuid import uuid4
from TextPrep import prepareTestTrainData
from TextPrep import preparePredData

class Model:

    def __init__(
        self, topicQuantity: int, fit: LogisticRegressionCV, topicLDAModel: LatentDirichletAllocation, 
        confusion_matrix, accuracy: float, recall: float, AUC: float):

        ############# INSTANCE VARIABLES #############

        # 1) "topicQuantity", the number of topics used in .this model
        self.topicQuantity = topicQuantity
        # 2) "fit", the fitted model itself
        self.fit = fit
        # 3) "topicLDAModel", the LDA object for the fitted topic model
        self.topicLDAModel = topicLDAModel
        # 4) "confusion_matrix":
        # Confusion matrix format, (Actual value, predicted value):
        # [[ (A-0,P-0)  (A-0, P-1)]
        # [ (A-1, P-0) (A-1, P-1)]]
        self.confusion_matrix = confusion_matrix
        # 5) "accuracy", the classification rate, i.e. the percent of correct predictions.
        #   i.e., the number of correct predictions made divided by the total number of predictions made.
        self.accuracy = accuracy
        # 6) "recall", a.k.a SENSITIVITY; P(Y=1 | X=1), proportion of true samples correctly identified as true.
        #   i.e., the number of true samples correctly identified divided by the total number of true samples.
        #   The recall is intuitively the ability of the classifier to find all the positive samples.
        self.recall = recall
        # 7) "AUC", a general measure of fit for the model as per the chosen predictive threshold for "true" vs 
        #       "false" from logreg predictive output
        self.AUC = AUC
        # 8) "id", a unique identifier for the model
        #       Note that this is a UUID object.  The actual id may be found in id.hex.
        #       The UUID wrapper object is kept for aesthetic reasons with regard to printing, etc.
        self.id = uuid4()
    
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
    
    ## Saves the model as a .pkl file
    #
    # @arg fileName the desired filename, including the .pkl extension.  e.g., "my_model.pkl"
    #
    # @postState the model object will be saved as fileName in the current working directory.
    def save(self, fileName: str = "my_model.pkl"):
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
    
    ## Prints to stdout a summary of the model.
    def print(self):
        print("Topic Quantity : ", self.topicQuantity)
        print("Model Id : ", self.id)
        print("Accuracy : ", self.accuracy)
        print("Recall : ", self.recall)
        print("AUC : ", self.AUC)

    ## Permits the standard print() function to print a summary of the model when called with a Model object as an argument.
    def __str__(self):
        tbr = (
            f"Topic Quantity : {self.topicQuantity}\n"
            f"Model Id : {self.id}\n"
            f"Accuracy : {self.accuracy}\n"
            f"Recall : {self.recall}\n"
            f"AUC : {self.AUC}"
        )
        return tbr
    
    ## Provides sorting logic for Model objects.
    # Model objects will be sorted based on AUC.
    def __lt__(self, other):
        return self.AUC < other.AUC
    def __le__(self, other):
        return self.AUC <= other.AUC
    def __gt__(self, other):
        return self.AUC > other.AUC
    def __ge__(self, other):
        return self.AUC >= other.AUC
    
    ## Provides equality definition for Model objects.
    # Model objects with the same Id are the "same" model.
    def __eq__(self, other):
        return isinstance(other, Model) and self.id.int == other.id.int
    
    ## Provides inequality definition for Model objects.
    # Model objects with different Ids are not the "same" model.
    def __ne__(self, other):
        return not self == other


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

    
## Prepares textual input for model creation, then creates Model objects for various topic quantities.
# The generated models are evaluated and five possible optimal model candidates are returned in a Dictionary object.
#
# The "best" model depends on the original corpus and purpose of the model, so it is left
# to the user to determine which of the five models returned by this method best suits their purposes.
#
# If the user is unsure of which model to use, it would behoove them to simply choose
# the model with the highest AUC, stored as "AUC" in the return Dictionary.
#
# Note that this is the main method of the nlpred package; it can be used as a black box for model creation by simply
#   passing the arguments "trueSource", "falseSource", and "isRedditData" while leaving the rest of the arguments as 
#   their defaults.
#
# @arg trueSource Either:
#                   1) if isRedditData = true, the filename of the JSON file containing "true" Strings, including the .json extension
#                   2) if isRedditData = false, an array of Strings which are known to be associated with a "true"/"1" prediction
# @arg falseSource Either:
#                   1) if isRedditData = true, the filename of the JSON file containing "false" Strings, including the .json extension
#                   2) if isRedditData = false, an array of Strings which are known to be associated with a "false"/"0" prediction
# @arg isRedditData Boolean:  
#                   1) True indicates the data is given as reddit JSON objects, and _getCorpusFromReddit will be called.
#                   2) False indicates the data is given as vectors of Strings, and _getCorpusFromVectors will be called.
#                 Note that redditData is TRUE by default.
# @arg max_enet_iterations the maximum number of iterations for the Stochastic Average Gradient Descent ('saga') solver 
#                           to go through for the elastic net CV.  Default is 4900.  Lower numbers drastically decrease 
#                           runtime for large datasets, but may create inferior models.
# @arg range_iterations the number of models to generate for each topic quantity in range(start, stop+1, step = step).
#                                Setting this to 0 will disable the model generation dictated by start, stop,
#                                   and focus.  i.e., models will be only generated in the Focus range.
#                                Must be a non-negative integer.  Default is 5.
# @arg focus_iterations the number of models to generate for each topic quanity in the interval [focus-5, focus+5].
#                                Setting this to 0 will disable extra model generation in the focus range.
#                                Must be a non-negative integer.  Default is 20.                                  
# @arg start the smallest quantity of topics to generate a model with.  Must be an integer.  Default is 10.
# @arg stop the largest quantity of topics to generate a model with.  Must be an integer.  Default is 50.
# @arg step the integer value to count by for the topic quantities to generate models with. Must be an integer.  Default is 5.
#                   e.g., if step = 5, start = 10, and stop = 50, then models will be generated for each of
#                       the following topic quantities: [10,15,20,25,30,35,40,45,50].
# @arg focus the user's best guess for the optimal number of topics.  Must be an integer.  Default is 10.
#                    Additional models will be generated for each topic quantity in the interval [focus-4, focus+4].
# @arg appendFullModelList boolean; if True, the full list of models generated sorted by AUC in 
#                              descending order will be appended to the return Dictionary.
#                          Default is False.
#
# @return A Dictionary containing five candidates for an optimal model, stored as Model objects:
#           1) "AUC", the model with the highest AUC
#           2) "acc", the model with the highest accuracy
#           3) "rec", the model with the highest recall
#           4) "acc_1se", the model with the highest accuracy among models whose AUC is within 1 standard error of the maximum
#           5) "rec_1se", the model with the highest recall among models whose AUC is within 1 standard error of the maximum
#         If optional argument appendFullModelList = True (default is False):
#           6) "modelList", a list containing Model objects for all models generated, sorted by AUC in descending order.
# 
# A reminder:
# Accuracy: a.k.a. SPECIFICITY, the classification rate; i.e. the percent of correct predictions.
#   i.e.,  the number of correct predictions made divided by the total number of predictions made.
#        
# Recall: a.k.a SENSITIVITY; P(Y=1 | X=1), proportion of true samples correctly identified as true.
#   i.e., the number of true samples correctly identified divided by the total number of true samples.
#   The recall is intuitively the ability of the classifier to find all the positive samples.
def getModels(
        trueSource, falseSource, isRedditData: bool = True, max_enet_iterations: int = 4900, 
        range_iterations: int = 5, focus_iterations: int = 10, start: int = 10, stop: int = 50, 
        step: int = 5, focus: int = 10, appendFullModelList: bool = False):
    
    # Preprocess the corpus
    preppedData = prepareTestTrainData(trueSource, falseSource, isRedditData)
    # Extract arguments for _modelMultiProc() method from preppedData
    count_d = preppedData['count_data']
    yVals = preppedData['processedCorpusDataFrame']["value"].tolist()

    # Populate a list with the topic quantities to create models for
    quantityList = np.arange(start= start, stop= stop + 1, step= step).tolist()
    # Add the 9 quantities, counting by 1, around focusQuantity to the list
    if(focus < 6):
        focusStart = 2
    else:
        focusStart = focus - 4
    # range() is exclusive on the stop parameter, so focus+5 gives an interval of +/- 4 from focus
    focusStop = focus + 5
    focusList = [*range(focusStart, focusStop)]
    # Add duplicate quantities as per the specified number of quantity iterations
    focusList = focusList * focus_iterations
    quantityList = quantityList * range_iterations
    # Note that if either of the iterations arguments are 0, their corresponding List will be empty
    # e.g., if quantity_focus_iterations = 0, then the Focus feature of this method is disabled
    quantityList = quantityList + focusList

    # Generate all specified models
    modelList = _modelMultiProc(quantityList, count_d, yVals, max_enet_iterations)
    
    # NOTE for future development of this method:
    # If ANYTHING is done with modelList--access, mutation, etc.--the code for it
    # MUST be put inside a 'if __name__ == "main"' bracket.
    # Non-descriptive and misleading exceptions will be thrown otherwise.
    if __name__ == "__main__":
        tbrDict = _getOptimalModelCandidates(modelList, yVals)
        if appendFullModelList:
            modelList.sort(key=lambda x: x.AUC, reverse=True)
            tbrDict['modelList'] = modelList
        return tbrDict    

## Takes a list of Model objects and places them in a pandas.DataFrame, sorted by AUC, for easy comparison.
#
# @arg modelList a list of Models to create the DataFrame from
#
# @return a pandas.DataFrame object containing pertinent statistics for model comparison, 
#           sorted by AUC of the models in descending order.
#         Each row of the DataFrame represents a Model from the argument modelList.
#         Note that the index of the data frame rows correspond to the index of the model in the argument modelList.
def getCompDF(modelList: List[Model]):
    # Create a pandas.DataFrame object to hold the AUC, recall, and accuracy of each model
    AUCList = []
    accList = []
    recList = []
    numTopicsList = []
    for model in modelList:
        AUCList.append(model.AUC)
        accList.append(model.accuracy)
        recList.append(model.recall)
        numTopicsList.append(model.topicQuantity)
    tempDict = {
        "AUC" : AUCList,
        "Accuracy" : accList,
        "Recall" : recList,
        "Topics" : numTopicsList
    }
    modelDF =  pd.DataFrame(data = tempDict)
    # Sort the models within the data frame by AUC in descending order.
    # Note that the "index" column will represent the index of each model in the 
    #   original List provided as an argument to this function.
    modelDF.sort_values(by = 'AUC', ascending = False, inplace = True)
    return modelDF
    

## Creates a Model object for each topic quantity specified in the argument topicQuantityVector.
# Python's implementation of multithreading, mutiprocessing, is used in this method to vastly decrease runtime.
# Multiprocessing is used instead of Python's "multithreading" since this method is entirely CPU bound and involves no file/internet I/O.
#
# @arg topicQuantityVector a vector of the topic quantities to test, e.g. [30,40,50,60,70,80].
#                           Note that the values in this vector must be integers.
#                           Note that if only a single quantity is desired, it must be passed as
#                               a list with one element, e.g. topicQuantityVector = [10].
# @arg count_data the input to LDA.fit(...)
# @arg responseValues the response, Y, for the input data; a vector;
#                       this is 'values' from the DataFrame 'processedCorpusDataFrame' which is in the Dictionary returned
#                       from the prepareTestTrainData() method in TextPrep.py
# @arg max_iterations the maximum number of iterations for the Stochastic Average Gradient Descent ('saga') solver 
#                       to go through for the elastic net CV.  Default is 4900.  Lower numbers drastically decrease 
#                       runtime for large datasets, but may create inferior models.
#
# @return a list of Model objects, one for each fitted model based on the topic quantities provided in the argument topicQuantityVector.
def _modelMultiProc(
        topicQuantityVector: List[int], count_data, responseValues: List[bool], 
        max_iterations: int = 4900) -> List[Model]:
    # Check for the number of simultaneous threads that can be supported by current hardware and OS
    numProcesses = multiprocessing.cpu_count()
    # Don't allocate extra threads if there are less quantities (models to create) than available threads
    numModelsToCreate = len(topicQuantityVector)
    if(numProcesses > numModelsToCreate):
        numProcesses = numModelsToCreate
    # Break the list of topic quantities into numProcesses parts
    topicQuantityChunks = np.array_split(topicQuantityVector, numProcesses)
    # Note that topicQuantityChunks is a List of Lists of topic quantities to be used.
    # topicQuantityChunks[i] is a list of the topic quantities to be used by thread i, for i an element of [0,numProcesses).
    
    tbrList = []
    if __name__ == "__main__":
        multiprocessing.freeze_support()
        # Create a Queue to indicate when the progress bar should be updated
        # This queue will also hold the generated Model objects
        pbarQueue = multiprocessing.Queue()
        # Print a header for the thread initialization printouts
        print("\nInitializing Threads: ")
        # Create an array to hold the threads
        threads = []
        # Creare a manual tqdm progress bar for thread initialization
        thread_pbar = tqdm.tqdm(total = numProcesses, position = 0, leave = True)
        # Generate threads
        for i in range(numProcesses):
            thread = multiprocessing.Process(target=_modelWithNTopics, args=(topicQuantityChunks[i], count_data, responseValues, max_iterations, pbarQueue, True))
            threads.append(thread)
            if __name__ == "__main__":
                thread.start()
                # Update the thread initialization progress bar
                thread_pbar.update(1)
        # Close the thread initialization progress bar
        thread_pbar.close()
        # Print a header for the model generation progress bar
        print("Generating Models: ")
        # Create a manual tqdm progress bar for model generation, total = num_models_to_create
        pbar = tqdm.tqdm(total= len(topicQuantityVector))
        # Listen for "created a model" events
        finishedModels = 0
        tbrList = []
        # Add Model objects to the return List as they are created.
        # Update the model generation progress bar for each model added to the return list.
        while(finishedModels < numModelsToCreate):
            modelToAdd = pbarQueue.get()
            tbrList.append(modelToAdd)
            pbar.update(1)
            finishedModels = finishedModels + 1
        # Allow all threads to complete
        for thread in threads:
            if __name__ == "__main__":
                multiprocessing.freeze_support()
                thread.join()
        # Close the model generation progress bar
        pbar.close()
        # Return a List of Model objects generated by the threads
        return tbrList
    else:
        print("__name__ != '__main__'")    

## Creates a Model object for each topic quantity specified in the argument topicQuantityVector.
#
# @arg topicQuantityVector a vector of the topic quantities to test, e.g. [30,40,50,60,70,80].
#                           Note that the values in this vector must be integers.
#                           Note that if only a single quantity is desired, it must be passed as
#                               a list with one element, e.g. topicQuantityVector = [10].
# @arg count_data the input to LDA.fit(...)
# @arg responseValues the response, Y, for the input data; a vector;
#                       this is 'values' from the DataFrame 'processedCorpusDataFrame' which is in the Dictionary returned
#                       from the prepareTestTrainData() method in TextPrep.py
# @arg max_iterations the maximum number of iterations for the Stochastic Average Gradient Descent ('saga') solver 
#                       to go through for the elastic net CV.  Default is 4900.  Lower numbers drastically decrease 
#                       runtime for large datasets, but may create inferior models.
# @arg pbarQueue the multiprocessing.Queue object used in multiprocessing implementations to handle progress bar progress.
#           Do not provide this argument manually.
# @arg isMultiProc true if called by the multiprocessing implementation; false otherwise.
#           Note that the default value is False, and this should not be changed.
#
# @return a list of Model objects, one for each fitted model based on the topic quantities provided in the argument topicQuantityVector.
def _modelWithNTopics(
        topicQuantityVector: List[int], count_data, responseValues: List[bool], max_iterations: int = 4900, 
        pbarQueue: multiprocessing.Queue= None, isMultiProc: bool = False) -> List[Model]:
    
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
        tempModel = Model(number_topics, fit, lda, cm, acc, rec, auc)

        # Add the Model object to the list
        fitList.append(tempModel)

        # If this method was called as part of the multiprocessing implementation, 
        # then update the progress bar and insert the model into the queue
        if(isMultiProc):
            pbarQueue.put(tempModel)
        
    return fitList

## Evaluates a list of Model objects and returns a Dictionary of the five "best" models.
#
# The "best" model depends on the original corpus and purpose of the model, so it is left
# to the user to determine which of those five best suits their purposes.
#
# If the user is unsure of which model to use, defaulting to using the one with the
# highest AUC--stored as "AUC" in the return Dictionary--would be best.
# 
# @arg modelList a list of Model objects to evaluate
# @arg yVals a list of 1/0 (True/False) values from the original test/train data set.
#               This is used to compute the standard error of the maximum AUC value.
#
# @return A Dictionary containing five candidates for an optimal model, stored as Model objects:
#           1) "AUC", the model with the highest AUC
#           2) "acc", the model with the highest accuracy
#           3) "rec", the model with the highest recall
#           4) "acc_1se", the model with the highest accuracy among models whose AUC is within 1 standard error of the maximum
#           5) "rec_1se", the model with the highest recall among models whose AUC is within 1 standard error of the maximum
def _getOptimalModelCandidates(modelList: List[Model], yVals: List[bool]):
    # Generate the dictionary of the five optimal model candidates
    # Note that since this is called once when generating large lists of models,
    #   minimizing memory footprint is prioritized over minimizing runtime
    # First, sort by accuracy
    modelList.sort(key=lambda x: x.accuracy, reverse=True)
    acc = modelList[0]
    # Sort by recall
    modelList.sort(key=lambda x: x.recall, reverse=True)
    rec = modelList[0]
    # Sort by AUC
    modelList.sort(key=lambda x: x.AUC, reverse=True)
    AUC = modelList[0]
    # Truncate the list to contain only models with AUC within 1 standard error of the maximum
    seAUC = _se_auc(AUC.AUC, yVals.count(1), yVals.count(0))
    auc_1se = AUC.AUC - seAUC
    modelList = list(filter(lambda x: x.AUC >= auc_1se, modelList))
    # Sort the truncated list by accuracy to find acc_1se
    modelList.sort(key=lambda x: x.accuracy, reverse=True)
    acc_1se = modelList[0]
    # Sort the truncated list by recall to find rec_1se
    modelList.sort(key=lambda x: x.recall, reverse=True)
    rec_1se = modelList[0]
    # Create the dictionary to be returned
    tempDict = {
        "AUC" : AUC,
        "acc" : acc,
        "rec" : rec,
        "acc_1se" : acc_1se,
        "rec_1se" : rec_1se
    }
    
    return tempDict

# Compute standard error of AUC score, using its equivalence to the Wilcoxon statistic.
# Shamelessly adapated from the method of the same name in the R package auctestr.
# The original implementation in R may be found at https://rdrr.io/cran/auctestr/src/R/auc_compare.R 
#
# @references Hanley and McNeil, The meaning and use of the area under a receiver
# operating characteristic (ROC) curve. Radiology (1982) 43 (1) pp. 29-36.
# @references Fogarty, Baker and Hudson, Case Studies in the use of ROC Curve Analysis
# for Sensor-Based Estimates in Human Computer Interaction,
# Proceedings of Graphics Interface (2005) pp. 129-136.
#
# @arg auc value of A' statistic (or AUC, or Area Under the Receiver operating
#          characteristic curve) (numeric).
# @arg n_p number of positive cases (integer).
# @arg n_n number of negative cases (integer).
# 
# @return the standard error of the AUC score passed as the argument "auc"
#
# @examples
# se_auc(0.75, 20, 200)
# ## standard error decreases when data become more balanced over
# ## positive/negative outcome class, holding sample size fixed
# se_auc(0.75, 110, 110)
# ## standard error increases when sample size shrinks
# se_auc(0.75, 20, 20)
def _se_auc(auc, n_p, n_n) -> float:
    D_p = (n_p - 1) * ((auc/(2 - auc)) - (auc**2))
    D_n = (n_n - 1) * ((2 * (auc**2))/(1 + auc) - (auc**2))
    SE_auc = ((auc * (1 - auc) + D_p + D_n)/(n_p * n_n))**0.5
    return(SE_auc)

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
# preppedData = prepareTestTrainData('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', isRedditData=True)
# count_d = preppedData['count_data']
# count_v = preppedData['count_vectorizer']
# yVals = preppedData['processedCorpusDataFrame']["value"].tolist()
# modelList = modelMultiProc([10,15,20,25], count_d, count_v, yVals)

##### getModels() method #####
#modelDict = getModels('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', start = 10, stop = 10)
#modelDict = getModels('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', start = 5, stop = 15, step = 1)
#modelDict = getModels('SuicideWatchRedditJSON.json', 'AllRedditJSON.json')
#modelDict = getModels('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', focus_iterations = 0, appendFullModelList = True, stop = 20)
#modelDict = getModels('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', appendFullModelList = True, focus_iterations = 0, start = 5, stop = 30, range_iterations = 40, step = 1)
#modelDict = getModels('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', appendFullModelList = True, focus_iterations = 0, start = 14, stop = 14, range_iterations = 500, step = 1)
#if __name__ == "__main__":
#    mod_df = getCompDF(modelDict["modelList"])