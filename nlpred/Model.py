import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pickle
import nlpred

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
        # 8) "AUC", a general measure of fit for the model as per the chosen predictive threshold for "true" vs "false" from logreg predictive output
        self.AUC = AUC
    
    # Predicts True/False (1/0) for each document (String) in the List listOfDocStrings argument using the model stored in this Model object.
    # @arg listOfDocStrings a List of Strings wherein each String contains the text of a document to predict True/False from
    # @return a DataFrame containing two columns:
    #           1)  'paper_text', the original input textual data
    #           2)  'paper_text_processed', the processed (tokenized with punctuation and capitalization removed) textual data
    #           3)  'value', the 1/0 (True/False) prediction for the 'paper_text' entry in the same row
    def predict(self, listOfDocStrings):
        pass
