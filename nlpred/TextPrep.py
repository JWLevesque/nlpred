import json
import pandas as pd
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.decomposition import LatentDirichletAllocation as LDA

# Prepares textual input for model creation.  Supports both reddit JSON input and vectors of Strings.
#
# @arg trueSource Either:
#                   1) if redditData = true, the filename of the JSON file containing "true" Strings, including the .json extension
#                   2) if redditData = false, an array of Strings which are known to be associated with a "true"/"1" prediction
# @arg falseSource Either:
#                   1) if redditData = true, the filename of the JSON file containing "false" Strings, including the .json extension
#                   2) if redditData = false, an array of Strings which are known to be associated with a "false"/"0" prediction
# @arg redditData Boolean:  
#                   1) True indicates the data is given as JSON objects, and _getCorpusFromReddit will be called.
#                   2) False indicates the data is given as vectors of Strings, and _getCorpusFromVectors will be called.
#                 Note that redditData is TRUE by default.
# @return A Dictionary with two elements: 
#         1)  "processedCorpusDataFrame", a data frame containing three columns:
#               I)   'paper_text', the original input textual data
#               II)  'value', the corresponding 1/0 (i.e., true/false) value associated with the 'paper_text' entry in the same row
#               III) 'paper_text_processed', the processed (tokenized with punctuation and capitalization removed) textual data
#         2)  "count_data", the token counts from the processed textual data to be used in model creation
def prepareTestTrainData(trueSource, falseSource, isRedditData = True):
    processedCorpusDataFrame = _getCorpusDataFrame(trueSource, falseSource, redditData=isRedditData)
    count_data = _processDataFrame(processedCorpusDataFrame)
    tempDict = {
        "processedCorpusDataFrame" : processedCorpusDataFrame,
        "count_data" : count_data
    }
    return tempDict

# Prepares a list of Strings for prediction using a previously created model.
#
# @arg docsToPredict a list of Strings containing the documents to have predictions made from
# @return A Dictionary with two elements: 
#         1)  "processedCorpusDataFrame", a data frame containing two columns:
#               I)   'paper_text', the original input textual data
#               II)  'paper_text_processed', the processed (tokenized with punctuation and capitalization removed) textual data
#         2)  "count_data", the token counts from the processed textual data to be used for prediction
def preparePredData(docsToPredict):
    processedCorpusDataFrame = _getPredDataFrame(stringList=docsToPredict)
    count_data = _processDataFrame(processedCorpusDataFrame)
    tempDict = {
        "processedCorpusDataFrame" : processedCorpusDataFrame,
        "count_data" : count_data
    }
    return tempDict

# Helper method for Reddit data.
# Extracts the 'selftext' corpus from a reddit JSON and deletes empty/removed submissions.
# @arg jsonFileString The filename of the JSON file, including the .json extension
# @return a list containing raw Strings of the textual reddit submissions.
def _getSubmissionStringArray(jsonFileString):
    
    # open the SuicideWatch JSON
    with open(jsonFileString, 'r') as myFile:
        sData = myFile.read()
    
    # parse the SuicideWatch JSON
    sList = json.loads(sData)

    # extract the text content of the submissions
    tbr = []
    for i in range(0,len(sList['data'])):
        if('selftext' in sList['data'][i]):
            if(sList['data'][i]['selftext'] != "" and sList['data'][i]['selftext'] != "[removed]" and sList['data'][i]['selftext'] != "[deleted]"):
                tbr.append(sList['data'][i]['selftext'])
    return tbr

# Helper method which creates a DataFrame from two vectors of Strings--one for entries designated 'true', and one for 'false'.
# @arg trueList An array of Strings which are known to be associated with a "true"/"1" prediction
# @arg falseList An array of Strings which are known to be associated with a "false"/"0" prediction
# @return A DataFrame with two columns: 
#         1)  'paper_text', the textual data
#         2)  'value', the corresponding 1/0 (i.e., true/false) value associated with the 'paper_text' entry in the same row
def _getCorpusFromVectors(trueList, falseList):
    # Generate vectors of 0s and 1s of appropriate size for labeling the input vectors
    trueLabels = [1]*len(trueList)
    falseLabels = [0]*len(falseList)

    # Concatenate the lists
    fullList = trueList + falseList
    fullLabels = trueLabels + falseLabels

    # Create a DataFrame object for the corpus
    papers = pd.DataFrame(list(zip(fullList,fullLabels)), columns = ['paper_text', 'value'])

    return papers

# Helper method for Reddit data.
# Creates a DataFrame from two reddit JSON objects, with empty/removed selftext entries deleted.
# @arg trueJsonFileString The filename of the JSON file containing "true" Strings, including the .json extension
# @arg falseJsonFileString The filename of the JSON file containing "false" Strings, including the .json extension
# @return A DataFrame with two columns: 
#         1)  'paper_text', the textual data
#         2)  'value', the corresponding 1/0 (i.e., true/false) value associated with the 'paper_text' entry in the same row
def _getCorpusFromReddit(trueJsonFileString, falseJsonFileString):
    tbr = _getCorpusFromVectors(_getSubmissionStringArray(trueJsonFileString), _getSubmissionStringArray(falseJsonFileString))
    return tbr

# Creates a Data Frame object to be processed.  Calls either _getCorpusFromReddit or _getCorpusFromVectors.
# @arg trueSource Either:
#                   1) if redditData = true, the filename of the JSON file containing "true" Strings, including the .json extension
#                   2) if redditData = false, an array of Strings which are known to be associated with a "true"/"1" prediction
# @arg falseSource Either:
#                   1) if redditData = true, the filename of the JSON file containing "false" Strings, including the .json extension
#                   2) if redditData = false, an array of Strings which are known to be associated with a "false"/"0" prediction
# @arg redditData Boolean:  
#                   1) True indicates the data is given as JSON objects, and _getCorpusFromReddit will be called.
#                   2) False indicates the data is given as vectors of Strings, and _getCorpusFromVectors will be called.
#                 Note that redditData is TRUE by default.
# @return A DataFrame with two columns: 
#         1)  'paper_text', the textual data
#         2)  'value', the corresponding 1/0 (i.e., true/false) value associated with the 'paper_text' entry in the same row
def _getCorpusDataFrame(trueSource, falseSource, redditData = True):
    if(redditData):
        return _getCorpusFromReddit(trueSource, falseSource)
    else:
        return _getCorpusFromVectors(trueSource, falseSource)

# Helper method for prediction.
# Creates a DataFrame containing a single column named 'paper_text' with the documents to be used in prediction.
# @arg stringList a List of documents in String form to be used in prediction
def _getPredDataFrame(stringList):
    return pd.DataFrame(data = stringList, columns=['paper_text'])


# Tokenizes the data for LDA by removing punctuation and capitalization, and then provides token counts.
# @arg papers a DataFrame containing two columns:
#      1) 'paper_text', the textual data
#      2) 'value', OPTIONAL, the corresponding 1/0 (i.e., true/false) value associated with the 'paper_text' entry in the same row
#      This is returned from either the _getCorpusDataFrame method or the _getPredDataFrame method.
# @postState A new column named 'paper_text_processed' will be added to the argument DataFrame.  This new column will be
#               a copy of the original 'paper text' column, except punctuation and capitalization will be removed.
# @return the token counts from the input data derived via sklearn.feature_extraction.text.CountVectorizer
def _processDataFrame(papers):
    
    # Load the regular expression library
    import re

    # Remove punctuation
    papers['paper_text_processed'] = papers['paper_text'].map(lambda x: re.sub('[,\.!?]', '', x))

    # Convert the titles to lowercase
    papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())

    from sklearn.feature_extraction.text import CountVectorizer

    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')

    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(papers['paper_text_processed'])

    return count_data


############# Test stuff
#testData = _getCorpusDataFrame('SuicideWatchRedditJSON.json', 'AllRedditJSON.json', redditData=True)

#count_data = _processDataFrame(testData)