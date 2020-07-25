# nlpred
Provides support for making predictions of binary outcomes based on natural language corpora.  Formally, creates predictive models using Logistic Regression with Elastic Net Regularization on topic models derived from Latent Dirichlet Allocation.  Supports multithreading for increased runtime efficiency on large data sets.

A black box method, getModels(), is provided so that users who do not wish to manually tweak their models can create high-quality models by simply providing two vectors of documents (Strings): one vector of documents associated with true responses, and one vector of documents associated with false responses.

# Quick Start Example

```python
# Note that all of the below methods are contained in Model.py of nlpred; appropriate imports are required.

# Declare a vector of Strings containing documents in the test/train set associated with TRUE reponses
trueDocs = ["true document 1 content", "true document 2 content", "true document 3 content"]

# Declare a vector of Strings containing documents in the test/train set associated with FALSE reponses
falseDocs = ["false document 1 content", "false document 2 content", "false document 3 content"]

# Call getModels() from Model.py to generate predictive models for the documents
modelDict = getModels(trueDocs, falseDocs, isRedditData = False)

# modelDict is now a Dictionary containing five candidates for an optimal model, stored as Model objects:
#   1) "AUC", the model with the highest AUC
#   2) "acc", the model with the highest accuracy
#   3) "rec", the model with the highest recall
#   4) "acc_1se", the model with the highest accuracy among models whose AUC is within 1 standard error of the maximum
#   5) "rec_1se", the model with the highest recall among models whose AUC is within 1 standard error of the maximum
```
The "best" model depends on the original corpus and purpose of the model, so it is left to the user to determine which of the five models returned by this method best suits their purposes.

If the user is unsure of which model to use, it would behoove them to simply choose the model with the highest AUC, stored as "AUC" in the return Dictionary.

# Model Object Functionality

To illustrate what can be done with a Model object, assume the model with the highest AUC was chosen as optimal:
```python
optimalModel = modelDict['AUC']
```
The model can be saved to disk as a .pkl file in the current work directory:
```python
optimalModel.save(fileName = "optimal_model.pkl")
```
Models may be manually loaded into the python environment from a .pkl file:
```python
loadedModel = loadModel(fileName = "optimal_model.pkl")
```
The model can be used to make predictions from new documents, passed as a vector of Strings:
```python
newDocs = ["new document 1 content", "new document 2 content", "new document 3 content"]

predictionDataFrame = optimalModel.predict(listOfDocStrings = newDocs)

# predictionDataFrame is now  a DataFrame containing three columns:
#   1)  'paper_text', the original input textual data
#   2)  'paper_text_processed', the processed (tokenized with punctuation and capitalization removed) textual data
#   3)  'value', the 1/0 (True/False) prediction for the 'paper_text' entry in the same row
```
Alternatively, predictions can be made directly from a .pkl file without having to load the model manually:
```python
predictionDataFrame = predictFromFile(fileName = "optimal_model.pkl", listOfDocStrings = newDocs)
```
A summary of the model may be printed to stdout using the python standard print() function:
```python
print(optimalModel)
```
Alternatively, a summary of the model may be printed to stdout with the instance method print():
```python
optimalModel.print()
```
Instance variables of the Model object may be obtained by the standard extensions, e.g.:
```python
optimalAUC = optimalModel.AUC
```
The instance variables accessible to the user are as follows:

| Variable  | Description |
| ------------- | ------------- |
|AUC| A general measure of fit for the model as per the chosen predictive threshold for "true" vs "false" from logreg predictive output.|
|accuracy| The classification rate, i.e. the percent of correct predictions; or, the number of correct predictions made divided by the total number of predictions made.|
|recall| a.k.a sensitivity; P(Y=1 \| X=1), the proportion of true samples correctly identified as true.  I.e., the number of true samples correctly identified divided by the total number of true samples.  The recall is intuitively the ability of the classifier to find all the positive samples.|
|confusion_matrix| The confusion matrix of the fitted model.
|topicQuantity| The number of topics used in .this model.|
|fit| The fitted logistic regression model itself, of type sklearn.linear_model.LogisticRegressionCV|
|topicLDAModel| The LDA object for the fitted topic model.|
|id| A unique identifier for the model.  Note that this is a UUID object.  The actual id may be found in “id.hex”.  The UUID wrapper object is kept for aesthetic reasons with regard to printing, etc.|

# Full Documentation
For full documentation, please see [documentation.pdf](documentation.pdf) in the repository.
