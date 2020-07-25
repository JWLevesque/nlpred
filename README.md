# nlpred
Provides support for making predictions of binary outcomes based on natural language corpora.  Formally, creates predictive models using Logistic Regression with Elastic Net Regularization on topic models derived from Latent Dirichlet Allocation.  Supports multithreading for increased runtime efficiency on large data sets.

A black box method, getModels(), is provided so that users who do not wish to manually tweak their models can create high-quality models by simply providing a vector of documents (Strings) and a vector of binary responses associated with the documents.
