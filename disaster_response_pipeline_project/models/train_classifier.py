'''
This ML pipeline is doing the following steps:
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
'''

# import libraries
import sys
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
import random

from sklearn.preprocessing import FunctionTransformer
from sklearn.neighbors import KNeighborsClassifier

import pickle

#%matplotlib inline
nltk.download(['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger'])


def load_data(database_filepath):
    """Load and merge datasets

    Args: database_filename -> filename for SQLite database

    Returns: X -> features dataset
             Y -> labels dataset
    """

    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("SELECT * FROM table", engine)

    # Define feature and target variables X and Y
    X = df['message']
    Y = df.iloc[:,4:]

    # Create list containing all category names
    category_names = list(Y.columns.values)

    return X, Y, category_names


def tokenize(text):
    """Tokenization function to process text data
    Normalize, tokenize, stem, lemmatize text string

    Args: text -> string

    Returns: lemmed -> list containing word tokens
"""

    # normalize
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = text.split()

    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]

    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]

    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]

    return lemmed


def build_model():
    """Build a machine learning pipeline

    Args: None

    Returns: cv -> a gridsearchcv object
    """
    # Create Pipeline
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                      ('tfidf', TfidfTransformer()),
                      ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # Create Parameter list
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
#        'vect__max_df': (0.5, 0.75, 1.0),
#        'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': [True, False],
#        'clf__estimator__n_estimators': [50, 100, 200],
        'clf__estimator__min_samples_split': [2, 3],
    }

    # Grid Search Algorithm
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    # predict on test data
    Y_pred = model.predict(X_test)

    # classification_report
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], Y_pred[:, i]))

    #for column in category_names:
    #    print('\n---- {} ----\n{}\n'.format(column, classification_report(Y_test[column], Y_pred[column])))

    # evaluate
    #results = []
    #labels = np.unique(Y_pred)
    #for i in range(len(category_names)):
    #    accuracy = accuracy_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values)
       # accuracy_score(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]
    #    precision = precision_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values, average='weighted')
    #    recall = recall_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values, average='weighted')
    #    f1 = f1_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values, average='weighted')

   #     results.append([accuracy, precision, recall, f1])
   #     print("Accuracy:", accuracy)
   #     print("Precision:", precision)
    #    print("f1 score:", f1)


def save_model(model, model_filepath):
    """Save  model

    Args: model -> fitted model
    model_filepath -> filepath for where model should be saved

    Returns: None
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
