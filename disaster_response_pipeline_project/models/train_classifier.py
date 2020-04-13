import sys


def load_data(database_filepath):
    """Load and merge datasets

    Args:
    database_filename: filename for SQLite database

    Returns:
    X: features dataset
    Y: labels dataset
    """

    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('table', engine)

    # Define feature and target variables X and Y
    X = df['message']
    Y = df.iloc[:,4:]

    # Create list containing all category names
    category_names = list(Y.columns.values)

    return X, Y, category_names


def tokenize(text):
    """Tokenization function to process text data
    Normalize, tokenize, stem, lemmatize text string

    Args:
    text: string

    Returns:
    lemmed: ist containing word tokens
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

    Returns: cv, a gridsearchcv object
    """


def evaluate_model(model, X_test, Y_test, category_names):

    # predict on test data
    Y_pred = model.predict(X_test)

    # classification_report
    for column in category_names:
        print('\n---- {} ----\n{}\n'.format(column, classification_report(Y_test[column], Y_pred[column])))

    # evaluate
    results = []
    labels = np.unique(Y_pred)
    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values)
       # accuracy_score(y_test.iloc[:,1:].values, np.array([x[1:] for x in y_pred]
        precision = precision_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values, average='weighted')
        recall = recall_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values, average='weighted')
        f1 = f1_score(Y_test.iloc[:, i].values, Y_pred.iloc[:, i].values, average='weighted')

        results.append([accuracy, precision, recall, f1])
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("f1 score:", f1)

    return

def save_model(model, model_filepath):
    """Save  model

    Args:
    model: fitted model
    model_filepath: filepath for where model should be saved

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
