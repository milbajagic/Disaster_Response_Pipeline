# Disaster Response Pipeline Project
This project was completed as part of the course requirements of Udacity's [Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) certification.

### Data
The project used a data set from [Figure Eight](https://www.figure-eight.com/) that contained labeled disaster messages received by an aid organization. 


### Main Project Components:
1. process_data.py, ETL Pipeline that:
Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

2.  train_classifier.py, a machine learning pipeline that:
Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

3. Flask Web App


### Libraries used
* This project requires **Python 3.x** and the following libraries installed:
pandas, sklearn, sqlite3, sqlalchemy, nltk, plotly, flask

### Included files
- app
| - templates
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- Database.db # database to save clean data to
|- process_data.py
|- ETL Pipeline Preparation.ipynb

- models
|- train_classifier.py
|- ML Pipeline Preparation.ipynb

- workspace_utils.py
- README.md

### How to run:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
