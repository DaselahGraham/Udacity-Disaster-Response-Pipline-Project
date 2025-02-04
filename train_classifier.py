import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Tokenization function
def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    lemmed = [lemmatizer.lemmatize(w) for w in words]
    words = [lemmatizer.lemmatize(w, pos='v') for w in lemmed]
    return words

# Load data from the database
def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DisasterResponse_table', engine)
    
    # Define feature and target variables
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    category_names = Y.columns.tolist()
    
    return X, Y, category_names

# Build the ML pipeline
def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(DecisionTreeClassifier()))  # Multi-output classifier using DecisionTree
    ])
    return pipeline

# Evaluate the model
def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(f"Classification report for {column}:")
        print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]), "\n")

# Save the model as a pickle file
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

# Main function
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        # Split the data into train and test sets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
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