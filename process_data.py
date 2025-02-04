import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets.

    Args:
        messages_filepath (str): Filepath for the messages dataset.
        categories_filepath (str): Filepath for the categories dataset.

    Returns:
        pd.DataFrame: Merged dataframe of messages and categories.
    """
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on the 'id' column
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Clean the merged dataframe by splitting categories, converting values to binary, 
    and removing duplicates.

    Args:
        df (pd.DataFrame): Merged dataframe of messages and categories.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    # Split the 'categories' column into separate columns by ';'
    categories = df['categories'].str.split(';', expand=True)

    # Use the first row of categories to create column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])  # Get category name before the '-'
    categories.columns = category_colnames

    # Convert category values to just numbers (0 or 1)
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])  # Get binary values
        categories[column] = pd.to_numeric(categories[column])  # Convert to numeric (0 or 1)

    # Drop the original categories column from df
    df = df.drop(columns=['categories'])

    # Concatenate the new category columns to the original dataframe
    df = pd.concat([df, categories], axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the cleaned data to an SQLite database.

    Args:
        df (pd.DataFrame): Cleaned dataframe.
        database_filename (str): Path to the SQLite database file.
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    
    # Save data to SQLite (replace existing table if it exists)
    df.to_sql('DisasterResponse_table', engine, index=False, if_exists='replace')  # Replace table if it exists


def main():
    """
    Main function that coordinates the loading, cleaning, and saving of data.
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()