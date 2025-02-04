# Udacity-Disaster-Response-Pipline-Project
Udacity Data Pipelin Project using Jupyter Notebooks and Bash terminal
### Project Summary

This project is designed to build an end-to-end data pipeline using Python, with a focus on ETL (Extract, Transform, Load) and machine learning for disaster response message classification. The project involves reading and cleaning disaster message data, training a machine learning model, and deploying the model through a web application built with Flask. The goal is to predict categories of disaster messages based on their content.

### How to Run the Python Scripts and Web App

1. **Running the ETL Pipeline:**
   The first step is to process the data and store it in a SQLite database. This is done by running the `process_data.py` script, which takes in the disaster message and category CSV files as inputs and outputs a database file. Here’s how you run the ETL pipeline:

   ```bash
   python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
   ```

   - `disaster_messages.csv`: The file containing the disaster message data.
   - `disaster_categories.csv`: The file containing the corresponding categories for the messages.
   - `DisasterResponse.db`: The SQLite database where the cleaned data will be saved.

2. **Training the Machine Learning Model:**
   After the data is loaded into the database, the next step is to train the machine learning model. This is done by running the `train_classifier.py` script. It loads the processed data from the SQLite database and trains a model using NLTK and scikit-learn’s Pipeline and GridSearchCV. Here’s how you run the script to train the classifier:

   ```bash
   python train_classifier.py ../data/DisasterResponse.db classifier.pkl
   ```

   - `DisasterResponse.db`: The SQLite database containing the clean data.
   - `classifier.pkl`: The file where the trained machine learning model will be saved.

3. **Running the Flask Web App:**
   After training the model, the final step is to run the Flask web app that will display the model’s predictions and data visualizations. Use the following command to run the web app:

   ```bash
   python run.py
   ```

   This will start a Flask web server, and you can access the web app by clicking the "PREVIEW" button in the Project Workspace IDE.
![IDE Workspace Terminal](https://github.com/user-attachments/assets/59c13dcb-a1ad-4aae-a76e-d7ffb960f90f)

4. **Website Preview:**
   Once the web app is running, you can open the homepage to view the basic structure. The web app is capable of showing results for the classification of disaster messages. You will also see the visualizations generated from the SQLite database.
![project ](https://github.com/user-attachments/assets/03b845e5-6628-496f-9973-8d8cb7695fc9)

### Explanation of the Files in the Repository

The file structure is organized as follows:

- **app/**  
  Contains the Flask app files, including templates for the web pages and the `run.py` file to run the app.
  - **templates/**
    - `master.html`: The main page of the web app.
    - `go.html`: The page that shows classification results.
  - `run.py`: The Flask script to run the app.
- ![image](https://github.com/user-attachments/assets/5bed21bf-77af-42c6-9634-74cbc36e8590)
- ![image](https://github.com/user-attachments/assets/63b5aa5b-6138-429e-abc3-110af4259e03)
- ![image](https://github.com/user-attachments/assets/cc7fe8b4-3587-48e1-8775-f17c3bff9c8c)


- **data/**  
  Contains data files and the ETL pipeline script to process the data.
  - `disaster_categories.csv`: Contains categories of messages.
  - `disaster_messages.csv`: Contains the disaster message data.
  - `process_data.py`: The script to process data and load it into the SQLite database.
  - `DisasterResponse.db`: The SQLite database where cleaned data is saved.

- **models/**  
  Contains the machine learning pipeline code and the trained classifier model.
  - `train_classifier.py`: The script to train the machine learning model using scikit-learn.
  - `classifier.pkl`: The saved classifier model.

- **README.md**: The readme file that includes information about the project, setup, and instructions.

### Code Quality and Documentation

- **Effective Comments:** The code is well-commented, and each function includes a docstring explaining its purpose and how it works. This improves the clarity and maintainability of the code.
- **Modular and Well-Documented:** Each part of the pipeline (ETL and machine learning) is written in separate scripts, making it easy to maintain and scale. Additionally, each function is modular and can be reused independently if needed.
- **Tested and Functional Code:** All the provided code works as intended:
  - The ETL pipeline successfully loads the data into the SQLite database.
  - The machine learning model is trained and saved as a pickle file.
  - The Flask web app displays the results and visualizations.

### Final Notes

- Make sure to follow the instructions carefully when running the Python scripts. The paths to the CSV files and database must be correctly specified.
- The Flask web app allows for creative customization, and you can modify the templates and visualizations as needed. The starter files already include a basic structure for the web app, so you just need to integrate the database and model files.
- 
