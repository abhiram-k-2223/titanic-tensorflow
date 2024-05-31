# Titanic Survival Prediction Using TensorFlow

This project demonstrates how to build and evaluate a machine learning model using TensorFlow 2.x to predict the survival of passengers on the Titanic. The dataset used is a public dataset provided by TensorFlow, which includes various features about the passengers.

### Dataset

The dataset consists of two CSV files: 
- `train.csv`: This file is used for training the model.
- `eval.csv`: This file is used for evaluating the model.

The target variable is `survived`, which indicates whether a passenger survived (`1`) or not (`0`). The features include personal details such as age, gender, class, number of siblings/spouses aboard, number of parents/children aboard, fare, embarkation town, and whether the passenger was alone.

### Data Preprocessing

The preprocessing steps involve:
1. Loading the data from the CSV files.
2. Separating the target variable (`survived`) from the features.
3. Exploratory data analysis, including:
   - Displaying the first few rows of the target variable.
   - Describing the statistics of the training dataset.
   - Plotting histograms of numerical features and bar charts of categorical features to understand their distributions and relationships with survival rates.

### Feature Columns

The features are divided into categorical and numerical columns:
- **Categorical Columns**: `sex`, `n_siblings_spouses`, `parch`, `class`, `deck`, `embark_town`, and `alone`.
- **Numerical Columns**: `age` and `fare`.

TensorFlow's feature columns are used to transform these raw features into a format suitable for the model:
- Categorical features are represented using `categorical_column_with_vocabulary_list`.
- Numerical features are represented using `numeric_column`.

### Input Functions

Two input functions are created for training and evaluation:
- `train_input_fn`: Used to load and preprocess the training data. It shuffles the data, batches it, and repeats it for a specified number of epochs.
- `eval_input_fn`: Used to load and preprocess the evaluation data. It does not shuffle the data and is run for only one epoch.

### Model Training and Evaluation

A `LinearClassifier` from TensorFlow's Estimator API is used to train the model:
1. The model is first trained using the base feature columns.
2. The accuracy and other evaluation metrics are calculated using the evaluation dataset.

### Feature Engineering

To improve the model's performance, feature engineering is applied:
- A crossed feature column `age_x_gender` is created by crossing `age` and `sex` features. This allows the model to learn interactions between age and gender.

The model is then retrained with the additional derived feature and evaluated again to see if the performance improves.

### Results

The accuracy and other evaluation metrics of the model are displayed after both the initial training and after adding the derived feature. These metrics help in understanding how well the model is performing and whether the feature engineering improved the predictions.
