# DIABETICS_PREDICTION
Introduction of the Project - Predictive Analysis in Diabetes

comments
Diabetes is a prevalent chronic disease characterized by high blood sugar levels, impacting millions of lives worldwide. Effective management and early detection of diabetes are crucial for preventing complications and improving patient outcomes. In this project, we delve into the realm of predictive analytics to forecast diabetes occurrence using classification techniques, particularly focusing on logistic regression.

Classification is a fundamental task in machine learning that involves categorizing data into predefined classes or categories based on input features. Logistic regression, a widely used classification algorithm, is particularly well-suited for binary classification problems like predicting diabetes, where the outcome is either positive (presence of diabetes) or negative (absence of diabetes).

Aim of the Project
First up, we'll dive into Exploratory Data Analysis (EDA). It's like exploring a treasure map! We'll pick out the most important clues (features) and fill in any missing pieces. Then, we'll look for any weird outliers that could mess up our predictions. Think of it as making sure all the numbers are playing fair.
Once we've tidied up our data, it's time to train our model – think of it like teaching a smart computer brain to understand patterns in our data. But there's a twist – sometimes, we don't have an equal number of examples of people with and without diabetes. We'll have to deal with that imbalance to make sure our predictions are accurate.
Now, moving on to logistic regression. It's a simple yet powerful tool that helps us classify things. In our case, it'll help us decide if someone is likely to have diabetes based on their health info.
Finally, it's time to put our model to the test. We'll use it to make predictions on new data and see how well it does. It's like checking if our crystal ball is working.
This project aims to demonstrate the application of logistic regression in predicting diabetes occurrence based on various features such as glucose levels, blood pressure, BMI, age and more. By leveraging a dataset containing historical health data of individuals, we'll train a logistic regression model to discern patterns and relationships between these features and the likelihood of diabetes onset.
In any Machine Learning model, the first step is to truly understand the data we're working with. In this project, we'll be using the diabetes dataset, and we'll be working on Google Colab, a fantastic platform for data analysis and machine learning in a collaborative environment.

Setting Up Google Colab
Access Google Colab
Open your web browser and navigate to Google Colab
Sign in with your Google account
Create a New Notebook
Click on "New" to create a new notebook
Rename your notebook to something descriptive like "Diabetes Prediction"
1. Importing Libraries and Loading Data
Since Google Colab comes pre-installed with essential libraries, we can skip the installation step. However, if you are using a different platform, make sure to install the required libraries.

Now, let's import the necessary libraries and load our dataset, using the following code:




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
2. Data Loading and Exploration
In this step, we'll load the dataset containing diabetes-related data. This dataset typically includes features such as blood glucose levels, insulin levels, age, BMI, etc.

Download the dataset from here.

The code loads the dataset from a CSV file and displays the first few rows to get a glimpse of the data. It also shows the column names and data types to understand what information is available and how it is represented.




dataframe = pd.read_csv("/content/diabetes.csv")
dataframe.head()
dataframe.columns
dataframe.dtypes
The outputs will be as shown below:

head
dataframe.head() output
dtypes
dataframe.dtypes output
columns
dataframe.columns
3. Handling Missing Values
This step checks for missing values in each column of the dataset and displays the total count of missing values.




dataframe.isnull().sum()
miss
missing values
We see, here in our dataset, the missing values are not available

4. Observing set of Values
Let's observe first 20 set of data using following code:




dataframe.head(20)
Screenshot-2024-05-07-155451
Observing the Dataset
It’s confusing to see things like blood pressure (BP) and insulin recorded as zero in a dataset, especially when studying diabetes. Sometimes zero values happen because of mistakes or missing data, but they make us question if the data is accurate.

For example, if someone’s blood pressure is recorded as zero and they don’t have diabetes, this probably isn’t right. It’s almost impossible for a person to have zero blood pressure since it’s a basic body function that isn’t directly linked to diabetes. This shows why it’s important to check and clean the data carefully, looking out for strange or incorrect values.

Finding these kinds of mistakes means we need to clean the data properly before using it. To fix these problems, we might fill in missing values using averages or knowledge about the data, or sometimes we have to remove or mark wrong entries so they don’t mess up the results.


Feature selection involves identifying and selecting the most informative variables from the dataset, ensuring that only relevant features are included in the predictive model.

Data imputation addresses missing or invalid values in the dataset, ensuring that the data is complete and suitable for analysis and modeling.

1. Data Size
To determine the size of the dataset.

This code snippet provides the dimensions of the dataset, indicating the number of rows (records) and columns (features).




dataframe.shape
2. Target Column and Correlation Coefficient
To identify the target column for classification and explore the correlation between features.

The 'Outcome' column represents the target variable (0 for non-diabetic, 1 for diabetic). This is a binary classification task. The correlation coefficient matrix is computed to understand the relationships between different features.




dataframe.corr()
plt.figure(figsize=(15,15))
ax = sns.heatmap(dataframe.corr(), annot=True)
plt.savefig('correlation-coefficient.jpg')
plt.show()
3. Descriptive Statistics
To compute descriptive statistics for numerical features in the dataset.

Descriptive statistics such as mean, standard deviation, minimum, maximum, etc., are calculated for numerical features to understand their distribution and variability.




dataframe.describe()
maxx
4. Data Imputation
To handle missing or zero values in specific features through data imputation. This step visualizes the distribution of certain features using histograms and replaces zero values with appropriate statistics (median or mean) to address data gaps.




sns.distplot(dataframe.Pregnancies)
sns.distplot(dataframe.BloodPressure)
sns.distplot(dataframe.Insulin)
preg
sns.distplot(dataframe.Pregnancies)
blood
sns.distplot(dataframe.BloodPressure)
insulin
sns.distplot(dataframe.Insulin)
Use the following code to replace the missing values




dataframe['Insulin'] = dataframe['Insulin'].replace(0, dataframe['Insulin'].median())
dataframe['Pregnancies'] = dataframe['Pregnancies'].replace(0, dataframe['Pregnancies'].median())
dataframe['Glucose'] = dataframe['Glucose'].replace(0, dataframe['Glucose'].mean())
dataframe['BloodPressure'] = dataframe['BloodPressure'].replace(0, dataframe['BloodPressure'].mean())
dataframe['SkinThickness'] = dataframe['SkinThickness'].replace(0, dataframe['SkinThickness'].median())
dataframe['BMI'] = dataframe['BMI'].replace(0, dataframe['BMI'].mean())
dataframe['DiabetesPedigreeFunction'] = dataframe['DiabetesPedigreeFunction'].replace(0, dataframe['DiabetesPedigreeFunction'].median())
dataframe['Age'] = dataframe['Age'].replace(0, dataframe['Age'].median())
dataframe['Pregnancies'] = dataframe['Pregnancies']. replace(0, dataframe['Pregnancies'].median()):

This line replaces zero values in the 'Pregnancies' column with the median value of the same column. It ensures that instances where 'Pregnancies' are recorded as zero, which is unlikely, are replaced with a more reasonable estimate of the central tendency of pregnancies in the dataset.

dataframe['Glucose'] = dataframe['Glucose'].replace(0, dataframe['Glucose'].mean()):

This line replaces zero values in the 'Glucose' column with the mean value of the same column. It assumes that missing or zero glucose values can be replaced with the average glucose level in the dataset.

dataframe['BloodPressure'] = dataframe['BloodPressure'].replace(0, dataframe['BloodPressure'].mean()):

This line replaces zero values in the 'BloodPressure' column with the mean value of the same column. It assumes that missing or zero blood pressure values can be replaced with the average blood pressure in the dataset.

dataframe['SkinThickness'] = dataframe['SkinThickness'].replace(0, dataframe['SkinThickness'].median()):

This line replaces zero values in the 'SkinThickness' column with the median value of the same column. It ensures that instances where 'SkinThickness' is recorded as zero are replaced with a more reasonable estimate of the central tendency of skin thickness in the dataset.

dataframe['BMI'] = dataframe['BMI'].replace(0, dataframe['BMI'].mean()):

This line replaces zero values in the 'BMI' column with the mean value of the same column. It assumes that missing or zero BMI values can be replaced with the average BMI in the dataset.

dataframe['DiabetesPedigreeFunction'] = dataframe['DiabetesPedigreeFunction'].replace(0,dataframe['DiabetesPedigreeFunction'].median()):

This line replaces zero values in the 'DiabetesPedigreeFunction' column with the median value of the same column. It ensures that instances where 'DiabetesPedigreeFunction' is recorded as zero are replaced with a more reasonable estimate of the central tendency of diabetes pedigree function in the dataset.

dataframe['Age'] = dataframe['Age'].replace(0, dataframe['Age'].median()):

This line replaces zero values in the 'Age' column with the median value of the same column. It ensures that instances where 'Age' is recorded as zero are replaced with a more reasonable estimate of the central tendency of age in the dataset.

These replace operations are aimed at handling missing or invalid values in the dataset, ensuring that the data is more complete and suitable for subsequent analysis and modeling.

5. Data Imputation Strategies
Mean Imputation
The mean is a measure of central tendency that represents the average value of a dataset.
Mean imputation involves replacing missing or invalid values with the mean of the available data.
It is suitable for symmetrically distributed data without significant outliers.
Mean imputation assumes that the missing values are randomly distributed across the dataset and that the mean accurately represents the central tendency of the data.
Median Imputation
The median is another measure of central tendency that represents the middle value of a dataset when arranged in ascending order.
Median imputation involves replacing missing or invalid values with the median of the available data.
It is more robust to outliers and skewed distributions compared to the mean.
Median imputation is preferred when the data distribution is non-symmetric or when outliers are present.
Since the median is less influenced by extreme values, it provides a more representative estimate of the central tendency in such cases.
Mode Imputation for Categorical Data
Mode: The mode is a measure of central tendency that represents the most frequently occurring value in a dataset.
Mode Imputation: Mode imputation involves replacing missing values in categorical data with the mode of the respective column. This means filling in missing values with the value that appears most frequently in the dataset for that particular categorical variable.
Applicability: Mode imputation is suitable for categorical data where the missing values are assumed to be more likely to take on the most common category or class.


Outliers and data normalization are crucial aspects of exploratory data analysis (EDA) that ensure the reliability and integrity of the dataset for subsequent analysis and modeling. In the context of the predictive analysis of diabetes, outlier detection helps identify unusual data points that may skew results, while normalization ensures that data is scaled uniformly, facilitating fair comparisons between features.

Outlier Detection
Splitting the data into input features (X) and target value (y) is a necessary step before performing outlier detection. This separation allows us to focus solely on the input features when identifying outliers, as outliers in the target variable may not necessarily be indicative of anomalies in the input data. Here's how we can split the data and proceed with outlier detection:




## Splitting the data into input features (X) and target value (y)
X = dataframe.drop(columns='Outcome', axis=1)
y = dataframe['Outcome']
Now, we proceed with outlier detection using box plots and the Interquartile Range (IQR) method, applied individually to each feature in the input data (X).




fig, ax = plt.subplots(figsize = (15, 15))
sns.boxplot(data = X, ax=ax)
plt.savefig('boxPlot.jpg')
What is a Box Plot?
The box plot is a graphical representation of the distribution of a dataset. It displays key summary statistics such as the median, quartiles, and potential outliers concisely and visually.

A box plot gives a five-number summary of a set of data, which is:

Minimum: It is the minimum value in the dataset, excluding the outliers.
First Quartile (Q1): 25% of the data lies below the First (lower) Quartile.
Median (Q2): It is the mid-point of the dataset. Half of the values lie below it and half above.
Third Quartile (Q3): 75% of the data lies below the Third (Upper) Quartile.
Maximum: It is the maximum value in the dataset excluding the outliers.
Outliers: Data points lying outside the whiskers are identified as potential outliers. These are points that significantly deviate from the typical distribution of the data and may represent unusual observations, measurement errors, or rare events.
box
1. Define Outlier Boundaries using IQR Method
The outliers are filtered out based on the boundaries defined by the IQR method for each feature.




cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
for col in cols:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    mask = (X[col] >= lower_bound) & (X[col] <= upper_bound)
2. Filter dataset to remove outliers
Finally, the dataset is filtered to remove outliers, resulting in the outlier-detected input features (X_outlier_detection) and corresponding target values (y_outlier_detection), using following code:




# Filter dataset to remove outliers
X_outlier_detection = X[mask]
y_outlier_detection = y[mask]
3. Shape comparison before and after outlier_detection
The shapes of X and X_outlier_detection as well as y and y_outlier_detection indicate the number of samples (rows) and features (columns) in each dataset after outlier detection:

X: The original dataset of input features (X) has a shape of (768, 8), indicating that it contains 768 samples and 8 features.
X_outlier_detection: After outlier detection and filtering, the dataset of input features (X_outlier_detection) has a shape of (759, 8), indicating that it contains 759 samples and 8 features. This means that some samples were identified as outliers and removed from the original dataset during the outlier detection process.
y: The original dataset of target values (y) has a shape of (768,), indicating that it contains 768 samples.
y_outlier_detection: After outlier detection and filtering, the dataset of target values (y_outlier_detection) has a shape of (759,), indicating that it contains 759 samples. Since the outlier detection process is performed solely on the input features (X), the corresponding target values (y) are not affected, and the number of samples remains the same as in the original dataset.
Standardization
Standardization transforms numerical features to a standard normal distribution, facilitating comparisons between features. It preserves the relative relationships between data points while removing differences in scale. Standardization is essential for algorithms sensitive to feature scales, such as gradient descent-based optimization algorithms.




from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_outlier_detection)
In the provided code snippet, the StandardScaler from Scikit-learn is utilized to standardize the numerical features in the dataset X_outlier_detection. Standardization involves transforming the features to have a mean of 0 and a standard deviation of 1, ensuring uniform scaling across all features.

After standardization, the box plot may exhibit changes in the positioning of the median, the length of the box, and the placement of outliers. These changes reflect the transformation of the data distribution achieved through standardization, providing insights into the standardized distribution of the numerical features in the dataset.

Outliers, or data points that significantly deviate from the rest of the dataset, can have a substantial impact on statistical analysis and machine learning models. Exploratory Data Analysis (EDA) involves various techniques for identifying and handling outliers to ensure the robustness and reliability of the analysis.

Outlier Detection
Outlier detection plays a crucial role in ensuring the quality and accuracy of machine learning models. By identifying and removing or handling outliers effectively, we can prevent them from biasing the model, reducing its performance and hindering its interpretability. Here’s an overview of various outlier detection methods:

Handling outliers in a dataset that follows a normal distribution involves understanding how outliers affect the properties of the distribution and devising appropriate strategies to deal with them:

1. Identifying Outliers
In a normal distribution, outliers are data points that lie far away from the mean, typically beyond a certain number of standard deviations. They may occur due to measurement errors, rare events, or genuine anomalies in the data.

2. Z-Score Approach
One common method for identifying outliers in a normal distribution is using the z-score. The z-score measures how many standard deviations a data point is away from the mean. Data points with z-scores beyond a certain threshold (e.g., ±3) are considered outliers.

Handling Outliers
Once outliers are identified, there are several strategies for handling them:

1. Removal
Outliers can be removed from the dataset if they are deemed to be the result of errors or do not represent genuine data patterns. However, caution should be exercised to ensure that important information is not lost.

2. Winsorization
Winsorization replaces extreme values with less extreme values within a specified range. This approach reduces the influence of outliers while preserving the overall distribution of the data.

3. Transformation
Transforming the data, such as taking the logarithm or square root of the values, can sometimes mitigate the effects of outliers by compressing their influence on the distribution.

4. Robust Methods
Robust statistical methods, such as robust regression or trimmed means, are less affected by outliers compared to traditional methods. These techniques downweight the influence of outliers and provide more reliable estimates.

5. Statistical Methods
Z-Score: This method calculates the standard deviation of the data points and identifies outliers as those with Z-scores exceeding a certain threshold (typically 3 or -3).
Interquartile Range (IQR): IQR identifies outliers as data points falling outside the range defined by Q1-k*(Q3-Q1) and Q3+k*(Q3-Q1), where Q1 and Q3 are the first and third quartiles and k is a factor (typically 1.5).
Normal or Gaussian distribution: It is often referred to simply as the bell curve due to its characteristic shape and is a probability distribution that is symmetric and unimodal. It is defined by two parameters: the mean (μ) and the standard deviation (σ). In a normal distribution:
norm
Normal distribution
Symmetry: The distribution is symmetric around the mean, with the mean, median and mode being equal and located at the center of the distribution.
Unimodality: The distribution has a single peak, meaning that most data points cluster around the mean, with fewer points as you move away from the mean in either direction.
Step-by-Step Guide for Quantile Approach
1. Resetting Indexes
Purpose: Resetting the indexes ensures that both X_scaled and y_outlier_detection have continuous and consistent index values, starting from 0.




X_scaled.reset_index(drop=True, inplace=True)
y_outlier_detection.reset_index(drop=True, inplace=True)
Explanation: The reset_index() method with the drop=True parameter removes the existing index and replaces it with a new one starting from 0. The inplace=True parameter modifies the DataFrame in place, without creating a new object.

Output: After resetting the indexes, both X_scaled and y_outlier_detection now have continuous index values, facilitating further analysis and visualization.

2. Quantile-based Filtering
Purpose: Filtering the data based on a high quantile threshold (95th percentile) for the 'Insulin' feature aims to remove extreme values or potential outliers.




q = X_scaled['Insulin'].quantile(.95)
mask = X_scaled['Insulin'] < q
dataNew = X_scaled[mask]
y_outlier_detection = y_outlier_detection[mask]
Explanation: A threshold quantile (q) of 0.95 is computed for the 'Insulin' feature in the standardized dataset (X_scaled). Data points with 'Insulin' values below this threshold are retained (mask = X_scaled['Insulin'] < q), while those exceeding the threshold are filtered out.

3. Visualization with Box Plot
Purpose: Visualizing the distribution of the filtered data using a box plot helps assess the impact of outlier removal and observe the updated distribution of the 'Insulin' feature.




fig, ax = plt.subplots(figsize = (15, 15))
sns.boxplot(data = dataNew, ax=ax)
plt.savefig('boxPlot.jpg')
Explanation: A box plot is created for the filtered dataset (dataNew) to visualize the central tendency, spread and potential outliers of the 'Insulin' feature after filtering.

Output: The box plot provides insights into the distribution of 'Insulin' values in the filtered dataset, highlighting any changes in the presence of outliers compared to the original distribution.

bocplot
Box plot


Model Training Splitting of data and Handling Of Imbalanced Data

comments
So far, we have looked at our diabetes data to understand it and clean it up. We fixed missing information and made sure the data is ready to use.

Next, we will split the data into two parts: one to teach our model and one to test it to see how well it works. Also, since we have fewer examples of people with diabetes, we will balance the data so the model can learn better. This will help us build a model that can predict diabetes more accurately and help doctors take better care of patients.

Below are the further steps along with code for model training, splitting of data, and handling of imbalanced data:

1. Splitting the Data into Training and Testing Sets
Splitting the dataset into training and testing sets allows us to evaluate the model's performance on unseen data. We'll use the train_test_split function from Scikit-learn to randomly split the dataset into training and testing sets, typically with a ratio such as 80% for training and 20% for testing.




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataNew, y_outlier_detection, test_size=0.33, random_state=42)
shape
Shape after data splitting
2. Handling Imbalanced Data
Oversampling: Minority Class and increase the number of the majority class
Undersampling: Majority class and decrease the number of the minority class
SMOTE: Synthetic data and increase the number of samples of the majority class
Purpose: Imbalanced datasets occur when one class (e.g., positive cases of diabetes) is significantly underrepresented compared to another class (e.g., negative cases of diabetes). Handling imbalanced data is crucial to prevent the model from being biased towards the majority class.

Explanation: We can address class imbalance using techniques such as resampling (oversampling minority class or undersampling majority class), using appropriate evaluation metrics, or using algorithms specifically designed to handle imbalanced data (e.g., ensemble methods like Random Forest or boosting algorithms).

Before applying any resampling techniques, it's essential to understand the class distribution in our training data. For instance, in our dataset, there are 318 instances of the negative class (no diabetes) and 165 instances of the positive class (diabetes). This imbalance needs to be addressed to ensure that the model does not disproportionately favor the majority class.




print(y_train.value_counts())


Outcome:

count
Classes in Training Data
3. Code (Example of Oversampling with SMOTE)



from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
​
# Check resampled class distribution
print("\nResampled class distribution:")
print(pd.Series(y_train_resampled).value_counts())


Output:

outc
outcome
In this example, we use SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class (positive cases of diabetes) in the training set. The fit_resample method applies the SMOTE algorithm to generate synthetic samples for the minority class, balancing the class distribution. This ensures that our model is trained on a balanced dataset, reducing the risk of bias towards the majority class.

The core of our work is model training, where we apply logistic regression, a classification algorithm ideal for binary tasks like predicting diabetes. By carefully training, evaluating, interpreting, and optimizing the logistic regression model, we aim to create a reliable tool that accurately identifies diabetes cases from the data.

1. Model Training
Purpose: Model training involves fitting a logistic regression model to the training data to learn the relationship between the input features and the target variable.

Explanation: We'll use the logistic regression algorithm to build a predictive model for classifying instances of diabetes based on the input features. The model will be trained on the training dataset, which has been split into input features (X_train) and target labels (y_train).




from sklearn.linear_model import LogisticRegression
# Initialize Logistic Regression classifier
logistic_regression_model = LogisticRegression()
# Fit the model to the training data
logistic_regression_model.fit(X_train, y_train)
2. Model Evaluation
Purpose: After training the model, we need to evaluate its performance on unseen data (testing set) to assess its effectiveness in making predictions.

Explanation: We'll use the testing dataset (X_test and y_test) to evaluate the logistic regression model's performance. This involves making predictions on the testing set and comparing them to the actual labels to calculate evaluation metrics such as accuracy, precision, recall, and F1-score.

Code: Implement evaluation metrics such as accuracy_score, precision_score, recall_score, and f1_score from Scikit-learn's metrics module to evaluate the model's performance.

3. Model Interpretation
Purpose: Understanding the coefficients of the logistic regression model helps interpret the importance of each feature in predicting the target variable.

Explanation: We'll examine the coefficients (weights) assigned to each feature by the logistic regression model. Positive coefficients indicate features that positively contribute to the likelihood of diabetes, while negative coefficients indicate features that negatively contribute.

Code: Access the coefficients using the coef_ attribute of the trained logistic regression model.

4. Model Optimization
Purpose: Fine-tuning the logistic regression model's hyperparameters can improve its performance and generalization ability.

Explanation: We can optimize hyperparameters such as regularization strength (C), penalty (l1 or l2), solver algorithm, and class weight balancing to improve the model's performance. Techniques like cross-validation and grid search can be used for hyperparameter tuning.

Code: Implement hyperparameter tuning using techniques such as GridSearchCV from Scikit-learn's model_selection module to find the optimal combination of hyperparameters for the logistic regression model.

By following these steps, we can effectively train, evaluate, interpret, and optimize the logistic regression model for predicting instances of diabetes based on the input features. This process enables us to develop a reliable and accurate predictive model for diabetes diagnosis.


Model Prediction and Evaluation

comments
Now, we move to an important step where we use our trained model to make predictions and check how well it works. This means testing the model on new data it hasn’t seen before to see if it can correctly identify diabetes cases. This step helps us understand how accurate and reliable our model is.

Model Prediction
After we train our logistic regression model, we use it to make predictions on new data that the model hasn’t seen before. The model looks at each example and gives a score showing how likely it is to have diabetes. If the score is higher than a certain number (like 0.5), we say it’s positive for diabetes. If it’s lower, we say it’s negative (no diabetes).




y_predictions = classification.predict(X_test)
print(y_predictions)
predict
y_predictions output
Model Evaluation
To see how well our logistic regression model is doing, we use different measurements like accuracy, precision, recall, F1-score, and a confusion matrix. These help us understand how good the model is at correctly identifying diabetes cases and where it might make mistakes.




from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predictions)
​
from sklearn.metrics import classification_report
target_names = ['Non-Diabetic', 'Diabetic']
print(classification_report(y_test, y_predictions, target_names=target_names))


In our diabetes project, recall is very important because it shows how good our model is at finding people who really have diabetes.

Here’s why recall matters:

Finding People with Diabetes
Recall tells us how many of the actual diabetes cases our model correctly finds. A high recall means we catch most people who have diabetes, so they can get help quickly.

Avoiding Missed Cases
Sometimes the model might miss people who do have diabetes (these are called false negatives). This is bad because those people might not get the treatment they need. A high recall means fewer missed cases.

Helping Patients and Public Health
When we find more people with diabetes early, they can get treatment sooner. This helps them stay healthier and live better lives.

recall
Model Evaluation


After training a machine learning model, it's crucial to save it for later use or deployment. One common way to do this in Python is by using the pickle module. Following are the steps to save your trained model using Pickle.

1. Import Necessary Libraries
Pickle is a module in Python that serializes and deserializes Python objects. It allows you to save the state of your objects, including machine learning models, to a file.




import pickle
2. Save the Trained Model
The pickle.dump() function serializes the trained model (model) and writes it to the opened file (model_file). This effectively saves the model to a file.

Syntax:

pickle.dump(model, model_file)




import pickle
pickle.dump(classification, open("classification_model.pkl", "wb"))
This line opens a file named 'classification_model.pkl' in binary write mode ('wb'). The 'wb' mode is used for binary files, as pickling involves writing binary data.



open('classification_model.pkl', 'wb')

When you save a machine learning model using pickle and name the file as "classification_model.pkl", the resulting file will contain the serialized version of the trained regression model.

classification_model.pkl file

Here's what you can expect the "classification_model.pkl" file to have:

Serialized Model Parameters: The serialized version of the trained regression model, including all the parameters, coefficients, and other information necessary to represent the model's state.
Model Type Information: Information about the type of regression model (linear regression, ridge regression, etc.) and any specific settings or hyperparameters used during training.
Scikit-Learn Version Compatibility: Pickle files are sensitive to the version of the libraries used. The file may contain information about the version of scikit-learn or other libraries used to train the model.
Dependencies: If there are any custom functions or objects used in the model (e.g., custom transformer classes), the pickle file may also include the necessary information to recreate those objects.
It's important to note that the "classification_model.pkl" file is a binary file and is not meant to be human-readable. Its purpose is to store the internal state of the trained model so that it can be later loaded and used for making predictions without having to retrain the model.

3. Confirm the Model is Saved
Deserialization is the process of loading a pickled file back into a Python object. The pickle.load() function achieves this.

loaded_model = pickle.load(model_file)

The pickle.load() function reads the content of the file and deserializes it into the loaded_model object. This object can now be used for making predictions, just like the original model.




classification_model = pickle.load(open("classification_model.pkl", "rb"))
Scaling New Data
When you load a pre-trained machine learning model and use it to make predictions on new data without retraining, it's because the model has already learned the underlying patterns from the training data.

The model.predict() method is applied to the data point to obtain the model's prediction. The result is a numerical value representing the predicted diabetes on testing data.




classification_model.predict(X_test)
test




