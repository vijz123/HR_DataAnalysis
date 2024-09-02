#!/usr/bin/env python
# coding: utf-8

# # **Providing data-driven suggestions for HR**

# 
# 

# ### business scenario and problem
# 
# The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?
# 

# ### Familiarize yourself with the HR dataset
# 
# The dataset that you'll be using in this lab contains 15,000 rows and 10 columns for the variables listed below. 
# 
# **Note:** you don't need to download any data to complete this lab. For more information about the data, refer to its source on [Kaggle](https://www.kaggle.com/datasets/mfaisalqureshi/hr-analytics-and-job-prediction?select=HR_comma_sep.csv).
# 
# Variable  |Description |
# -----|-----|
# satisfaction_level|Employee-reported job satisfaction level [0&ndash;1]|
# last_evaluation|Score of employee's last performance review [0&ndash;1]|
# number_project|Number of projects employee contributes to|
# average_monthly_hours|Average number of hours employee worked per month|
# time_spend_company|How long the employee has been with the company (years)
# Work_accident|Whether or not the employee experienced an accident while at work
# left|Whether or not the employee left the company
# promotion_last_5years|Whether or not the employee was promoted in the last 5 years
# Department|The employee's department
# salary|The employee's salary (U.S. dollars)

# 💭
# ### Reflect on these questions as you complete the plan stage.
# 
# *  Who are your stakeholders for this project?
# - What are you trying to solve or accomplish?
# - What are your initial observations when you explore the data?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 
# 

# [Double-click to enter your responses here.]

# ## Step 1. Imports
# 
# *   Import packages
# *   Load dataset
# 
# 

# ### Import packages

# In[2]:


# Import packages
### YOUR CODE HERE ### 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns',None)

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from xgboost import plot_importance

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV , train_test_split
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score, confusion_matrix,ConfusionMatrixDisplay, classification_report
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import plot_tree

import pickle


# ### Load dataset
# 
# `Pandas` is used to read a dataset called **`HR_capstone_dataset.csv`.**  As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.

# In[3]:


# RUN THIS CELL TO IMPORT YOUR DATA. 

# Load dataset into a dataframe
### YOUR CODE HERE ###
df0 = pd.read_csv("HR_capstone_dataset.csv")


# Display first few rows of the dataframe
### YOUR CODE HERE ###
df0.head()


# ## Step 2. Data Exploration (Initial EDA and data cleaning)
# 
# - Understand your variables
# - Clean your dataset (missing data, redundant data, outliers)
# 
# 

# ### Gather basic information about the data

# In[4]:


# Gather basic information about the data
### YOUR CODE HERE ###
df0.info()


# ### Gather descriptive statistics about the data

# In[5]:


# Gather descriptive statistics about the data
### YOUR CODE HERE ###
df0.describe()


# ### Rename columns

# As a data cleaning step, rename the columns as needed. Standardize the column names so that they are all in `snake_case`, correct any column names that are misspelled, and make column names more concise as needed.

# In[7]:


# Display all column names
### YOUR CODE HERE ###
df0.columns


# In[8]:


# Rename columns as needed
### YOUR CODE HERE ###
df0 = df0.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})


# Display all column names after the update
### YOUR CODE HERE ###
df0.columns


# ### Check missing values

# Check for any missing values in the data.

# In[9]:


# Check for missing values
### YOUR CODE HERE ###
df0.isna().sum()


# ### Check duplicates

# Check for any duplicate entries in the data.

# In[12]:


# Check for duplicates
### YOUR CODE HERE ###
df0.duplicated().sum()


# In[14]:


# Inspect some rows containing duplicates as needed
### YOUR CODE HERE ###
df0[df0.duplicated()].head()


# In[15]:


# Drop duplicates and save resulting dataframe in a new variable as needed
### YOUR CODE HERE ###
df1 = df0.drop_duplicates(keep='first')

# Display first few rows of new dataframe as needed
### YOUR CODE HERE ###
df1.head()


# ### Check outliers

# Check for outliers in the data.

# In[16]:


# Create a boxplot to visualize distribution of `tenure` and detect any outliers
### YOUR CODE HERE ###
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure',fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()


# In[18]:


# Determine the number of rows containing outliers
### YOUR CODE HERE ###
percentile25 = df1['tenure'].quantile(0.25)

percentile75 = df1['tenure'].quantile(0.75)

iqr = percentile75 - percentile25

upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print('Lower limit:',lower_limit)
print('Upper limit:',upper_limit)

outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]

print("Number of rows in the data containing outliers in tenure:",len(outliers))


# Certain types of models are more sensitive to outliers than others. When you get to the stage of building your model, consider whether to remove outliers, based on the type of model you decide to use.

# # pAce: Analyze Stage
# - Perform EDA (analyze relationships between variables)
# 
# 

# 💭
# ### Reflect on these questions as you complete the analyze stage.
# 
# - What did you observe about the relationships between variables?
# - What do you observe about the distributions in the data?
# - What transformations did you make with your data? Why did you chose to make those decisions?
# - What are some purposes of EDA before constructing a predictive model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 
# 

# [Double-click to enter your responses here.]

# ## Step 2. Data Exploration (Continue EDA)
# 
# Begin by understanding how many employees left and what percentage of all employees this figure represents.

# In[19]:


# Get numbers of people who left vs. stayed
### YOUR CODE HERE ###
print(df1['left'].value_counts())
print()

# Get percentages of people who left vs. stayed
### YOUR CODE HERE ###
print(df1['left'].value_counts(normalize=True))


# ### Data visualizations

# Now, examine variables that you're interested in, and create plots to visualize relationships between variables in the data.

# In[20]:


# Create a plot as needed
### YOUR CODE HERE ###
fig, ax = plt.subplots(1,2,figsize = (22,8))

sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')

# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')

# Display the plots
plt.show()


# In[21]:


# Create a plot as needed
### YOUR CODE HERE ###
df1[df1['number_project']==7]['left'].value_counts()


# In[24]:


# Create a plot as needed
### YOUR CODE HERE ###
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');


# In[25]:


# Create a plot as needed
### YOUR CODE HERE ###
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')

plt.show();


# In[26]:


# Create a plot as needed
### YOUR CODE HERE ###
df1.groupby(['left'])['satisfaction_level'].agg([np.mean,np.median])


# In[27]:


# Create a plot as needed
### YOUR CODE HERE ###
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]

# Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]

# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')

# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1, 
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');


# In[28]:


# Create a plot as needed
### YOUR CODE HERE ###
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');


# In[29]:


# Create a plot as needed
### YOUR CODE HERE ###
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14');


# ### Insights

# [What insights can you gather from the plots you created to visualize the data? Double-click to enter your responses here.]

# # paCe: Construct Stage
# - Determine which models are most appropriate
# - Construct the model
# - Confirm model assumptions
# - Evaluate model results to determine how well your model fits the data
# 

# 🔎
# ## Recall model assumptions
# 
# **Logistic Regression model assumptions**
# - Outcome variable is categorical
# - Observations are independent of each other
# - No severe multicollinearity among X variables
# - No extreme outliers
# - Linear relationship between each X variable and the logit of the outcome variable
# - Sufficiently large sample size
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the constructing stage.
# 
# - Do you notice anything odd?
# - Which independent variables did you choose for the model and why?
# - Are each of the assumptions met?
# - How well does your model fit the data?
# - Can you improve it? Is there anything you would change about the model?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# [Double-click to enter your responses here.]

# ## Step 3. Model Building, Step 4. Results and Evaluation
# - Fit a model that predicts the outcome variable using two or more independent variables
# - Check model assumptions
# - Evaluate the model

# ### Identify the type of prediction task.

# [Double-click to enter your responses here.]

# ### Identify the types of models most appropriate for this task.

# [Double-click to enter your responses here.]

# ### Modeling
# 
# Add as many cells as you need to conduct the modeling process.

# In[30]:


### YOUR CODE HERE ###
df_enc = df1.copy()

# Encode the `salary` column as an ordinal numeric category
df_enc['salary'] = (
    df_enc['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode the `department` column
df_enc = pd.get_dummies(df_enc, drop_first=False)

# Display the new dataframe
df_enc.head()


# # pacE: Execute Stage
# - Interpret model performance and results
# - Share actionable steps with stakeholders
# 
# 

# ✏
# ## Recall evaluation metrics
# 
# - **AUC** is the area under the ROC curve; it's also considered the probability that the model ranks a random positive example more highly than a random negative example.
# - **Precision** measures the proportion of data points predicted as True that are actually True, in other words, the proportion of positive predictions that are true positives.
# - **Recall** measures the proportion of data points that are predicted as True, out of all the data points that are actually True. In other words, it measures the proportion of positives that are correctly classified.
# - **Accuracy** measures the proportion of data points that are correctly classified.
# - **F1-score** is an aggregation of precision and recall.
# 
# 
# 
# 
# 

# 💭
# ### Reflect on these questions as you complete the executing stage.
# 
# - What key insights emerged from your model(s)?
# - What business recommendations do you propose based on the models built?
# - What potential recommendations would you make to your manager/company?
# - Do you think your model could be improved? Why or why not? How?
# - Given what you know about the data and the models you were using, what other questions could you address for the team?
# - What resources do you find yourself using as you complete this stage? (Make sure to include the links.)
# - Do you have any ethical considerations in this stage?
# 
# 

# Double-click to enter your responses here.

# ## Step 4. Results and Evaluation
# - Interpret model
# - Evaluate model performance using metrics
# - Prepare results, visualizations, and actionable steps to share with stakeholders
# 
# 
# 

# ### Summary of model results
# 
# Logistic Regression
# 
# The logistic regression model achieved precision of 80%, recall of 83%, f1-score of 80% (all weighted averages), and accuracy of 83%, on the test set.
# 
# Tree-based Machine Learning
# 
# After conducting feature engineering, the decision tree model achieved AUC of 93.8%, precision of 87.0%, recall of 90.4%, f1-score of 88.7%, and accuracy of 96.2%, on the test set. The random forest modestly outperformed the decision tree model.

# ### Conclusion, Recommendations, Next Steps
# 
# The models and the feature importances extracted from the models confirm that employees at the company are overworked.
# 
# To retain employees, the following recommendations could be presented to the stakeholders:
# 
# Cap the number of projects that employees can work on.
# Consider promoting employees who have been with the company for atleast four years, or conduct further investigation about why four-year tenured employees are so dissatisfied.
# Either reward employees for working longer hours, or don't require them to do so.
# If employees aren't familiar with the company's overtime pay policies, inform them about this. If the expectations around workload and time off aren't explicit, make them clear.
# Hold company-wide and within-team discussions to understand and address the company work culture, across the board and in specific contexts.
# High evaluation scores should not be reserved for employees who work 200+ hours per month. Consider a proportionate scale for rewarding employees who contribute more/put in more effort.
# 
# Next Steps
# 
# It may be justified to still have some concern about data leakage. It could be prudent to consider how predictions change when last_evaluation is removed from the data. It's possible that evaluations aren't performed very frequently, in which case it would be useful to be able to predict employee retention without this feature. It's also possible that the evaluation score determines whether an employee leaves or stays, in which case it could be useful to pivot and try to predict performance score. The same could be said for satisfaction score.
# 
# For another project, you could try building a K-means model on this data and analyzing the clusters. This may yield valuable insight.

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.
