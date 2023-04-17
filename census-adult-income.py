#!/usr/bin/env python
# coding: utf-8

# # Income Prediction - Machine Learning End-to-End Project
# 
# This lab is a guided project that will walk you through the process of building a machine learning model to predict whether an adult's income exceeds $50K/yr based on census data.
# 
# > The data set was extracted in 1996. Don't make life decisions based on your findings here.
# For information about this dataset, what parameters were used to extract this information from the Census (maybe you can do this with more current data) [Go here](https://archive.ics.uci.edu/ml/datasets/Adult)
# 
# **Objectives**
# - Practice building a machine learning project from start to finish
# 
# **Emojis Legend**
# - 👨🏻‍💻 - Instructions; Tells you about something specific you need to do.
# - 🦉 - Tips; Will tell you about some hints, tips and best practices
# - 📜 - Documentations; provides links to documentations
# - 🚩 - Checkpoint; marks a good spot for you to commit your code to git
# - 🕵️ - Tester; Don't modify code blocks starting with this emoji

# ## Setup
# First, let's import a few common modules, ensure `MatplotLib` plots figures inline. We also ensure that you have the correct version of Python (3.10) installed.
# 
# - **Task 👨🏻‍💻**: Keep coming back to update this cell as you need to import new packages.
# - **Task 👨🏻‍💻**: Check what's already been imported here

# In[1]:


# Python ≥3.10 is required
import sys
assert sys.version_info >= (3, 10)

# Common imports
import numpy as np
import pandas as pd
import pandas_profiling
import os

# Scikit Learn imports
## For the pipeline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
## For preprocessing
from sklearn.preprocessing import (
  OneHotEncoder,
  OrdinalEncoder,
  StandardScaler
)
from sklearn.impute import (
  SimpleImputer
)
## For model selection
from sklearn.model_selection import (
  StratifiedShuffleSplit,
  train_test_split,
  cross_val_score,
  KFold,
  GridSearchCV
)

# Classifier Algorithms
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (
  RandomForestClassifier, 
  GradientBoostingClassifier,
  BaggingClassifier
)

# To save and load models
import pickle

# To plot pretty figures
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
plt.style.use("bmh")

# to make this notebook's output stable across runs
np.random.seed(42)


# ## Adult Dataset
# 
# ### 1️⃣ Ask
# The dataset is credited to Ronny Kohavi and Barry Becker and was drawn from the 1994 United States Census Bureau data and involves using personal details such as education level to predict whether an individual will earn more or less than $50,000 per year.
# > "Extraction was done by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0))"
# 
# The dataset provides 14 input variables that are a mixture of categorical, ordinal, and numerical data types. The complete list of variables is as follows:
# 
# - **age**: continuous.
# - **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
# - **fnlwgt**: continuous.
# - **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# - **education-num**: continuous.
# - **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
# - **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
# - **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
# - **race**: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
# - **sex**: Female, Male.
# - **capital-gain**: continuous.
# - **capital-loss**: continuous.
# - **hours-per-week**: continuous.
# - **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
# 
# The dataset contains missing values that are marked with a question mark character (?).
# 
# There are a total of 48,842 rows of data, and 3,620 with missing values, leaving 45,222 complete rows.
# 
# The project objective is to develop a model that can predict whether a person makes over 50K a year.
# 

# ### 2️⃣ Prepare
# Here we will load the dataset and split it into training and test sets. We will also perform some Exploratory Data Analysis to get some insights about the processing steps we'll need to take.

# **Task 👨🏻‍💻** : Load the dataset from `data/adult.csv` and store it in a variable called `adult`

# In[2]:


adult = pd.read_csv("data/adult.csv")


# We need to learn about the composition of the dataset. Let's look at the first few rows of the dataset.
# 
# **Task 👨🏻‍💻** : Use the `head()` method to look at the first few rows of the dataset.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/adult-assignment/head.png" />
# </details>

# In[3]:


adult.head()


# > 🚩 : Make a git commit here

# We need to know the number of rows and columns in the dataset. Let's use the `.shape` attribute to find out.
# 
# **Task 👨🏻‍💻:** Use the `.shape` attribute to find out the number of rows and columns in the dataset.
# 
# *Hint 🦉 :* 
# - `.shape` is an attribute not a method/function; you don't add `()` at the end of it to execute it.
# - `.shape` It returns a tuple of the form `(rows, columns)`

# In[4]:


adult.shape


# We need to check if we have any missing values in the dataset. Let's use the `isnull()` method to find out.
# 
# **Task 👨🏻‍💻** : Use the `isnull()` method to find out if there are any missing values in the dataset.
# 
# *Hint 🦉 :* 
# - `isnull()` returns a dataframe of the same shape as the original dataframe with boolean values. `True` indicates a missing value and `False` indicates a non-missing value.
# - because it would be difficult to read a dataframe of boolean values, we can use the `sum()` method to get the total number of missing values in each column. `sum()` returns a series with the column names as the index and the total number of missing values in each column as the values.
#   - We've done this in a previous lab, so you can refer to that if you need to.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/missing1.png" />
# </details>
# 

# In[5]:


adult.isnull().sum()


# **But that's not true** The dataset contains missing values that are marked with a question mark character (?). As indicated in the dataset description.
# 
# We have 2 alternatives to deal with missing values:
# - we can replace the `?` with `NaN`. And we have to do this for all 14 columns. This is tedious and error-prone.
# - We can re-import the dataset and specify that `?` is a missing value. using the `na_values` parameter.
# 
# **Task 👨🏻‍💻** : Use the `read_csv()` method to re-import the dataset and specify that `?` is a missing value. using the `na_values` parameter.
# 
# *Hint 🦉 :*
# - you want to override the `adult` variable with the new dataframe. So you can use the same variable name.
# 

# In[6]:


adult = pd.read_csv("data/adult.csv", na_values="?")


# **Task 👨🏻‍💻** : Check again if we have any missing values in the dataset using the `isnull()` method.
# 
# *Hint 🦉 :*
# - you can use the same code as before to check for missing values.
# - Same hints apply; we want to use the `sum()` method to get the total number of missing values in each column.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/missing2.png" />
# </details>

# In[7]:


adult.isnull().sum()


# We could also use the `info()` method to get a summary of the dataset. Let's use it to find out more about the dataset.
# 
# **Task 👨🏻‍💻** : Use the `info()` method to get a summary of the dataset.
# 
# *Hint 🦉 :* 
# - `info()` returns a summary of the dataset. 
# - It includes the number of rows and columns, the number of **non-missing values** in each column, the data type of each column and the memory usage of the dataframe.

# In[8]:


adult.info()


# We could also use the `describe()` method to get a summary of the dataset. Let's use it to find out more about the dataset.
# 
# **Task 👨🏻‍💻** : Use the `describe()` method to get a summary of the dataset.
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/describe.png" />
# </details>

# In[9]:


adult.describe()


# > 🚩 : Make a git commit here

# We also need to visualize the data we have to get more insights about the dataset. Let's use the `hist()` method to visualize the distribution of the data.
# 
# **Task 👨🏻‍💻** : Use the `.hist()` method to plot a histogram of each column in the dataset.
# 
# *Hint 🦉 :* 
# - use the `figsize` parameter to set the size of the figure to `(20, 15)`
# - use the `bins` parameter to set the number of bins to `50`
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="800" src="https://github.com/IT4063C/images/raw/main/adult-assignment/adult_hist.png" />
# </details>

# In[10]:


adult.hist(bins=50, figsize=(20,15))
plt.show()


# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : <u><strong>For each</strong></u> of the (8) categorical columns, use the `value_counts()` method to get the number of unique values in each column.
# 
# *Hint 🦉 :*
# - `value_counts()` returns a series with the unique values as the index and the number of occurrences of each unique value as the values.
# - if you want to put them all in a one cell, you'll need to wrap each call in the `display()` function.
# 
# <details>
#   <summary>the work-class category for example should look like this: (🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/work-class-count.png" />
# </details>

# In[11]:


adult['work-class'].value_counts()


# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : <u><strong>For each</strong></u> of the (8) categorical columns, Plot a bar chart that shows the distribution of the values in each column.
# 
# *Hint 🦉 :*
# - [matplotlib - plotting categorical_variables](https://matplotlib.org/stable/gallery/lines_bars_and_markers/categorical_variables.html)
# - You could als call the `plot` function on the output of the `value_counts()` method. But you'll need to specify the `kind` parameter to `bar` to get a bar chart.
# 
# <details>
#   <summary>Work class bar chart, for example, should look like this: (🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/work-class-count-bar.png" />
# </details>
# 
# **✨ For Extra Credit 👨🏻‍💻**: <u>For 5 points of Extra Credit:</u> Plot all the categorical columns in a single figure using subplots.

# In[12]:


adult['work-class'].value_counts().plot(kind='bar')


# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : Make some observations about each of the features in the histograms. 
# 
# *Hint 🦉 :*
# Here are some questions to get you started:
# - What do you notice about the distribution of the data?
# - when imputing the data, what values would you use for each feature? (mean, median, most frequent/mode) and why?
# - For the categorical features we have, what type of categorical features are they? (nominal, ordinal)
🦉: type your observations in this cell here

# > 🚩 : Make a git commit here

# ### 🏖 ☕️ Take a break here
# make sure you understand what we've done so far.
# 
# ____________________________

# #### Split the dataset into training and test sets
# Now before we go any further, we need to split the dataset we have into two parts:
# - a training set
# - a test set
# 
# This step is important because we need to train our model, then test it against some data that it hasn't seen before. If we don't do this, we won't be able to tell if our model is overfitting or not.

# Becuase the dataset here is quite large, maybe we won't need to do a stratified sampling. and we can just do with the a random split using `train_test_split()`.
# 
# **Task 👨🏻‍💻** : Use the `train_test_split()` function from the `sklearn.model_selection` package to split the dataset into a training set named `train_set` and a test set and `test_set`.
# 
# *Hint 🦉 :* 
# - use the `random_state` parameter to set the random seed to `32` - this will ensure that we get the same results every time we run the code.
# - use the `test_size` parameter to set the size of the test set to `0.2`; the test set is 20% of the size of the dataset.

# In[13]:


train_set, test_set = train_test_split(adult, test_size=0.2, random_state=42)


# > 🚩 : Make a git commit here

# **✨ Extra Credit Task 👨🏻‍💻**: <u>For 2 points of Extra Credit:</u> Perform a stratified split on the dataset. 
# - Justify your choice of the target column to perform the stratified split on. 
# - Make sure the test set is 20% of the size of the dataset.

# In[ ]:





# > 🚩 : Make a git commit here

# #### Separate the features and labels
# Let's separate the features `X` from the labels `y`. We'll use the training set for this.

# **Task 👨🏻‍💻** : Create a copy of the training set <u>without</u> the output `income` and store it in a variable called `adult_X`, and create a copy of the dataset with <u>Only</u> the column `income` and name it `adult_y`.

# In[14]:


adult_X = train_set.drop('income', axis=1)
adult_y = train_set[['income']].copy()


# <details>
#   <summary>Running the following cell, should produce an output that looks like this: (🦉 click me)</summary>
#   <img width="600" src="https://github.com/IT4063C/images/raw/main/adult-assignment/x_y_split.png" />
# </details>

# In[15]:


display(adult_X.head())
display(adult_y.head())


# > 🚩 : Make a git commit here

# ### 🏖 ☕️ Take a break here
# make sure you understand what we've done so far.
# 
# ________________________________

# ### 3️⃣ Process
# In this section, we'll process and clean it in preparation for the model creation and analysis work the data. 
# 
# Here are some of what we will do:
# - dropping columns
# - impute missing values (numerical data)
# - scale numerical features (numerical data)
# - encode categorical features (categorical data)
# 
# We will also compose all of these steps into a single pipeline.

# #### Compose the pipeline 
# We'll use the `ColumnTransformer` class to compose the pipeline. The `ColumnTransformer` class allows us to apply different transformations to different columns in the dataset.
# 
# **Task 👨🏻‍💻** : Create a `ColumnTransformer` object called `full_pipeline` that applies the following transformations to the dataset:
# 
# *Hint 🦉 :*
# Here's a template to get you started: 
#   - You'll need to rename the variables, and the pipelines, and you'll also need to fill in the missing parts.
# ```python
# feature_group1 = []
# feature_group2 = []
# 
# group1_pipeline = Pipeline([
# 
# ])
# 
# group2_pipeline = Pipeline([
# 
# ])
# 
# full_pipeline = ColumnTransformer([
#   ('group1', group1_pipeline, feature_group1),
#   ('group1', group2_pipeline, feature_group2)
# ])
# ```

# In[16]:


feature_group1 = []
feature_group2 = []

group1_pipeline = Pipeline([

])

group2_pipeline = Pipeline([

])

full_pipeline = ColumnTransformer([
  ('group1', group1_pipeline, feature_group1),
  ('group1', group2_pipeline, feature_group2)
])


# #### Impute Missing Values
# We'll use the `SimpleImputer` class from the `sklearn.impute` package to impute the missing values in the dataset.
# 
# The dataset showed that we're missing values in the `work-class`, `occupation`, and `native-country` columns. All of which are categorical columns.
# 
# You already answered a question above regarding what strategy you would use to impute the missing values in the categorical columns. I'm only going to give away the answer here for the categorical columns, you'll still need to justify why.
# 
# > *Keep in mind*, even if there was no missing values in the dataset, we would still need to implement the imputer. Because we need to make sure that our processing pipeline can be applied to new data that <u>may</u> have missing values. 
# 
# So even though the numerical columns don't have missing values, we'll still need to implement the imputer for them.
# 
# **Task 👨🏻‍💻** : Start modifying the template pipeline provided above, and add an imputation step to the pipelines.
# 
# *Hint 🦉 :*
# - Add a `SimpleImputer` step that imputes the missing values in the dataset using the appropriate strategies.
# - For the categorical columns, use the `most_frequent` strategy.
# - For the numerical columns, use whatever strategy you think is best.
#   - you can create multiple pipelines for the numerical columns, and use the `ColumnTransformer` to apply the appropriate pipeline to the appropriate column.
#   - For example if you think that certain columns should use the `median` and others should use the `mean`, you can create two pipelines.
#   - Similar to what done on the categorical columns in the Titanic Notebook.
# 

# In[17]:


num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_features = ['work-class', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

num_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='median')),
])

cat_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='most_frequent')),
])

full_pipeline = ColumnTransformer([
  ('num', num_pipeline, num_features),
  ('cat', cat_pipeline, cat_features)
])


# if the pipeline was implemented correctly, the following cell should produce an output that looks like this:
# 
# <details>
#   <summary>(🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/transform1.png" />
# </details>
# 
# > **Note**: Keep in mind, all the `sklearn` transforms return a numpy array. if you want to print the data such that we can see them with the column names as we're used to in `pandas`, we need to convert that back to a DataFrame. (demonstrated in the following cell)
# 

# In[18]:


# Transform the data
adult_prepared = full_pipeline.fit_transform(adult_X)

# Transform the numpy n-dimensional array into a pandas dataframe
adult_prepared = pd.DataFrame(adult_prepared, columns=full_pipeline.get_feature_names_out(), index=adult_X.index)

# confirm we no longer have missing values
adult_prepared.isnull().sum()


# noting how the column names were modified by the pipeline, we'll need to rename them back to their original names using **list_comprehension** as demonstrated in the notebook.

# In[19]:


# Transform the data
adult_prepared = full_pipeline.fit_transform(adult_X)

column_names = [ 
  feature
    .replace('num__', '')
    .replace('cat__', '') 
  for feature in full_pipeline.get_feature_names_out()
]

# Transform the numpy n-dimensional array into a pandas dataframe
adult_prepared = pd.DataFrame(adult_prepared, columns=column_names, index=adult_X.index)

# confirm we no longer have missing values
adult_prepared.isnull().sum()


# > 🚩 : Make a git commit here

# #### Scaling and Normalizing Numerical Features
# <img width="500" src="https://github.com/IT4063C/images/raw/main/adult-assignment/adult_hist.png" />
# 
# As you can see from the histograms and the `describe()` output, the values in each column are on different scales. This leads the machine learning algorithms giving more weight to the features. We need to scale the values in each column to the same scale. 

# **Task 👨🏻‍💻** : use the `StandardScaler` class to transform/scale the values in each column to the same scale.
# 
# *Hint 🦉 :* 
# - This only applies to the numerical columns, so you'll need to modify the numerical pipeline adding this step.

# In[20]:


num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
cat_features = ['work-class', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

num_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='median')),
  ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='most_frequent')),
])

full_pipeline = ColumnTransformer([
  ('num', num_pipeline, num_features),
  ('cat', cat_pipeline, cat_features)
])


# if the pipeline was implemented correctly, the following cell should produce an output that looks like this:
# 
# <details>
#   <summary>(🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/transform2.png" />
# </details>

# In[21]:


# Transform the data
adult_prepared = full_pipeline.fit_transform(adult_X)

column_names = [ 
  feature
    .replace('num__', '')
    .replace('cat__', '') 
  for feature in full_pipeline.get_feature_names_out()
]

# Transform the numpy n-dimensional array into a pandas dataframe
adult_prepared = pd.DataFrame(adult_prepared, columns=column_names, index=adult_X.index)
adult_prepared.head()


# > 🚩 : Make a git commit here

# #### Encode Categorical Features
# Let's now process and transform the categorical features. In the videos we mentioned 2 types of categorical feature encoders: 
# - `OrdinalEncoder` for ordinal categories. 
# - and `OneHotEncoder`for nominal categories.
# 
# here are all the categorical data we have: (I might be wrong about the classification)
# - `work-class` - nominal
# - `education` - ordinal
# - `marital-status` - nominal
# - `occupation` - nominal
# - `relationship` - nominal
# - `race` - nominal
# - `sex` - nominal
# - `native-country` - nominal

# **Task 👨🏻‍💻** : Modify the `full_pipeline` to encode the categorical features using the appropriate encoders.
# 
# *Hint 🦉 :*
# - We will need to separate the categorical columns into ordinal and nominal columns. 
# - Then we'll need to create a pipeline for each of them.
# - We will now have:
#   - one or more numerical pipelines (depending on how you implemented the imputer)
#   - 2 categorical pipelines (one for ordinal and one for nominal)
# 

# In[22]:


num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
ordinal_cat_features = ['education']
# nominal_cat_features = ['work-class', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
nominal_cat_features = ['work-class', 'marital-status', 'occupation', 'relationship', 'race', 'sex']

num_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='median')),
  ('std_scaler', StandardScaler())
])

nominal_cat_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('encode', OneHotEncoder())
])

ordinal_cat_pipeline = Pipeline([
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('encode', OrdinalEncoder())
])

full_pipeline = ColumnTransformer([
  ('num', num_pipeline, num_features),
  ('ord_cat', ordinal_cat_pipeline, ordinal_cat_features),
  ('nom_cat', nominal_cat_pipeline, nominal_cat_features),
])


# if the pipeline was implemented correctly, the following cell should produce an output that looks like this:
# 
# <details>
#   <summary>(🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/transform2.png" />
# </details>
# 
# - not how because the `oneHotEncoder` returned a sparse matrix instead of a nd-array, we needed to convert it to a numpy array when passing it to the DataFrame constructor
# 

# In[23]:


# Transform the data
adult_prepared = full_pipeline.fit_transform(adult_X)

column_names = [ 
  feature
    .replace('num__', '')
    .replace('ord_cat__', '') 
    .replace('nom_cat__', '') 
  for feature in full_pipeline.get_feature_names_out()
]

# Transform the numpy n-dimensional array into a pandas dataframe
## Because the oneHotEncoder returned a sparse matrix, we needed to convert it to array when passing it to the DataFrame constructor
adult_prepared = pd.DataFrame(data=adult_prepared.toarray(), columns=column_names, index=adult_X.index)
adult_prepared.head()


# > 🚩 : Make a git commit here

# For the `sex_Female` and `sex_Male` columns, we can drop one of them. This is because the values in the other column can be inferred from the values in the other column.
# 
# **Task 👨🏻‍💻** : Drop the `sex_Male` column from the dataset, and rename the `sex_Female` column to `is_female`.
# 
# *Hint 🦉 :*
# - in the notebook I chained the `drop()` and `rename()` methods.
# - I didn't do enough research to know if there is a way to do this as part of the pipeline. **(Extra Credit)**
# 
# <details>
#   <summary>Output of the `head()` function should look like this: (🦉 click me)</summary>
#   <img width="100" src="https://github.com/IT4063C/images/raw/main/adult-assignment/transform4.png" />
# </details>
# 
# **✨ Extra Credit Task 👨🏻‍💻**: <u>For 7 points of Extra Credit:</u> find a way to drop the `sex_Male` column from the dataset, and rename the `sex_Female` column to `is_female` as part of the pipeline.
# - You may not necessarily need to have a column drop step.
# - There might be a way to do this as part of the `OneHotEncoder` step.
# - Maybe a Custom Transformer.
# 

# In[24]:


adult_prepared = adult_prepared.drop('sex_Male', axis=1).rename(columns={ 'sex_Female': 'is_female' })
adult_prepared.head()


# ### 🏖 ☕️ Take a break here
# make sure you understand what we've done so far.
# 
# ______________

# ### 4️⃣ Analyze
# In this section, we'll train  our machine learning models to make predictions about whether a person makes over 50K a year.
# 
# At this stage we should have the following datasets:
# - `adult_prepared` - the inputs for the training set
# - `adult_y` - the outputs for the training set
# - `test_set` - the test set (both X and y)

# In[25]:


# ⛔️ Do not uncomment this cell. This is what I used to save a copy of the prepared data to csv files.

# adult_prepared.to_csv('data/adult_prepared.csv', index=False)
# adult_y.to_csv('data/adult_y.csv', index=False)

# test_set.to_csv('data/test_set.csv', index=False)


# Just in case you didn't get the prepare and process steps right, you can use the `adult_prepared`, `adult_y`, and `test_set` datasets that I've created for you.
# 
# uncomment the following cell and execute it.

# In[26]:


# adult_prepared = pd.read_csv("data/adult_prepared.csv")
# adult_y = pd.read_csv("data/adult_y.csv")
# test_set = pd.read_csv("data/test_set.csv")


# We'll start with a dummy classifier to get a baseline for our models.
# 
# **Task 👨🏻‍💻** : Create a dummy classifier and train it on the training set. `adult_prepared` and `adult_y` are the inputs and outputs for the training set respectively.
# 
# *Hint 🦉 :*
# - The `DummyClassifier` class is in the `sklearn.dummy` module and is already imported for you.
# - use the `most_frequent` strategy for the `strategy` parameter.

# In[27]:


# imported DummyClassifier and the metrics packages (first cell)
dummy_classifier = DummyClassifier(strategy='most_frequent')
dummy_classifier.fit(adult_prepared, adult_y)


# To actually get a baseline, we need some metrics to compare the performance of our models against. We'll use `.score()` function (via the model itself). 
# 
# We will also use the Area-Under-Curve Score using the`AUC` scoring function in the `cross_val_score` function.
# 
# **Task 👨🏻‍💻** : print the `dummy_classifier`'s score.
# 

# In[28]:


dummy_classifier.score(adult_prepared, adult_y)


# **Task 👨🏻‍💻** : Use the `cross_val_score` function to get the AUC score for the dummy classifier.

# In[29]:


scores = cross_val_score(
  dummy_classifier, adult_prepared, adult_y,
  scoring="roc_auc", cv=10)
print(
    f"Dummy Classifier  AUC: {scores.mean():.3f} STD: {scores.std():.2f}"
)


# Using the same approach demonstrated in this module's notebook, we'll train a few models and compare their performance.
# 
# **Task 👨🏻‍💻** : Train multiple models and compare their performance.
# 
# *Hint 🦉 :*
# - These are the different models you need to test (all of their libraries are imported for you):
#   - DummyClassifier,
#   - DecisionTreeClassifier,
#   - RandomForestClassifier,
#   - GradientBoostingClassifier,
#   - BaggingClassifier
# -  when the target variable has columns called anything other than `y`, the `cross_val_score` function throws a lot of errors. For that instead of passing the labels as `adult_y`, you'll need to pass `adult_y['income']`
# -  Set up the kfolds variable, the same way as on this week's notebook. (to ensure we get the same results)
# 
# ```python
#   kfold = KFold(
#         n_splits=10, random_state=42, shuffle=True
#     )
# ```
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="650" src="https://github.com/IT4063C/images/raw/main/adult-assignment/models-comparison.png" />
# </details>

# In[33]:


for model in [
    DummyClassifier,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier
]:
    classifier_model = model()
    # defining the kfolds, will ensure that all models will be trained with the same data
    kfold = KFold(
        n_splits=10, random_state=42, shuffle=True
    )
    scores = cross_val_score(
        classifier_model,
        adult_prepared, 
        adult_y['income'], 
        scoring="roc_auc", cv=kfold
    )
    print(
    f"{model.__name__:22}  AUC: {scores.mean():.3f} STD: {scores.std():.2f}"
)


# Use the ConfusionMatrixDisplay to visualize the confusion matrix for the best model.
# 
# **Task 👨🏻‍💻** : Use the `ConfusionMatrixDisplay` to visualize the confusion matrix for the best model.
# 
# *Hint 🦉 :*
# - you don't need to provide `display_labels` like we did in the notebook. The `ConfusionMatrixDisplay` will automatically use the labels from the `adult_y` dataset.
# 
# 
# <details>
#   <summary>Output should look like this: (🦉 click me)</summary>
#   <img width="650" src="https://github.com/IT4063C/images/raw/main/adult-assignment/confusion-matrix.png" />
# </details>

# In[36]:


gb_model = GradientBoostingClassifier()
gb_model.fit(adult_prepared, adult_y['income'])


metrics.ConfusionMatrixDisplay.from_estimator(
  estimator=gb_model,
  X=adult_prepared, y=adult_y,
  cmap="Blues", colorbar=False
)
plt.show()


# **✨ Extra Credit Task 👨🏻‍💻**: <u>For 5 points of Extra Credit:</u> Choose a model that performed well and try to improve its performance by tuning its hyperparameters using the `GridSearchCV` class.
# 
# *Hint 🦉 :*
# - demonstrated in this week's notebook
# 

# In[ ]:





# > 🚩 : Make a git commit here

# #### Evaluate against the test set
# Let's evaluate the models using the test set.
# 
# **Task 👨🏻‍💻** : Calculate the `auc_score` for the best model using the test set.
# 
# *Hint*:
# - remember to separate the X and y from the test set
# - remember to transform the X using the `full_pipeline`
# - remember to drop and renamed the sex columns; we didn't do that as part of the pipeline

# In[ ]:





# > 🚩 : Make a git commit here

# **Task 👨🏻‍💻** : Plot the confusion matrix for the best model using the test set.

# In[ ]:





# > 🚩 : Make a git commit here

# ### 5️⃣. Deploy
# When you're ready to deploy the model, we don't need to keep running the transformation and training pipelines. instead we can save the trained model and load when needed.
# 
# **Task 👨🏻‍💻** :use the `pickle` module to save the best model to a file called `best_model.pkl`.

# In[ ]:





# > 🚩 : Make a git commit here

# ## Wrap up
# - **Task 👨🏻‍💻** : Remember to update the self reflection and self evaluations on the `README` file.
# - **Task 👨🏻‍💻** : Make sure you run the following cell. It converts this Jupyter notebook to a Python script. This allows me to provide feedback on your code.
# 

# In[ ]:


get_ipython().system('jupyter nbconvert --to python diabetes-analysis.ipynb')


# > 🚩 : Make a git commit here
