#!/usr/bin/env python
# coding: utf-8

# In[42]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj2a.ok')


# # Project 2 Part A: Spam/Ham Classification
# ## EDA, Feature Engineering, Classifier
# ### The assignment is due on Monday, April 20th at 11:59pm PST.
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# ## This Assignment
# In project 2, you will use what you've learned in class to create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails. In addition to providing some skeleton code to fill in, we will evaluate your work based on your model's accuracy and your written responses in this notebook.
# 
# After this project, you should feel comfortable with the following:
# 
# - Feature engineering with text data
# - Using sklearn libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 
# In project 2A, you will undersatand the data through EDAs and do some basic feature engineerings. At the end, you will train your first logistic regression model to classify Spam/Ham emails. 
# 
# ## Warning
# We've tried our best to filter the data for anything blatantly offensive as best as we can, but unfortunately there may still be some examples you may find in poor taste. If you encounter these examples and believe it is inappropriate for students, please let a TA know and we will try to remove it for future semesters. Thanks for your understanding!

# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 1
# 1b | 1
# 1c | 2
# 2 | 3
# 3a | 2
# 3b | 2
# 4 | 2
# 5 | 2
# Total | 15

# In project 2a, we will try to undersatand the data and do some basic feature engineerings for classification.

# In[43]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading in the Data
# 
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). Your labeled training dataset contains 8348 labeled examples, and the test set contains 1000 unlabeled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that you will use to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. You will predict labels for these emails and submit your predictions to Kaggle for evaluation.

# In[44]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()


# ### Question 1a
# First, let's check if our data contains any missing values. Fill in the cell below to print the number of NaN values in each column. If there are NaN values, replace them with appropriate filler values (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings). Print the number of NaN values in each column after this modification to verify that there are no NaN values left.
# 
# Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels. Doing so without consideration may introduce significant bias into our model when fitting.
# 
# *The provided test checks that there are no missing values in your dataset.*
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[45]:


empty_string = ''
post_missing = []

def f(x):
    for index, value in x.items():
        if pd.isnull(value):
            x.iloc[index] = 'Missing'
    return x

for i in original_training_data.columns:
    original_training_data.apply(f)
    post_missing += [1 for k in original_training_data[i] if pd.isnull(k)]
    
print(f"There are now " + str(sum(post_missing)) + " missing values.")



# In[46]:


ok.grade("q1a");


# ### Question 1b
# 
# In the cell below, print the text of the first ham and the first spam email in the original training set.
# 
# *The provided tests just ensure that you have assigned `first_ham` and `first_spam` to rows in the data, but only the hidden tests check that you selected the correct observations.*
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 1
# -->

# In[47]:


first_ham = original_training_data[original_training_data['spam'] == 0][['email']].iloc[0,0]
first_spam = original_training_data[original_training_data['spam'] == 1][['email']].iloc[0,0]
print(first_ham)
print(first_spam)


# In[48]:


ok.grade("q1b");


# ### Question 1c
# 
# Discuss one thing you notice that is different between the two emails that might relate to the identification of spam.
# 
# <!--
# BEGIN QUESTION
# name: q1c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# One thing I noticed is that the spam email contains HTML related text, so elements such as '<' and '>', which might be helpful in identifying spam if this feature is consistent. 

# ## Training Validation Split
# The training data we downloaded is all the data we have available for both training models and testing the models that we train.  We therefore need to split the training data into separate training and testing datsets. Note that we set the seed (random_state) to 42. This will produce a pseudo-random sequence of random numbers that is the same for every student. **Do not modify this in the following questions, as our tests depend on this random seed.**

# In[49]:


from sklearn.model_selection import train_test_split

train, test = train_test_split(original_training_data, test_size=0.1, random_state=42)


# # Basic Feature Engineering
# 
# We would like to take the text of an email and predict whether the email is ham or spam. This is a *classification* problem, so we can use logistic regression to train a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.  Unfortunately, our data are text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $X$ is an email. Each column of $X$ contains one feature for all the emails. We'll guide you through creating a simple feature, and you'll create more interesting ones when you are trying to increase your accuracy.

# ### Question 2
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello worldhello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```
# 
# *The provided tests make sure that your function works correctly, so that you can use it for future questions.*
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 3
# -->

# In[50]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
    indicator_array = np.asarray([[1 if words[i] in j else 0 for i in range(len(words))] 
                                  for j in [texts.iloc[k] for k in range(texts.size)]])
    return indicator_array


# In[51]:


ok.grade("q2");


# # Basic EDA
# 
# We need to identify some features that allow us to distinguish spam emails from ham emails. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is itself a binary indicator, such as whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 

# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words. 
# 
# ![training conditional proportions](./images/training_conditional_proportions.png "Class Conditional Proportions")
# 
# Hint:
# - You can use DataFrame's `.melt` method to "unpivot" a DataFrame. See the following code cell for an example.

# In[52]:


from IPython.display import display, Markdown
df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our Original DataFrame has some words column and a type column. You can think of each row as a sentence, and the value of 1 or 0 indicates the number of occurances of the word in this sentence."))
display(df);
display(Markdown("> `melt` will turn columns into variale, notice how `word_1` and `word_2` become `variable`, their values are stored in the value column"))
display(df.melt("type"))


# ### Question 3a
# 
# Create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words. Choose a set of words that are different from the ones above, but also have different proportions for the two classes. Make sure to only consider emails from `train`.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# points: 2
# manual: true
# image: true
# -->
# <!-- EXPORT TO PDF -->

# In[53]:


train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts

#Create table new_table
array = words_in_texts(['win', 'buy', 'life', 'earn'], train['email'])
new_table = pd.concat([train, pd.DataFrame(array)], axis=1)
new_table.columns = ['id', 'subject', 'email', 'spam', 'win', 'buy', 'life', 'earn']
new_table = new_table[['spam', 'win', 'buy', 'life', 'earn']]
new_table[['spam']] = new_table[['spam']].applymap(lambda x: 'spam' if x == 1 else 'ham')
new_table = new_table.melt('spam')
new_table = pd.pivot_table(new_table, values='value', index=['variable', 'spam'], aggfunc=np.mean, fill_value=0)
new_table = new_table.reset_index()
#Visualization:
ax = sns.barplot(x="variable", y="value", hue="spam", data=new_table)
ax.set(xlabel = 'Words', ylabel = 'Proportion of Emails', title = 'Frequency of Words in spam/ham emails');


# When the feature is binary, it makes sense to compare its proportions across classes (as in the previous question). Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes. 
# 
# ![training conditional densities](./images/training_conditional_densities2.png "Class Conditional Densities")
# 

# ### Question 3b
# 
# Create a *class conditional density plot* like the one above (using `sns.distplot`), comparing the distribution of the length of spam emails to the distribution of the length of ham emails in the training set. Set the x-axis limit from 0 to 50000.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# points: 2
# manual: true
# image: true
# -->
# <!-- EXPORT TO PDF -->

# In[54]:


dist_table = train[['spam']]
dist_table['Length of email body'] = train["email"].apply(lambda x: len(x))
dist_table.head()

ax1 = sns.distplot(dist_table["Length of email body"][dist_table["spam"] == 0], hist=False, label = "Ham")
ax1 = sns.distplot(dist_table["Length of email body"][dist_table["spam"] == 1], hist=False, label = "Spam")
ax1.set(ylabel = 'Distribution', xlim=(0,50000));


# # Basic Classification
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a classifier!

# ### Question 4
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `train` DataFrame to create two NumPy arrays: `X_train` and `Y_train`.
# 
# `X_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# 
# `Y_train` should be a vector of the correct labels for each email in the training set.
# 
# *The provided tests check that the dimensions of your feature matrix (X) are correct, and that your features and labels are binary (i.e. consists of 0 and 1, no other values). It does not check that your function is correct; that was verified in a previous question.*
# <!--
# BEGIN QUESTION
# name: q4
# points: 2
# -->

# In[55]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = train['spam']

X_train[:5], Y_train[:5]


# In[56]:


ok.grade("q4");


# ### Question 5
# 
# Now that we have matrices, we can use to scikit-learn! Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, train a logistic regression model using `X_train` and `Y_train`. Then, output the accuracy of the model (on the training data) in the cell below. You should get an accuracy around 0.75.
# 
# *The provided test checks that you initialized your logistic regression model correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q5
# points: 2
# -->

# In[57]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression(solver="lbfgs")
model.fit(X_train, Y_train)

training_accuracy = accuracy_score(Y_train, model.predict(X_train))
print("Training Accuracy: ", training_accuracy)


# In[58]:


ok.grade("q5");


# You have trained your first logistic regression model and it can correctly classify around 76% of the training data! Can we do better than this? The answer is yes! In project 2B, you will learn to evaluate your classifier. Moreover, you will have the chance to extract your own features and build your own classifier!

# ## Submission
# Congratulations! You are finished with this assignment. Please don't forget to submit by 11:59pm PST on Monday, 04/20!

# # Submit
# Make sure you have run all cells in your notebook in order before running the cell below, so that all images/graphs appear in the output.
# **Please save before submitting!**
# 
# <!-- EXPECT 3 EXPORTED QUESTIONS -->

# In[59]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj2a.ipynb', 'proj2a.pdf')
ok.submit()


# In[ ]:




