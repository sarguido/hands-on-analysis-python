
# coding: utf-8

# # Lesson: Let's preprocess the data!
# 
# ## File I/O and exploring the data


import pandas as pd

f = '../data/census_data.csv'

# read in the file


# look at the first 5 lines




# What are the column names?



# Sort the DataFrame by age and print out the last 5 lines


# ## Selecting columns and rows


# create a subset of the data with the columns education, occupation, hours_per_week. Look at the first 5 rows.



# create a subset of the data with the middle 50 rows and the columns work_class and race. Look at the first 5 rows.



# create a subset of the data where education_num is greater than 8 and where sex is equal to female. Look at the first 5 rows


# ## groupby


# Group by work_class and output the group names (hint: add .keys() to the end of your line of code).



# Let's group by work_class and use the mean as the aggregate function



# ## Pivoting


# Let's pivot on education_num and sex, with hours_per_week as the values and mean as the aggfunc




# Create a pivot table of your choosing, with any columns for rows and cols and a numerical columns for values.



# ## Let's do the following:
# 
# For the machine learning section, can you extract a subset of the data where:
# 
# - native_country equals United-States
# - hours_per_week is greater than 10
# - age is greater than 20 and less than 50
# - education is Masters
# 
# It's going to be a bunch of booleans!


# Store that new dataframe in new_df and print out the first five rows.




# Split the DataFrame so that all of the columns except the last one are in new_df_data, and the last one is in new_df_labels.

new_df_data = new_df.iloc[:, :10]
new_df_labels = new_df.iloc[:, -1]


# # For those using IPython Notebook/Wakari/NBViewer: Go to the scikit_learn notebook!
# 
# # For those using code files, go to scikit_learn_materials.py!
# 
# End of Pandas section.
# 
# # Lesson: let's classify our data!
# 
# ## Final preprocessing touches
# 
# Scikit-learn estimators take in continuous data only, which means that we'll have to transform our categorical data into something the scikit-learn estimators can handle. This is actually much easier than it sounds! We're going to convert our dataframe into a dictionary, and then encode the data in that dictionary into arrays of 1s and 0s.
# 
# Let's first transform the dataframe into a dictionary. We first have to transpose our DataFrame, so there is one row per nested dictionary. Finally, we'll put each item into a list, because scikit-learn's DictVectorizer object takes a list of dictionaries to encode. We also only need the values from our list of dictionaries.


new_df_transpose = new_df_data.transpose()

data_into_dict = new_df_transpose.to_dict()
census_data = [v for k, v in data_into_dict.iteritems()]


# Now, let's encode those features and instances.


from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer()
transformed_data = dv.fit_transform(census_data).toarray()
#print transformed_data


# Now that we've done that, let's encode the labels.


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
transformed_labels = le.fit_transform(new_df_labels)


# Now that we've done that, can you separate the transformed_data and transformed_labels into training and test sets?


from sklearn.cross_validation import train_test_split


# Let's fit and predict both kNN and Naive Bayes to this data.


from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# fit both

# predict


# Outbpu kNN predictions on first 10 labels



# Output NB predictions on first 10 labels



# Let's cross validate each, and then I'll show you something cool you can do.


# Run cross-validation on kNN. Set cv=5




# Run cross-validation on NB. Set cv=5


# # For those using IPython Notebook/Wakari/NBViewer: Go to the matplotlib notebook!
# 
# # For those using code files, go to matplotlib_materials.py!
# 
# End of scikit-learn section.
# 
# # Lesson: what can plots tell us about our data and results?


# Let's look at the distribution of classes in our dataset. Can you get the value_counts of new_df_labels and make a bar chart?




# Now, can you make a bar chat comparing the predicted kNN labels to the actual labels? You can use the first 10 only if you want.



# What other visualizations would be helpful? Can you come up with a few more?


