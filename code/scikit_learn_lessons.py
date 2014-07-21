
# coding: utf-8

# # Scikit-learn (code file pandas_lessons.py)
# 
# ## A brief intro to machine learning
# 
# There's a fair bit of backround knowledge that's important to know before we dive into the code. The actual code is rather simple, but I want you to understand exactly what's going on.
# 
# ### What is machine learning?
# 
# Machine learning is the study and application of algorithms that learn from examples. It's concerned with constructing systems that learn from data, systems that can then make future predictions based on past input. It's based on the ideas of representation and generalization: the representation of relationships within the data, and the ability to generalize those relationships to new data. This means that we can create a model that will learn something from the data that we have, and then apply what it learns to data that it hasn't seen before. Machine learning provides a way to build executable data summaries; it helps us build better software by predicting more accurately on future inputs.
# 
# ### Why is it useful?
# 
# This is an important topic because machine learning is everywhere. For example, your email spam filter is already trained to mark certain emails as spam, based on things like frequency of capital letters or number of suspicious links within an email. If a spam email does get through to your inbox and you manually mark it as spam, your spam filter learns from that input, and will mark similar emails as spam in the future. Another example is Netflix's recommender system. The more movies you rate on Netflix, the more that the recommender system learns what kind of movies you like to watch. The system will then get better at recommending to you appropriate movie choices. Machine learning is especially useful in data analysis.
# 
# ### Some terms
# 
# - observation/instance/data point: these all mean the same thing, and that is one particular piece of the data that we can grab information about and learn relationships from.
# - label/class: in classification, the label/class is what we aim to classify our new data as. Ex: email as spam or not spam.
# - feature: features describe the data. Features of email spam could be number of capital letter or frequency of known spam words.
# - categorical: discrete and finite data; has categories. Ex. spam or not spam.
# - continuous: subset of real numbers, can take on any value between two points. Ex. temperature degrees.
# 
# ### Types of machine learning
# 
# #### Supervised
# Supervised learning is machine learning that makes use of labeled data. Supervised learning algorithms can use past observations to make future predictions on both categorical and continuous data. The two main types of supervised learning are classification and regression. Classification predicts labels, such as spam or not spam. Regression predicts the relationship between continuous variables, such as the relationship between temperature and elevation.
# 
# #### Unsupervised
# Unsupervised learning is used when the data is unlabeled. You might not know what you're looking for within your data, and unsupervised learning can help you figure it out. Clustering is an example of unsupervised learning, where data instances are grouped together in a way that observations in the same group are more similar to each other than to those in other groups. Another example is dimensionality reduction, where the number of random variables is reduced, and is used for both feature selection and feature extraction.
# 
# ## What is scikit-learn?
# Scikit-learn is an open-source machine learning module. The scikit-learn project is constantly being developed and improved, and it has a very active user community. The documentation on the website is very thorough with plenty of examples, and there are a number of tutorials and guides for learning how scikit-learn works.
# 
# ### Why scikit-learn?
# You might be wondering why you'd want to use Python and scikit-learn, rather than other popular tools like MATLAB or R. Because scikit-learn is open source, it's free to use! Also, it's currently the most comprehensive machine learning tool for Python. There are also a number of Python libraries that work well with scikit-learn and extend its capabilities. 
# 
# ## About this section
# We're going to cover supervised learning due to time constraints. We'll talk about a couple of classifiers as well as linear regression. For the final lesson of this section, we'll use the two classifiers we learn about, k-nearest neighbor and the Naive Bayes classifier, on our census_data dataset. We'll then compare the classifiers and see which one is better for our data.
# 
# ## Let's start with classification
# Classification, again, classifies data into specific categories, and solves the task of figuring out which category new data belong to. There are many different kinds of classifiers, and which one you want to use depends on your data. We're only going to be covering k-Nearest Neighbors (kNN) and the Naive Bayes classifier (NB) because they're among the simplest to implement and understand.
# 
# For both algorithms, I'll walk you through simple examples of each, so that you'll have an idea of how they work. I'll also show you how to evaluate the models we create.
# 
# Something important to notice in my examples is that when we train, we use a different dataset than when we predict. This is to avoid the problem of overfitting. So, what's overfitting? Well, let's say we train our model on the entire dataset. If we want to also test with that dataset, we won't be able to get an accurate picture of how good our model is, because now it knows our entire dataset by heart. This is why we split up our sets.
# 
# ## k-Nearest Neighbors
# 
# The k-Nearest Neighbors (kNN) algorithm finds a predetermined number of "neighbor" samples that are closest in distance to a starting data point and makes predictions based on the distances. kNN predicts labels by looking at the labels of its nearest neighbors. The metric used to calcuate the distances between points can be any distance metric measure, such as the Euclidean metric or the Manhattan distance.
# 
# kNN is useful when your data is linear in nature and can therefore be measured with a distance metric. Also, kNN does well when the decision boundary (or the delineation between classes) is hard to identify. 
# 
# kNN comes with a couple of caveats. If the classes in your dataset are unevenly distributed, the highest-occuring label will tend to dominate predictions. Also, choosing the *k* of kNN can be tricky. Choosing *k* deserves its own three hour tutorial, so we'll just go with the defaults for today.
# 
# ### Classifying in scikit-learn: kNN
# 
# As we go through these examples, you'll notice that the basic fitting and predicting process is basically the same, which is one of the things that makes scikit-learn relatively easy to use.
# 
# Let's start by reading in some data and its labels, and then split it up so we don't overfit. The default split for the train_test_split() function is 0.25, meaning that 75% of the data is split into the training set and %25 is split into the test set. If you want a different split, that's something that can be changed.



import pandas as pd
from sklearn.cross_validation import train_test_split

wine_data = pd.read_csv('../data/wine_data.csv')
wine_labels = pd.read_csv('../data/wine_labels.csv', squeeze=True)

wine_data_train, wine_data_test, wine_labels_train, wine_labels_test = train_test_split(wine_data, wine_labels)


# Here's what one row from wine_data_train looks like.
                


wine_data_train[0]


# Let's compare the lengths of the original DataFrame and the training set.



#print len(wine_data), len(wine_data_train)


# Now we can fit kNN to our training data. This is pretty easy. We create our estimator object and then use the fit() function to fit the algorithm to our data.



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(wine_data_train, wine_labels_train)


# And finally, let's use our fitted model to predict on new data.



pred = knn.predict(wine_data_test)
#print pred

# Let's look at the real labels.



#print wine_labels_test


# You can see that there are some differences between the predictions and the actual labels. Let's actually calculate how accurate our classifier is. We can do that using cross-validation. Cross-validation is a method that takes a dataset, randomly splits it into training and test sets, and computes how accurate the model is by checking it against the real labels. It does this multiple times, and splits the dataset differently each time. 
# 
# The cross_val_score function takes several parameters. The first is the model you've fitted (in this case it's knn), the second is the entire dataset, the second is the entire list of labels, and if you'd like you can specify how many times you want to cross-validate (the cv parameter).



from sklearn.cross_validation import cross_val_score

#print cross_val_score(knn, wine_data, wine_labels, cv=5)


# So our model is approximately 70% accurate. That's not so great, but you get the idea.

# ## Lesson: classification with kNN!



# We're going to be using scikit-learn's built in datasets for this.

from sklearn.datasets import load_iris

iris = load_iris()
iris_data = iris.data
iris_labels = iris.target

# Can you split the data into training and test sets? Uncomment and fill in.

#iris_data_train, iris_data_test, ...fill in the rest!




# Now, let's use the training data and labels to train our model. Uncomment and fill in.

#knn_iris = fill in the rest




# And now, let's predict on our test set.





# Let's compare the predictions to the actual labels. Output the real labels.





# Let's score our model using cross-validation to see how good it is.



# ## Naive Bayes
# 
# The Naive Bayes classifier is a probabilistic classifier based on Bayes' Theorem, which states that the probability of *A* given the probability of *B* is equal to the probability of *B* given *A* times the probability of *A*, divided by the probability of *B*. In Naive Bayes classification, the classifier assumes that the features in your dataset are independent of each other; that is, one feature being a certain way has no effect on what values the other features take. This is a naive assumption because this doesn't always hold true in reality, but despite this naivety and oversimplified assumptions, the classifier performs decently and even quite well in certain classification situations.
# 
# The Naive Bayes classifier is useful when your features are independent and your data is normally distributed. More sophisticated methods generally perform better.
# 
# ### Classifying in scikit-learn: Naive Bayes
# 
# We're going to basically do the same thing we just did, but with a different classifier. We're going to use the GaussianNB estimator object, because our data is for the most part normally distributed. We're also going to use the same wine training and test sets we made earlier.



from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(wine_data_train, wine_labels_train)
gnb_pred = gnb.predict(wine_data_test)
#print gnb_pred

# Well, that was easy! Let's look at the real test labels again.



#print wine_labels_test


# And let's cross-validate again. Let's only run it four times.



#print cross_val_score(gnb, wine_data_train, wine_labels_train, cv=4)


# Wow! This classifier does much better on this dataset.

# ## Lesson: classification with Naive Bayes!



# We're going to be using scikit-learn's built in datasets for this.

from sklearn.datasets import load_digits

digits = load_digits()
digits_data = digits.data
digits_labels = digits.target

# Once again, split the data into training and test sets.




# Fit the model to our data.

gnb_digits = GaussianNB()





# Predict on the test set





# Look at the test set labels





# Finally, cross-validate



# ## Linear regression
# 
# Linear regression is used when the target value is expected to be a linear combination of the input variables. The goal of linear regression, in creating a linear model, is to minimize the sum of squared residuals between the observed data and the responses predicted by linear approximation. Linear regression can be used to represent the relationship between variables like temperature and elevation, or something like housing prices and square footage.
# 
# Linear regression is appropriate when your data is continuous and linear.
# 
# ### Linear regression in scikit-learn
# 
# Let's try this on subset of our wine data, since those values are continuous other than wine_type. Let's see what the relationship is between magnesium and abv. First, let's subset the data.



wine_data_mag = wine_data.loc[:, ['magnesium', 'color']]
wine_data_abv = wine_data.loc[:, 'abv']
#print wine_data_mag.head()


# And, as always, let's split up the data. Our target values are going to be the continuous abv values.



wine_mag_train, wine_mag_test, wine_abv_train, wine_abv_test = train_test_split(wine_data_mag, wine_data_abv)


# Then, we fit the model to our data.


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(wine_mag_train, wine_abv_train)


# And finally, we predict.


lr_pred = lr.predict(wine_mag_test)
#print lr_pred

# Let's compare those predictions to the actual abv values.



#print wine_abv_test


# We can check the accuracy of our linear regression model by using the score() function. The score() function returns the R^2 coefficient, which is a measure of how far away from the actual values are predictions were. The closer to 1, the better the regression model



#print lr.score(wine_mag_test, wine_abv_test)


# So our score is rather low.
# 
# ## Lesson: linear regression!



# We're going to be using scikit-learn's built in datasets for this.

from sklearn.datasets import load_boston

boston = load_boston()
boston_data = boston.data
boston_target = boston.target

# Once again, split the data into training and test sets.




# Fit the model to our data.

boston_lr = LinearRegression()





# Predict on the test set.





# Take a look at the actual target values.





# Score the model!



# # For those using IPython Notebook/Wakari/NBViewer: Go to the data_analysis notebook!
# 
# # For those using code files, go to data_analysis.py!
