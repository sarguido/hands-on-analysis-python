from numpy import mean, std
# coding: utf-8

# # Pandas (code file pandas_lessons.py)
# 
# ## The importance of data preprocessing
# 
# Data preprocessing (also called data wrangling, data munging, cleaning, scrubbing, etc) is the most important thing you will do with your data because it sets the stage for the analysis part of your data analysis workflow. The preprocessing you do largely depends on what kind of data you have, what sort of analysis you'll be doing with your data, and what you intend to do with the results.
# 
# Preprocessing is also a process for getting to know your data, and can answer questions such as these (and more): 
# 
# - What kind of data are you working with? 
# - Is it categorical, continuous, or a mix of both? 
# - What's the distribution of features in your dataset? 
# - What sort of wrangling do you have to do?
# - Do you have any missing data? 
# - Do you need to remove missing data?
# - Do you need only a subset of your data?
# - Do you need more data?
# - Or less?
# 
# The questions you'll have to answer are, again, dependent upon the data that you're working with, and preprocessing can be a way to figure that out.
# 
# ## What is Pandas?
# 
# Pandas is by far my favorite preprocessing tool. It's a data wrangling/modeling/analysis tool that is similar to R and Excel; in fact, the DataFrame data structure in Pandas was named after the DataFrame in R. Pandas comes with several easy-to-use data structures, two of which (the Series and the DataFrame) I'll be covering here.
# 
# I'll also be covering a bunch of different wrangling tools, as well as a couple of analysis tools.
# 
# ## Why Pandas?
# 
# So, why would you want to use Python, as opposed to tools like R and Excel? I like to use it because I like to keep everything in Python, from start to finish. It just makes it easier if I don't have to switch back and forth between other tools. Also, if I have to build in preprocessing as part of a production system, which I've had to do at my job, it makes sense to just do it in Python from the beginning. 
# 
# Pandas is great for preprocessing, as we'll see, and it can be easily combined with other modules from the scientific Python stack.
# 
# ## Pandas data structures
# 
# Pandas has several different data structures, but we're going to talk about the Series and the DataFrame.
# 
# ### The Series
# 
# The Series is a one-dimensional array that can hold a variety of data types, including a mix of those types. The row labels in a Series are collectively called the index. You can create a Series in a few different ways. Here's how you'd create a Series from a list.



import pandas as pd

some_numbers = [2, 5, 7, 3, 8]

series_1 = pd.Series(some_numbers)
#print series_1


# To specify an index, you can also pass in a list.



ind = ['a', 'b', 'c', 'd', 'e']

series_2 = pd.Series(some_numbers, index=ind)
#print series_2


# We can pull that index back out again, too, with the index attribute.

#print series_2.index


# You can also create a Series with a dictionary. The keys of the dictionary will be used as the index, and the values will be used as the Series array.



more_numbers = {'a': 9, 'b': 'eight', 'c': 7.5, 'd': 6}

series_3 = pd.Series(more_numbers)
#print series_3


# Notice how, in that previous example, I created a Series with integers, a float, and a string.

# ### The DataFrame
# 
# The DataFrame is Pandas' most used data structure. It's a two and greater dimensional structure that can also hold a variety of mixed data types. It's similar to a spreadsheet in Excel or a SQL table. You can create a DataFrame with a few different methods. First, let's look at how to create a DataFrame from multiple Series objects.



combine_series = pd.DataFrame([series_2, series_3])
#print combine_series


# Notice how in column 'b', we have two kinds of data. If a column in a DataFrame contains multiple types of data, the data type (or dtype) of the column will be chosen to accomodate all of the data. We can look at the data types of different columns with the dtypes attribute. Object is the most general, which is what has been chosen for column 'b'.



#print combine_series.dtypes


# Another way to create a DataFrame is with a dictionary of lists. This is pretty straightforward:



data = {'col1': ['i', 'love', 'pandas', 'so', 'much'],
        'col2': ['so', 'will', 'you', 'i', 'promise']}

df = pd.DataFrame(data)
#print df


# ## File I/O
# 
# It's really easy to read data into Pandas from a file. Pandas will read your file directly into a DataFrame. There are multiple ways to read in files, but they all work in the same way. Here's how you read in a CSV file:



wine = pd.read_csv('../data/wine.csv')
#print wine.head()


# Reading in a text file is just as easy. Make sure to pass in '\t' to the delimter parameter.



auto_mpg = pd.read_csv('../data/auto_mpg.txt', delimiter="\t")
#print auto_mpg.head()


# ## Exploring the data
# 
# Here are some different ways to explore the data we have. You've already seen the head() function, which returns the first five lines in the dataset. To grab the last 5 lines, you can use the tail() function:


#print auto_mpg.tail()


# Getting column names from a DataFrame is also easy and can be done using the 'columns' attribute.


#print wine.columns


# Another useful thing you can do is generate some summary statistics using the describe() function. The describe() function calculates descriptive statistics like the mean, standard deviation, and quartile values for continuous and integer data that exist in your dataset. Don't worry, Pandas won't try to calculate the standard deviation of your categorical values!



#print wine.describe()


# Another useful thing you can do to explore your data is to sort it. Let's say we wanted to sort our auto_mpg DataFrame by mpg. This is very easy as well:



#print auto_mpg.sort(columns='mpg').tail()


# ## Lesson: let's see what's going on in our data!
# 
# This dataset is data on credit approvals. The column names and data were changed to protect the confidentiality of the data.


f = '../data/credit_approval.csv'

# How do you read in that file? Uncomment and fill in.
#credit_approval = your code here

# Can you grab just the column names?




# Now, look at the first 5 lines




# Now, look at the last 5 lines




# Can you describe() the data? (Notice how Pandas only "describes" the numerical data!)




# Let's sort on column H


# ## Working with dataframes
# 
# Pandas has a ton of functionality for manipulating and wrangling the data. Let's look at a bunch of different ways to select and subset our data.
# 
# ### Selecting columns and rows
# 
# There are multiple ways to select by both rows and columns. From index to slicing to label to position, there are a variety of methods to suit your data wrangling needs.
# 
# Let's select just the mpg column from the auto_mpg DataFrame. This works similar to how you would access values from a dictionary:


#print auto_mpg['mpg']


# You can do exactly the same thing by using mpg as an attribute:



#print auto_mpg.mpg


# To extract rows from a DataFrame, you can use the slice method, similar to how you would slice a list. Here's how we would grab rows 7-13 from the wine DataFrame:



#print wine[7:14]


# Pandas also has tools for purely label-based selection of rows and columns using the loc attribute. The loc indexer takes input as [row, column]. For example, let's say we wanted to select the abv value in the 8th instance in our wine DataFrame:



#print wine.loc[8,'abv']


# We can also use loc to grab slices. Let's grab the abv for rows 8 to 11 from the wine DataFrame.



#print wine.loc[8:11, 'abv']


# And, as you might expect, we can select multiple columns by passing in a list of column names. Let's also grab ash and color for rows 8 to 11.



#print wine.loc[8:11, ['abv', 'ash', 'color']]


# Finally, let's just grab all columns for rows 8 to 11.



#print wine.loc[8:11, :]


# So, loc provides functionality for a very specific and precise selection method.
# 
# Pandas has tools for purely position-based selection of rows and columns using the iloc attribute, which works similar to how loc works. The iloc indexer also takes input as [row, column], but takes only integer input. If we wanted to access the 60th row and the model value from auto_mpg, it would look like this (remember that integer indexing is 0-based):



#print auto_mpg.iloc[60, 6]


# To grab rows 60-63 and the last three columns from the auto_mpg DataFrame, we would need to do the following:



#print auto_mpg.iloc[60:64, 6:9]


# And to grab all values and those last three columns from the auto_mpg DataFrame:



#print auto_mpg.iloc[:, 6:9]


# One of my favorite methods for selecting data is through boolean indexing. Boolean indexing is similar to the WHERE clause in SQL in that it allows you to filter out data based on certain criteria. Let's see how this works.
# 
# Let's select from the wine DataFrame where wine_type is type 1.



#print wine[wine['wine_type'] == 1]


# This works with any comparison operators, like >, < >=, !=, and so on. For example, we can select everything from the wine DataFrame where the value in the magnesium column is less than 100.



#print wine[wine['magnesium'] < 100]


# You can also say 'not' with the tilde: ~
# 
# Let's select from the wine DataFrame where magnesium is NOT less than 100, which is equivalent to saying greater than or equal to.



#print wine[~wine['magnesium'] < 100]


# It's also possible to combine these boolean indexers. Make sure you enclose them in parentheses. This is something I usually forget.
# 
# Let's select from wine where magnesium is less than 100 and the type of wine is type 1.



#print wine[(wine['magnesium'] < 100) & (wine['wine_type'] == 1)]


# If you wanted to, you could just keep on chaining the booleans together. Let's add on where the abv is greater than 14.



#print wine[(wine['magnesium'] < 100) & (wine['wine_type'] == 1) & (wine['abv'] > 14)]


# Another method of selecting data is using the isin() function. If you pass in a list to isin(), it will return a DataFrame of booleans. True means that the value at that index is in the list you passed into isin().
# 
# Let's take the first five rows of the auto_mpg DataFrame and check for certain values existing in the DataFrame.



#print auto_mpg_5 = auto_mpg.head()

vals = [8, 150, 12.0, 'ford torino']
#print auto_mpg_5.isin(vals)


# If it says 'True', it means that one of the values from the vals list occurs there.
# 
# ## Lesson: let's try some of these on some data!



# Extract column C from the credit_approval dataframe we read in above




# Slice rows 5-10 from the credit_approval dataframe




# How would you look up the value for the 13th row in column C by label (loc)?




# How would you look up the same thing by position (iloc)?




# What if I wanted to select all data from credit_approval based on column C being greater than 5?




# What if I wanted to select data based on column C being greater than 5 and column F being equal to 'w'?




# What if I wanted to look at a boolean DataFrame of where values are in ['t', 's', 100, 0] in credit_approval?


# ## Groupby
# 
# Groupby is just like SQL's 'group by' clause. What groupby does is a three-step process:
# 
# - Split the data
# - Apply a function to the split groups
# - Recombine the data
# 
# In the apply step, you can do things like apply a statistical function, filter out data, or transform the data.
# 
# Let's groupby the wine_type in our wine DataFrame! Let's start with just groupby(), and then build it from there. This will produce a DataFrame groupby object.



#print wine.groupby('wine_type')


# Not so interesting yet. This object has some attributes you can access. We can get lists of which rows are in which group by using the groups attribute:



#print wine.groupby('wine_type').groups


# The dataset was in order by wine_type to begin with, so that makes sense. To get just the keys, add the .keys() function to the end of that line.



#print wine.groupby('wine_type').groups.keys()


# Let's group our auto_mpg dataset by cylinders, just for contrast.



#print auto_mpg.groupby('cylinders').groups


# You can see we have four observations with three cylinders, many more with four, and so on.
# 
# Going back to the wine example, let's apply an aggregate function. Let's generate the mean of all the other values and group them by wine_class.



#print wine.groupby('wine_type').mean()


# So, the mean abv for wine with type 1 is 13.74, type 2 is 12.27, type 3 is 13.15. The mean malic_acid for wine with type 1 is 2.01, and so on. So, with one line of code, we're able to apply a function to the entire dataset and see what's going on within different groups.
# 
# Selecting from a groupby DataFrame works the same way as selecting from any other DataFrame. Let's select the abv where wine_type is 2.



wine_type_mean = wine.groupby('wine_type').mean()

#print wine_type_mean.loc[2, 'abv']


# It's also possible to apply multiple functions to the entire DataFrame using the agg() function. Let's get not only the mean, but the count and the standard deviation as well for each value in the DataFrame, still grouping by wine_type.



#print wine.groupby('wine_type').agg(['mean', 'count', 'std'])


# It's also possible to run different functions on different columns. Let's get the mean for abv, the standard deviation for ash, and the sum of the values for hue. To do this, you'll need to create a dictionary with these functions, with the column names as the dictionary keys.



multiple_funcs = {'abv': mean, 'ash': std, 'hue': sum}

#print wine.groupby('wine_type').agg(multiple_funcs)


# ## Lesson: Groupby galore
# 
# Let's take this one step at a time.



# Let's group credit_approval by column G. Uncomment and fill in.
#credit_approval_group = write the code here




# Can you generate a list of all of the groups in the groupby object we just made?




# Let's use mean() on credit_approval_group to get the mean of our numeric values.




# Let's see both the standard deviation and the sum of everything in credit_approval_group




# Let's see the count on column H, the sum on column C, and the mean on column N.


# ## Merge/join; or, how Pandas can be like SQL
# 
# In Pandas, it's possible to combine DataFrames and Series much like you would in SQL. For the examples in this section, we'll work with smaller DataFrames rather than our datasets. It's easier to provide proof of concept this way, as well as explain what's going on
# 
# Let's start by appending a row to a DataFrame. We can do that by passing in a dictionary to the append function, and setting ignore_index equal to True.



data = pd.DataFrame({'col1': ['i', 'love', 'pandas', 'so', 'much'],
        'col2': ['so', 'will', 'you', 'i', 'promise']})
data.append({'col1': 'dude', 'col2': 'dude'}, ignore_index=True)


# Appending a column is also easy. You can do that by setting a new column name equal to a list or a Series.



data['col3'] = ['how', 'do', 'you', 'like', 'oscon']
#print data


# However, this will not work if your new column in a different length than the original DataFrame. Uncomment this and run it, and you'll get an error here.



#data['col4'] = ['I', 'am', 'too', 'short']
#print data


# ### Merge
# 
# You can merge() in different ways, just like joining in SQL. Let's look at an imaginary taco dataset:



tacos = pd.read_csv('../data/tacos.csv')
#print tacos


# Let's also look at an imaginary taco toppings dataset:



taco_toppings = pd.read_csv('../data/taco_toppings.csv')
#print taco_toppings


# Notice that we have a unique identifier in each dataset: the name column. We have the same five people. Let's merge these DataFrames together. You don't even need to pass the key to merge; merge() will automatically infer which key to use based on if it exists in both DataFrames. 


#print pd.merge(tacos, taco_toppings)


# By default, merge() performs a left outer join, which means it takes the key from the "left" DataFrame - the DataFrame that is passed in as the first parameter - and matches the right to it.
# 
# Generally speaking, full outer joins will join everything as a union, meaning that everything will be joined even if there are missing values; inner joins will join everything as an intersection, meaning that if a value does not appear in a row in a DataFrame, that row will be left out.
# 
# Let's look at a couple of other ways of merging. First, let's append a row to our tacos DataFrame.



tacos = tacos.append({'name': 'Luka', 'restaurant': 'Tres Carnes', 'number_of_tacos': 7, 'score': 3.8}, ignore_index=True)
#print tacos


# Now, let's do a full outer merge.



#print pd.merge(tacos, taco_toppings, how='outer')


# You can see that the entire tacos DataFrame has been merged, even though 'Luka' does not exist in the taco_toppings DataFrame.
# 
# However, if we do the same thing and use a right outer join, we'll only use the keys from the taco_toppings DataFrame and Luka will be left out.



#print pd.merge(tacos, taco_toppings, how='right')


# ### Join
# 
# The join() function gives you a way way to combine DataFrames without needing a key. Taco_extra, which contains data about chips and spiciness level, has no name column.



taco_extra = pd.read_csv('../data/taco_extra.csv')
#print taco_extra


# It's easy to join this to our taco DataFrame.



#print tacos.join(taco_extra)


# You can also specify how to join. The default is outer, but we can change it to inner and Luka will be left out again.



#print tacos.join(taco_extra, how='inner')


# It's possible to join more than two DataFrames at a time. Let's slice off the name column from taco_toppings.



taco_toppings_noname = taco_toppings.iloc[:, 1:]

#print taco_toppings_noname


# Joining this frame with tacos and taco_extra is as easy as chaining two joins together. Again, it's all an outer join, so even though there's no toppings or extra data for Luka, he's still included in the DataFrame.



#print tacos.join(taco_toppings_noname).join(taco_extra)


# ## Lesson: Let's merge some dataframes!



# Can you merge following DataFrames together?

pizza = pd.read_csv('../data/pizza.csv')
pizza_toppings = pd.read_csv('../data/pizza_toppings.csv')





# Let's inner merge those DataFrames





# Let's join pizza to pizza_extra, read in below

pizza_extra = pd.read_csv('../data/pizza_extra.csv')





# Let's only join them together where all the data is present





# Can you join all three dataframes together, first by merging pizza and pizza_toppings, then joining that to pizza_extra?


# ## Pivoting
# 
# You can pivot in Pandas just like you would in Excel. pivot_table() takes in four requires parameters: the DataFrame, the column to use for the index, the column to use for the columns, and the column to use for the values. pivot_table() also has an aggfunc parameter that defaults to the mean of the values, but you can pass in other functions, just as we did in the agg() function before.
# 
# Let's look at the mean weight per model number and number of cylinders combination.



#print pd.pivot_table(auto_mpg, values='weight', rows='model', cols='cylinders')


# If a cell contains NaN, it means that that combination doesn't exist within the DataFrame.
# 
# We can pass in multiple column names to the rows and cols parameters. This creates a multiindex.
# 
# If we add the origin column to our pivot table, we can look at the average weight of all of the model/origin combinations against the number of cylinders the cars have.



#print pd.pivot_table(auto_mpg, values='weight', rows=['model', 'origin'], cols='cylinders')


# You can apply different aggregate functions to a pivot table. Let's look at the total weight per model/cylinder combination.



#print pd.pivot_table(auto_mpg, values='weight', rows='model', cols='cylinders', aggfunc='sum')


# ## Lesson: let's pivot!



# Create a pivot_table for credit_approval with column A as the rows, column P as the cols, and column H as the values.





# Now, change the aggfunc to the standard deviation.





# Finally, can you come up with your own pivot_table?



# # For those using IPython Notebook/Wakari/NBViewer: Go to the data_analysis notebook!
# 
# # For those using code files, go to data_analysis.py!
