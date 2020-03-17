# %% markdown
# # Titanic Challenge - Kaggle
# %%
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
# Importing the Titanic Train dataset
titanic_df = pd.read_csv('train.csv')
# %%
titanic_df.head()
# %%
# In order to see the info
titanic_df.info()
# %%
# Basic questions to answer:
# 1. Who were the passengers? (ages,class,gender...)
# 2. What deck the passengers were and how that relates to their class
# 3. Where did the passengers come from?
# 4. Who was alone and who wasn't?
# 5. What factors helped someone survive the sinking?
# %%
%matplotlib inline
# %%
# gender check up to see the passengers on Titanic
sns.catplot('Sex', data=titanic_df, kind='count')
# %%
# Separating the gender by classes

sns.countplot('Pclass', data = titanic_df, hue = 'Sex')
# %%
# Here we can see that there is way more males in the first class than females.
# What about the children?
# Building a function to visualize male, female, and children

def male_female_child(passenger):
    age,sex = passenger

    if age < 16:
        return 'child'
    else:
        return sex
# %%
# Now that the function was created, it should be applied.
# I'm creating a new column called 'person' and applying the male_female_child function on it

titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis=1)
# %%
titanic_df[0:10]
# %%
# Now that the column 'person' was created, we can see how many male, female, and children there is in each class

sns.countplot('Pclass', data = titanic_df, hue = 'person')
# %%
# In order to have a more precise distribution of the passengers' age, we can build a Histogram

titanic_df['Age'].hist(bins = 70)
# %%
# The mean age is:

titanic_df['Age'].mean()
# %%
# Now the total count of male, female, and children

titanic_df['person'].value_counts()
# %%
# Another way to visualize that data is by using a facet visualization and to map for kde plots in order to generate multiple plots

fig = sns.FacetGrid(titanic_df, hue = 'Sex', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
oldest = titanic_df['Age'].max()
fig.set(xlim = (0,oldest))
fig.add_legend()
# %%
# To visualize the children:

fig = sns.FacetGrid(titanic_df, hue = 'person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)

oldest = titanic_df['Age'].max()

fig.set(xlim = (0,oldest))

fig.add_legend()
# %%
# To see it by class

fig = sns.FacetGrid(titanic_df, hue = 'Pclass', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)

oldest = titanic_df['Age'].max()

fig.set(xlim = (0,oldest))

fig.add_legend()
# %%
# Now we can have some idea of who the passengers were.
# It's time to answer the second question
# Taking a look at the dataset again:

titanic_df.head()
# %%
# There are a lot of missing values in the Cabin column.
# Now I'm dropping all NAs from the Cabin column

deck = titanic_df['Cabin'].dropna()
# %%
deck.head()
# %%
# As we can see, we have the letter for the deck level, and the number for the cabin number.
# We only need the letter to classify the deck, so we are grabbing only the letter

levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']
sns.countplot('Cabin', data = cabin_df, palette = 'winter_d')
# %%
# Getting rid of the T column (it's a value in the end that doesn't make sense)

cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.countplot('Cabin', data = cabin_df, palette = 'summer', order = ['A','B','C','D','E','F','G'])
# %%
titanic_df.head()
# %%
# Where did the passengers come from?

sns.countplot('Embarked', data = titanic_df, hue = 'Pclass', order = ['C','Q','S'])
# %%
# We can notice that in Queen's Town, most of the passengers that were born there are third class
# What we may think about it is: what are the economics of the town in that period?
# what are the economics of the 'C' city where we have most first class?
# %%
# Who was alone and who was with family?

titanic_df.head()
# %%
# Let's define 'alone'
# %%
# If 'SibSp' and 'Parch' columns (siblings and parents) are 0, it means they are alone

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch
titanic_df['Alone']
# %%
# If the alone column is greater than zero, it means they have some sort of family onboard

# %%
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'
# %%
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
# %%
titanic_df.head()
# %%
# Visualizing this:

sns.countplot('Alone', data = titanic_df, palette = 'Blues')
# %%
# It seems more people were alone than with family

# %%
# What factors help people survive the sinking?
# %%
titanic_df['Survivor'] = titanic_df.Survived.map({0: 'no', 1: 'yes'})

sns.countplot('Survivor', data = titanic_df, palette = 'Set1')
# %%
# more people did not survive
# %%
# Let's see if the class was a factor (the movie show that third class passengers did not as well as first and second class ones)

sns.catplot('Pclass', 'Survived', data = titanic_df, kind = 'point')
# %%
# Women and children were the first to go, so if the third class was composed mostly by male, as we saw before,
# it makes sense that it was affected more than the other classes
# %%
# Now let's see if gender and class together can tell us something

sns.catplot('Pclass', 'Survived', data = titanic_df, kind = 'point', hue = 'person')
# %%
# It looks like being male and being on the third class, it's not favorable for survival
# As a matter of fact, being a male, regardless of the class, is not favorable for survival
# %%
# Is age also a factor?

sns.lmplot('Age', "Survived", titanic_df)
# %%
# The trend line shows that the older the passenger, the less likely to survive
# %%
# Let's see the trend among the classes

sns.lmplot('Age', "Survived", titanic_df, hue = 'Pclass', palette = 'winter')
# %%
# Getting the figure a little more clean by binnig by age

generations = [10,20,40,60,80]

sns.lmplot('Age', 'Survived', hue = 'Pclass', data = titanic_df, palette = 'winter', x_bins = generations)
# %%
# Let's check how gender and age relate to survival

sns.lmplot('Age', 'Survived', hue = 'Sex', data = titanic_df, palette = 'winter', x_bins = generations)
# %%
# It's interesting. It looks like if you were an old female you have more chances to survive as if you were an old male
# However, the old male has quite a bit of standard deviation on it, so it would be interesting to have a closer look on that
