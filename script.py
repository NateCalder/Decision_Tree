import codecademylib3_seaborn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

flags = pd.read_csv('flags.csv', header=0)
print(flags.head())

#Create decision tree that determines where a country is based only on the colors of its flag
labels = flags[['Landmass']]
#data = flags[['Red', 'Green', 'Blue', 'Gold', 'White', 'Black', 'Orange']]

#changing the labels to include more information which better effects accuracy performance
data = flags[["Red", "Green", "Blue", "Gold",
 "White", "Black", "Orange",
 "Circles",
"Crosses","Saltires","Quarters","Sunstars",
"Crescent","Triangle"]]


#Splitting into training and test sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, random_state = 1)

#Create DT classifier
tree = DecisionTreeClassifier(random_state = 1)
tree.fit(train_data, train_labels)

#Finding the score of the model - 34.69% correct
#print(tree.score(test_data, test_labels))

#Inspecting other possible tree depths for better accuracy
scores = []
for i in range(1, 21):
  tree = DecisionTreeClassifier(random_state = 1, max_depth = i)
  tree.fit(train_data, train_labels)
  scores.append(tree.score(test_data, test_labels))

#Graph shows, with more labels, depths 5 and 6 convey the highest accuracy (55.0%)
plt.plot(range(1, 21), scores)
plt.show()

