
#Ashwin Goyal


#%%

#building a regression classifier

#1

import numpy as np

def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets


#2

import numpy as np
from sklearn.tree import DecisionTreeRegressor

# copy in your get_features_targets function here
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# load the data and generate the features and targets
data = np.load('sdss_galaxy_colors.npy')
features, targets = get_features_targets(data)
  
# initialize model
dtr = DecisionTreeRegressor()
# train the model
dtr.fit(features, targets)
# make predictions using the same features
predictions = dtr.predict(features)
# print out the first 4 predicted redshifts
print(predictions[:4])



#3

import numpy as np

# write a function that calculates the median of the differences
# between our predicted and actual values
def median_diff(predicted, actual):
  arr = np.abs(predicted - actual)
  val = np.median(arr)
  return val


#4


import numpy as np
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  arr = np.abs(predicted - actual)
  val = np.median(arr)
  return val


# write a function that splits the data into training and testing subsets
def split_data(features,targets):
  split = features.shape[0]//2
  train_features = features[:split]
  test_features = features[split:]
  split = targets.shape[0]//2
  train_targets = targets[:split]
  test_targets = targets[split:]
  return train_features,test_features,train_targets,test_targets

# trains the model and returns the prediction accuracy with median_diff
def validate_model(model, features, targets):
  # split the data into training and testing features and predictions
  train_features,test_features,train_targets,test_targets = split_data(features,targets)
  
  # train the model
  model.fit(train_features,train_targets)
  
  # get the predicted_redshifts
  predictions = model.predict(test_features)
  # use median_diff function to calculate the accuracy
  return median_diff(test_targets, predictions)



#5
import numpy as np
from matplotlib import pyplot as plt

# Complete the following to make the plot
if __name__ == "__main__":
    data = np.load('sdss_galaxy_colors.npy')
    # Get a colour map
    cmap = plt.get_cmap('YlOrRd')

    # Define our colour indexes u-g and r-i
    ug = data['u'] - data['g']
    ri = data['r'] - data['i']
    # Make a redshift array
    redshift = data['redshift']
    # Create the plot with plt.scatter and plt.colorbar
    plt.scatter(ug,ri,cmap = cmap,s = 1, c = redshift)
    # Define your axis labels and plot title
    plt.xlabel('Colour index u-g')
    plt.ylabel('Colour index r-i')
    # Set any axis limits
    plt.xlim(-0.5,2.5)
    plt.ylim(-0.5,1)
    plt.colorbar()
    plt.show()



#%%

#improving and evaluating our classifier

#1

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  arr = np.abs(predicted - actual)
  val = np.median(arr)
  return val


# write a function that splits the data into training and testing subsets
def split_data(features,targets):
  split = features.shape[0]//2
  train_features = features[:split]
  test_features = features[split:]
  split = targets.shape[0]//2
  train_targets = targets[:split]
  test_targets = targets[split:]
  return train_features,test_features,train_targets,test_targets

# Complete the following function
def accuracy_by_treedepth(features, targets, depths):
  # split the data into testing and training sets
  train_features,test_features,train_targets,test_targets = split_data(features,targets)
  # initialise arrays or lists to store the accuracies for the below loop
  train_acc = []
  test_acc = []
  # loop through depths
  for depth in depths:
    # initialize model with the maximum depth. 
    dtr = DecisionTreeRegressor(max_depth=depth)
    dtr.fit(train_features,train_targets)
    # train the model using the training set
    train_predictions = dtr.predict(train_features)
    train_med = median_diff(train_predictions,train_targets)
    train_acc.append(train_med)
    # get the predictions for the training set and calculate their median_diff
    test_predictions = dtr.predict(test_features)
    test_med = median_diff(test_predictions,test_targets)
    test_acc.append(test_med)
    # get the predictions for the testing set and calculate their median_diff
  return train_acc,test_acc
  # return the accuracies for the training and testing sets

#2

import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  arr = np.abs(predicted - actual)
  val = np.median(arr)
  return val



# complete this function
def cross_validate_model(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # initialise a list to collect median_diffs for each iteration of the loop below
  l = []
  for train_indices, test_indices in kf.split(features):
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    dtr = DecisionTreeRegressor(max_depth=19)
    dtr.fit(train_features,train_targets)
    # fit the model for the current set
    
    # predict using the model
    predictions = dtr.predict(test_features)
    # calculate the median_diff from predicted values and append to results array
    med_diff = median_diff(predictions,test_targets)
    l.append(med_diff)

  return l
  # return the list with your median difference values




#3

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here
def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  arr = np.abs(predicted - actual)
  val = np.median(arr)
  return val


# complete this function
def cross_validate_predictions(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # declare an array for predicted redshifts from each iteration
  all_predictions = np.zeros_like(targets)

  for train_indices, test_indices in kf.split(features):
    # split the data into training and testing
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    dtr = DecisionTreeRegressor(max_depth=19)
    dtr.fit(train_features,train_targets)
    # fit the model for the current set
        
    # predict using the model
    predictions = dtr.predict(test_features)
    # put the predicted values in the all_predictions array defined above
    all_predictions[test_indices] = predictions

  # return the predictions
  return all_predictions    



#4


import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor

# paste your get_features_targets function here

def get_features_targets(data):
  features = np.zeros((data.shape[0], 4))
  features[:, 0] = data['u'] - data['g']
  features[:, 1] = data['g'] - data['r']
  features[:, 2] = data['r'] - data['i']
  features[:, 3] = data['i'] - data['z']
  targets = data['redshift']
  return features, targets

# paste your median_diff function here
def median_diff(predicted, actual):
  arr = np.abs(predicted - actual)
  val = np.median(arr)
  return val


# complete this function
def cross_validate_model(model, features, targets, k):
  kf = KFold(n_splits=k, shuffle=True)

  # initialise a list to collect median_diffs for each iteration of the loop below
  l = []
  for train_indices, test_indices in kf.split(features):
    train_features, test_features = features[train_indices], features[test_indices]
    train_targets, test_targets = targets[train_indices], targets[test_indices]
    
    dtr = DecisionTreeRegressor(max_depth=19)
    dtr.fit(train_features,train_targets)
    # fit the model for the current set
    
    # predict using the model
    predictions = dtr.predict(test_features)
    # calculate the median_diff from predicted values and append to results array
    med_diff = median_diff(predictions,test_targets)
    l.append(med_diff)

  return l   


# complete this function
def split_galaxies_qsos(data):
  # split the data into galaxies and qsos arrays
  gal = data[data['spec_class'] == b'GALAXY']
  QSO = data[data['spec_class'] == b'QSO']
  # return the seperated galaxies and qsos arrays
  return gal,QSO



def cross_validate_median_diff(data):
  features, targets = get_features_targets(data)
  dtr = DecisionTreeRegressor(max_depth=19)
  return np.mean(cross_validate_model(dtr, features, targets, 10))








#%%

