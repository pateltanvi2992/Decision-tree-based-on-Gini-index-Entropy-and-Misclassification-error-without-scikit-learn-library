#Importing libraries
import numpy as np
import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pandas.core import window

import argparse
from argparse import ArgumentParser


# create argument input 
parser = argparse.ArgumentParser(description="dataset Path")
parser.add_argument("--dataset", type=str, default=None, help="dataset path")
#Wine_dataset = pd.read_csv("/data/patelt6/School_files/winequality-red.csv")
args = parser.parse_args()
Wine_dataset = pd.read_csv(args.dataset)
#Read dataset
#data_path = "/content/WineQT.csv"
Wine_dataset = pd.read_csv(args.dataset)

#SneakPeak first 5 rows of the dataset
Wine_dataset.head()

##Description of table
len(Wine_dataset)
Wine_dataset.shape
Wine_dataset.info()

#data cleaning: remove unecessary column
del Wine_dataset ["Id"]
#SneakPeak first 5 rows of the reduced column
Wine_dataset.head(n=5)

plt.figure(figsize=(10,10))
sns.heatmap(Wine_dataset.corr(),annot=True,linewidth=0.5,center=0,cmap='GnBu')
plt.show()
plt.savefig("Heatmap_wine_quality_dataset.png")

# STEP 1: Calculate gini(D) for all the given classes
def gini_impurity (value_counts):
    n = value_counts.sum()
    sum = 0
    for key in value_counts.keys():
        sum = sum  +  (value_counts[key] / n ) * (value_counts[key] / n ) 
    gini = 1 - sum
    return gini

class_value_counts = Wine_dataset['quality'].value_counts()
print(f'Number of samples in each class is:\n{class_value_counts}')

gini_class = gini_impurity(class_value_counts)
print(f'\nGini Impurity of the class is {gini_class:.3f}')

# STEP 2: 
# Calculating  gini impurity for all the attiributes
def gini_split_a(attribute_name):
    attribute_values = Wine_dataset[attribute_name].value_counts()
    gini_A = 0 
    for key in attribute_values.keys():
        Wine_dataset_k = Wine_dataset['quality'][Wine_dataset[attribute_name] == key].value_counts()
        n_k = attribute_values[key]
        n = Wine_dataset.shape[0]
        gini_A = gini_A + (( n_k / n) * gini_impurity(Wine_dataset_k))
    return gini_A

gini_attiribute ={}
for key in Wine_dataset:
    gini_attiribute[key] = gini_split_a(key)
    print(f'Gini for {key} is {gini_attiribute[key]:.3f}')

# STEP 3: 
# Compute Gini gain values to find the best split
# An attribute has maximum Gini gain is selected for splitting.

min_value = min(gini_attiribute.values())
max_value = max(gini_attiribute.values())
print('The minimum value of Gini Impurity : {0:.3} '.format(min_value))
print('The maximum value of Gini Gain     : {0:.3} '.format(max_value))

selected_attribute = min(gini_attiribute.keys())
print('The selected attiribute is: ', selected_attribute)

Wine_dataset.loc[:, 'alcohol':'quality'].groupby(by='quality').describe()

#Don't run this part if you don't want violin plots for all the features - data distribution from the dataset. 
"""
ax = sns.violinplot(data=Wine_dataset,palette="Blues")
ax.set_title('alcohol', fontsize=16);
fig = ax.get_figure()
fig.savefig('alcohol.png', dpi = 600) 

ax = sns.violinplot(y="fixed acidity", data=Wine_dataset, palette="Blues")
ax.set_title('fixed acidity', fontsize=16);
fig = ax.get_figure()
fig.savefig('fixed_acidity.png', dpi = 600)

ax = sns.violinplot(y="volatile acidity", data=Wine_dataset, palette="Blues")
ax.set_title('volatile acidity', fontsize=16);
fig = ax.get_figure()
fig.savefig('volatine_acidity.png', dpi = 600)

ax = sns.violinplot(y="citric acid", data=Wine_dataset,palette="Blues")
ax.set_title('citric acid', fontsize=16);
fig = ax.get_figure()
fig.savefig('citric_acid.png', dpi = 600) 

ax = sns.violinplot(y="residual sugar", data=Wine_dataset, palette="Blues")
ax.set_title('residual sugar', fontsize=16);
fig = ax.get_figure()
fig.savefig('residual.png', dpi = 600) 

ax = sns.violinplot(y="chlorides", data=Wine_dataset, palette="Blues")
ax.set_title('chlorides', fontsize=16);
fig = ax.get_figure()
fig.savefig('chlorides.png', dpi = 600) 

ax = sns.violinplot(y="free sulfur dioxide", data=Wine_dataset,palette="Blues")
ax.set_title('free sulfur dioxide', fontsize=16);
fig = ax.get_figure()
fig.savefig('free_sulfur_dioxide.png', dpi = 600) 

ax = sns.violinplot(y="total sulfur dioxide", data=Wine_dataset,palette="Blues")
ax.set_title('total sulfur dioxide', fontsize=16);
fig = ax.get_figure()
fig.savefig('total_sulfur.png', dpi = 600) 

ax = sns.violinplot(y="density", data=Wine_dataset,palette="Blues")
ax.set_title('density', fontsize=16);
fig = ax.get_figure()
fig.savefig('density.png', dpi = 600)  

ax = sns.violinplot(y="pH", data=Wine_dataset,palette="Blues")
ax.set_title('pH', fontsize=16);
fig = ax.get_figure()
fig.savefig('pH.png', dpi = 600) 

ax = sns.violinplot(y="sulphates", data=Wine_dataset,palette="Blues")
ax.set_title('sulphates', fontsize=16);
fig = ax.get_figure()
fig.savefig('sulphates.png', dpi = 600) 

ax = sns.violinplot(y="quality", data=Wine_dataset,palette="Blues")
ax.set_title('quality', fontsize=16);
fig = ax.get_figure()
fig.savefig('quality.png', dpi = 600)
"""

#Defining a function for the tree node
class TreeNode:
    def __init__(self, data,output):
        # data represents the feature upon which the node was split when fitting the training data
        # data = None for leaf node
        self.data = data
        # children of a node are stored as a dicticionary with key being the value of feature upon which the node was split
        # and the corresponding value stores the child TreeNode
        self.children = {}
        # output represents the class with current majority at this instance of the decision tree
        self.output = output
        # index will be used to assign a unique index to each node
        self.index = -1
        
    def add_child(self,feature_value,obj):
        self.children[feature_value] = obj

class DecisionTreeClassifier:
    def __init__(self):
        # root represents the root node of the decision tree built after fitting the training data
        self.__root = None

    def count_unique_wine_quality(self,Y):
        # returns a dictionary with keys as unique values of Y(i.e no of classes) and the corresponding value as its frequency
        d = {}
        for i in Y:
            if i not in d:
                d[i]=1
            else:
                d[i]+=1
        return d

    #**************************************************************************
    #Entropy of whole dataset
    #*************************************************************************
    def __entropy(self,Y):
        # returns the entropy 
        freq_map = self.count_unique_wine_quality(Y)
        entropy_ = 0
        total = len(Y) # 1143
        for i in freq_map:
            p = freq_map[i]/total
            entropy_ += (-p)*math.log2(p)
        file1 = open("/content/entropy.txt", "w")  # write mode
        file1.write(str(entropy_))
        file1.close()
        return entropy_
    #**************************************************************************
    #gain ratio of whole dataset
    #*************************************************************************
    def __gain_ratio(self,X,Y,selected_feature):
        # returns the gain ratio
        info_orig = self.__entropy(Y) # info_orig represents entropy before splitting
        info_f = 0  # info_f represents entropy after splitting upon the selected feature
        split_info = 0
        values = set(X[:,selected_feature])
        df = pd.DataFrame(X)
        # Adding Y values as the last column in the dataframe 
        df[df.shape[1]] = Y
        initial_size = df.shape[0] 
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            info_f += (current_size/initial_size)*self.__entropy(df1[df1.shape[1]-1])
            split_info += (-current_size/initial_size)*math.log2(current_size/initial_size)

        # to handle the case when split info = 0 which leads to division by 0 error
        if split_info == 0 :
            return math.inf

        info_gain = info_orig - info_f
        gain_ratio = info_gain / split_info
        return gain_ratio

#**************************************************************************
#Gini Index of whole dataset
#*************************************************************************

    def __gini_index(self,Y):
        # returns the gini index 
        freq_map = self.count_unique_wine_quality(Y)
        gini_index_ = 1
        total = len(Y)
        for i in freq_map:
            p = freq_map[i]/total
            gini_index_ -= p**2 #loop( 1 - (p**2))
        return gini_index_

    def __gini_gain(self,X,Y,selected_feature):
        # returns the gini gain
        gini_orig = self.__gini_index(Y) # gini_orig represents gini index before splitting
        gini_split_f = 0 # gini_split_f represents gini index after splitting upon the selected feature
        values = set(X[:,selected_feature])
        df = pd.DataFrame(X)
        # Adding Y values as the last column in the dataframe 
        df[df.shape[1]] = Y
        initial_size = df.shape[0] 
        for i in values:
            df1 = df[df[selected_feature] == i]
            current_size = df1.shape[0]
            gini_split_f += (current_size/initial_size)*self.__gini_index(df1[df1.shape[1]-1])

        gini_gain_ = gini_orig - gini_split_f
        return gini_gain_
    
    def __mis_classification(self, X, Y, selected_feature):
      # returns the gini index 
        freq_map = self.count_unique_wine_quality(Y)
        gini_index_ = 1
        total = len(Y)
        for i in freq_map:
            p = freq_map[i]/total
            mis_classification_ += (1 - max(p))
        return mis_classification_

    def __decision_tree(self,X,Y,features,level,metric,classes):
        # returns the root of the Decision Tree(which consists of TreeNodes) built after fitting the training data
        # Here Nodes are printed as in PREORDER traversl
        # classes represents the different classes present in the classification problem 
        # metric can take value gain_ratio or gini_index
        # level represents depth of the tree
        # We split a node on a particular feature only once (in a given root to leaf node path)
        
        
        # If the node consists of only 1 class
        if len(set(Y)) == 1:
            print("Level",level)
            output = None
            for i in classes:
                if i in Y:
                    output = i
                    print("Count of",i,"=",len(Y))
                else :
                    print("Count of",i,"=",0)
            if metric == "gain_ratio":
                print("Current Entropy is =  0.0")
            elif metric == "gini_index":
                print("Current Gini Index is =  0.0")

            print("Reached leaf Node")
            print()
            return TreeNode(None,output)

        # If we have run out of features to split upon
        # In this case we will output the class with maximum count
        if len(features) == 0:
            print("Level",level)
            freq_map = self.count_unique_wine_quality(Y)
            output = None
            max_count = -math.inf
            for i in classes:
                if i not in freq_map:
                    print("Count of",i,"=",0)
                else :
                    if freq_map[i] > max_count :
                        output = i
                        max_count = freq_map[i]
                    print("Count of",i,"=",freq_map[i])
            with open('sample.txt', 'w') as f:
              if metric == "gain_ratio":
                  print("Current Entropy  is =",self.__entropy(Y))
              elif metric == "gini_index":
                  print("Current Gini Index is =",self.__gini_index(Y))           

            print("Reached leaf Node")
            print()
            return TreeNode(None,output)

        
        # Finding the best feature to split upon
        max_gain = -math.inf
        final_feature = None
        for f in features :
            if metric == "gain_ratio":
                current_gain = self.__gain_ratio(X,Y,f)
            elif metric =="gini_index":
                current_gain = self.__gini_gain(X,Y,f)

            if current_gain > max_gain:
                max_gain = current_gain
                final_feature = f

        print("Level",level)
        freq_map = self.count_unique_wine_quality(Y)
        output = None
        max_count = -math.inf

        for i in classes:
            if i not in freq_map:
                print("Count of",i,"=",0)
            else :
                if freq_map[i] > max_count :
                    output = i
                    max_count = freq_map[i]
                print("Count of",i,"=",freq_map[i])

        if metric == "gain_ratio" :        
            print("Current Entropy is =",self.__entropy(Y))
            print("Splitting on feature  X[",final_feature,"] with gain ratio ",max_gain,sep="")
            print()
        elif metric == "gini_index":
            print("Current Gini Index is =",self.__gini_index(Y))
            print("Splitting on feature  X[",final_feature,"] with gini gain ",max_gain,sep="")
            print()

            
        unique_values = set(X[:,final_feature]) # unique_values represents the unique values of the feature selected
        df = pd.DataFrame(X)
        # Adding Y values as the last column in the dataframe
        df[df.shape[1]] = Y

        current_node = TreeNode(final_feature,output)

        # Now removing the selected feature from the list as we do not want to split on one feature more than once(in a given root to leaf node path)
        index  = features.index(final_feature)
        features.remove(final_feature)
        for i in unique_values:
            # Creating a new dataframe with value of selected feature = i
            df1 = df[df[final_feature] == i]
            # Segregating the X and Y values and recursively calling on the splits
            node = self.__decision_tree(df1.iloc[:,0:df1.shape[1]-1].values,df1.iloc[:,df1.shape[1]-1].values,features,level+1,metric,classes)
            current_node.add_child(i,node)

        # Add the removed feature     
        features.insert(index,final_feature)

        return current_node
    
    def fit(self,X,Y,metric="gain_ratio"):
        # Fits to the given training data
        # metric can take value gain_ratio or gini_index
        features = [i for i in range(len(X[0]))]
        classes = set(Y)
        level = 0
        if metric != "gain_ratio" :
            if metric != "gini_index":
                metric="gain_ratio"  # if user entered a value which was neither gini_index nor gain_ratio
        self.__root = self.__decision_tree(X,Y,features,level,metric,classes)
        
    def __predict_for(self,data,node):
        # predicts the class for a given testing point and returns the answer
        
        # We have reached a leaf node
        if len(node.children) == 0 :
            return node.output

        val = data[node.data] # represents the value of feature on which the split was made 
        print(val)      
        if val not in node.children :
            return node.output
        
        # Recursively call on the splits
        return self.__predict_for(data,node.children[val])
    def prune_tree(self, node, dataset, best_score):
      # if node is a leaf
      if len(node.children) == 0:
          # get its classification
          classification = node.classification
          # run validate_tree on a tree with the nodes parent as a leaf with its classification
          self.root = True
          node.parent.classification = node.classification
          if (node.height < 10):
              new_score = validate_tree(root, dataset)
          else:
              new_score = 0
    
          # if its better, change it
          if (new_score >= best_score):
              return new_score
          else:
              node.parent.is_leaf = False
              node.parent.classification = None
              return best_score
      # if its not a leaf
      else:
          # prune tree(node.upper_child)
          new_score = prune_tree(root, node.upper_child, dataset, best_score)
          # if its now a leaf, return
          if len(node.children) == 0:
              return new_score
          # prune tree(node.lower_child)
          new_score = prune_tree(root, node.lower_child, dataset, new_score)
          # if its now a leaf, return
          if len(node.children) == 0:
              return new_score

          return new_score

    def predict(self,X):
        # This function returns Y predicted
        # X should be a 2-D np array
        Y = np.array([0 for i in range(len(X))])
        for i in range(len(X)):
            Y[i] = self.__predict_for(X[i],self.__root)
        return Y
    
    def score(self,X,Y):
        # returns the mean accuracy
        Y_pred = self.predict(X)
        count = 0
        for i in range(len(Y_pred)):
            if Y_pred[i] == Y[i]:
                count+=1
        return count/len(Y_pred)

    def export_tree_pdf(self,filename=None):
        # returns the tree as dot data
        # if filename is specified the function 
        # will save the pdf file in current directory which consists of the visual reresentation of the tree
        import pydotplus
        from collections import deque
        
        dot_data = '''digraph Tree {
node [shape=box] ;'''
        
        queue = deque()
        
        r = self.__root
        queue.append(r)
        count = 0
        if r.index == -1:
            r.index = count
        
        dot_data = dot_data + "\n{} [label=\"Feature to split upon : X[{}]\\nOutput at this node : {}\" ];".format(count,r.data,r.output) 
        
        # Doing LEVEL ORDER traversal in the tree (using a queue)
        while len(queue) != 0 :
            node = queue.popleft()
            for i in node.children:
                count+=1
                if(node.children[i].index==-1):
                    node.children[i].index = count
                
                # Creating child node
                dot_data = dot_data + "\n{} [label=\"Feature to split upon : X[{}]\\nOutput at this node : {}\" ];".format(node.children[i].index,node.children[i].data,node.children[i].output) 
                # Connecting parent node with child
                dot_data = dot_data + "\n{} -> {} [ headlabel=\"Feature value = {}\"]; ".format(node.index,node.children[i].index,i)
                # Adding child node to queue
                queue.append(node.children[i])
        
        dot_data = dot_data + "\n}"

        if filename != None:    
            graph = pydotplus.graph_from_dot_data(dot_data)
            graph.write_pdf(filename)  

        file1 = open("/content/tree.txt", "w")  # write mode
        file1.write(dot_data)
        file1.close()
        
        return dot_data



#data_path = "/content/WineQT.csv"
#Wine_dataset = pd.read_csv(Wine_dataset)
X =Wine_dataset.drop(["quality"],1)
y = Wine_dataset["quality"]
#Alternate - Train & Test
#Declaring the training & test data set
n_train = math.floor(0.8 * X.shape[0])
n_test = math.ceil((1-0.2) * X.shape[0])
x_train = X[:n_train]
y_train = y[:n_train]
x_test = X[n_train:]
y_test = y[n_train:]
print("Number of training dataset:",x_train.shape[0])
print("Number of test dataset:",x_test.shape[0])

# Prunning Class
class Node:
    def __init__(self, data_df, depth=0):

        self.left = None            # Left son node
        self.right = None           # Right son node
        self.data = data_df         # Data of the node
        self.depth = depth          # The depth level of the node in the tree
        self.classification = None  # The class of the node
        self.prev_condition = None  # Condition that brings the data to the node
        self.prev_feature = None    # The splitting feature
        self.prev_thresh = None     # The splitting threshold
        self.backuperror = None     # Backuperror for post-pruning
        self.mcp = None             # Misclassification probability
    
    def set_splits(self, prev_condition, prev_feature, prev_thresh):

        self.prev_condition = prev_condition  
        self.prev_feature = prev_feature
        self.prev_thresh = prev_thresh

def calculate_entropy(data):

    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()
    entropy = sum(-probs * np.log2(probs))

    return entropy

def calculate_overall_entropy(data1, data2):

    total_num = len(data1) + len(data2)
    prob_data1 = len(data1) / total_num
    prob_data2 = len(data2) / total_num

    overall_entropy = prob_data1 * calculate_entropy(data1) + prob_data2 * calculate_entropy(data2)

    return overall_entropy

def calculate_gini(data):

    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()
    gini = 1 - sum(np.square(probs))

    return gini

def calculate_overall_gini(data1, data2):

    total_num = len(data1) + len(data2)
    prob_data1 = len(data1) / total_num
    prob_data2 = len(data2) / total_num
    

    overall_gini = prob_data1 * calculate_gini(data1) + prob_data2 * calculate_gini(data2) 
    return overall_gini

def calculate_mce(data):

    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()
    mce = 1 - np.max(probs)

    return mce

def calculate_overall_mce(data1, data2):

    total_num = len(data1) + len(data2) 
    prob_data1 = len(data1) / total_num
    prob_data2 = len(data2) / total_num
   

    overall_mce = prob_data1 * calculate_mce(data1) + prob_data2 * calculate_mce(data2)
    return overall_mce

def calculate_overall_impurity(data1, data2, method):
    
    if method == 'entropy':
        return calculate_overall_entropy(data1, data2)
    elif method == 'gini':
        return calculate_overall_gini(data1, data2)
    elif method == 'mce':
        return calculate_overall_mce(data1, data2)
    else:
        raise ValueError


def calculate_laplace_mcp(data):

    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    c = np.max(counts)
    k = counts.sum()

    mcp = (k-c+1)/(k+2)

    return mcp

def check_purity(data):

    labels = data[:,-1]
    unique_classes = np.unique(labels)

    if len(unique_classes) == 1:
        return True
    else:
        return False

def classify_data(data):

    labels = data[:,-1]
    unique_classes, count_unique_classes = np.unique(labels, return_counts=True)

    index = count_unique_classes.argmax()
    classification = unique_classes[index]
    return classification

def get_splits(data):
  
    splits = {}
    n_cols = data.shape[1]  # Number of columns
    for i_col in range(n_cols - 1): # Disregarding the last label column
        splits[i_col] = []
        values = data[:,i_col]
        unique_values = np.unique(values)   # All possible values
        for i_thresh in range(1,len(unique_values)):
            prev_value = unique_values[i_thresh - 1]
            curr_value = unique_values[i_thresh]
            splits[i_col].append((prev_value + curr_value)/2)   # Return the average of two neighbour values
    
    return splits

def split_data(data, split_index, split_thresh):

    split_column_values = data[:, split_index]

    data_below = data[split_column_values <= split_thresh]
    data_above = data[split_column_values >  split_thresh]

    return data_below, data_above

def find_best_split(data, splits, method):

    global best_index
    global best_thresh
    
    min_overall_impurity = float('inf') # Store the largest overall impurity value
    for index in splits.keys():
        for split_thresh in splits[index]:
            data_true, data_false = split_data(data=data,split_index=index, split_thresh=split_thresh)
            overall_impurity = calculate_overall_impurity(data_true, data_false, method)

            if overall_impurity <= min_overall_impurity:    # Find new minimised impurity
                min_overall_impurity = overall_impurity     # Replace the minimum impurity
                best_index = index
                best_thresh = split_thresh
    
    return best_index, best_thresh

from tabulate import tabulate

class DesicionTree:
    def __init__(self, criterion='entropy', post_prune=False, min_samples=2, max_depth=7):

        self.root = None
        self.criterion = criterion
        self.post_prune = post_prune
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.features = None

    def feed(self, data_df):

        self.root = Node(data_df, 0)
        self._fit(self.root)
    
    def _fit(self, node):

        # Prepare data
        data = node.data  # pandas DataFrame
        depth = node.depth

        if depth == 0:
            self.features = data.columns
        data = data.values  # numpy array

        # Pre-pruning
        if (check_purity(data)) or (len(data) < self.min_samples) or (depth == self.max_depth): # Stop splitting?
            classification = classify_data(data)
            node.classification = classification
        
        # Recursive
        else:   # Keep splitting
            # Splitting
            splits = get_splits(data)
            split_index, split_thresh = find_best_split(data, splits, self.criterion)
            data_left, data_right = split_data(data, split_index, split_thresh)

            # Pre-pruning: Prevent empty split
            if (data_left.size == 0) or (data_right.size == 0):
                classification = classify_data(data)
                node.classification = classification

            else:
                depth += 1  # Deeper depth
                
                # Transform the numpy array into pandas DataFrame for the node
                data_left_df = pd.DataFrame(data_left,columns=list(self.features))
                data_right_df = pd.DataFrame(data_right,columns=list(self.features))

                # Get condition description
                feature_name = self.features[split_index]
                true_condition = "{} <= {}".format(feature_name, split_thresh)
                false_condition = "{} > {}".format(feature_name, split_thresh)

                # Set values of the node
                node.left = Node(data_left_df,depth=depth)
                node.right = Node(data_right_df, depth=depth)
                node.left.set_splits(true_condition, feature_name, split_thresh)
                node.right.set_splits(false_condition, feature_name, split_thresh)

                # Recursive process
                self._fit(node.left)
                self._fit(node.right) 

                self._merge()   # Merge the son nodes with the same class

                if self.post_prune: # Post-pruning
                    self._post_prune()

    def _merge(self): 
        # First the root
        stack = []  # LIFO, Build a stack to store the Nodes
        stack.append(self.root)
        while True:
            if len(stack):
                pop_node = stack.pop()
                if pop_node.left:
                    if pop_node.left.classification:    # Already classified
                        if pop_node.left.classification == pop_node.right.classification:   # Same classification
                            pop_node.classification = pop_node.left.classification
                            pop_node.left = None
                            pop_node.right = None
                        else:   # Different classifications
                            stack.append(pop_node.right)
                            stack.append(pop_node.left)
                    else:   # Not classified
                        stack.append(pop_node.right)
                        stack.append(pop_node.left)
            else:
                break

    def _calculate_error(self, node):
        # Misclassification probability using Laplace's Law
        if node.left:   # There are son nodes, the backuperror of this node is the weighted sum of the backuperrors of sons
            backuperror_left = self._calculate_error(node.left)
            backuperror_right = self._calculate_error(node.right)
            node.backuperror = len(node.left.data)/len(node.data)*backuperror_left + len(node.right.data)/len(node.data)*backuperror_right
            node.mcp = calculate_laplace_mcp(node.data.to_numpy())  # And we still need mcp
        else:   # No son nodes, backuperror = mcp
            node.backuperror = node.mcp = calculate_laplace_mcp(node.data.to_numpy())
        return node.backuperror


    def _post_prune(self):
        self._calculate_error(self.root)
        # LIFO processing
        stack = []
        stack.append(self.root)
        while True:
            if len(stack):
                pop_node = stack.pop()
                if pop_node.left:   # We only prune nodes with childeren nodes
                    if pop_node.backuperror > pop_node.mcp:
                        node = None
                    else:
                        stack.append(pop_node.right)
                        stack.append(pop_node.left)
            else:
                break

    
    def view(self, method, saveflag=False, savename='Decision Tree'):
 
        # Object type check and analysis to avoid invalid input
        if isinstance(method, str) == True:
            if method == 'text' or method == 't':
                method = 0
            elif method == 'graph' or method == 'g':
                method = 1
            else:
                raise ValueError
        elif isinstance(method, int) == True:
            if method == 0 or method == 1:
                pass
            else:
                raise ValueError
        else:
            raise TypeError
        
        # Visualise by calling specific functions
        if method == 0:
            print('Visulising the decision tree in {}.'.format('text'))
            self._view_text(saveflag, savename)
        else:
            print('Visulising the decision tree {}.'.format('graphically'))
            self._view_graph(saveflag, savename)
        
    def _get_prefix(self, depth):

        default_prefix = '|---'
        depth_prefix = '|\t'
        prefix = depth_prefix * (depth - 1) + default_prefix
        return prefix

    def _view_node_text(self, node, fw):
        data_df = node.data
        # Number of samples
        str_samples = str(len(data_df))
         # Impurity
        str_method = self.criterion
        if str_method == 'entropy':
            impurity = calculate_entropy(data_df.values)
        elif str_method == 'gini':
            impurity = calculate_gini(data_df.values)
        elif str_method == 'mce':
            impurity = calculate_mce(data_df.values)
        else:
            raise ValueError
        str_predicted_class = str(node.classification) + '\n' if str(node.classification) else ''
        np_classes = np.unique(data_df[data_df.columns[-1]].to_numpy())
        #str_actual_classes = list(np.array2string(np.unique(np_classes)))
        #print(str_actual_classes)
        if node.prev_condition: # If there is a condition rather than None
            line = f'{self._get_prefix(node.depth)}{node.prev_condition}{"samples: ",str_samples}'
            # save to .txt
            if fw:
                fw.write(line+'\n')
            print(line)
        if node.classification: # If there is a classification rather than None
            line = f'{self._get_prefix(node.depth+1)}{node.classification}{"samples: ", str_samples}'
            if fw:
                fw.write(line+'\n')
            print(line)

    def _view_text(self, saveflag=False, savename='Decision Tree'):

        # First the root
        stack = []  # LIFO, Build a stack to store the Nodes
        stack.append(self.root)
        fw = None   # Open file
        if saveflag:
            fw = open(savename+'.txt','w')
        while True:
            if len(stack):
                pop_node = stack.pop()  # Pop out the visiting node
                self._view_node_text(pop_node, fw)    # Recursice process
                if pop_node.left:
                    stack.append(pop_node.right)
                    stack.append(pop_node.left)
            else:
                break
        if fw:
            fw.close()

    def _view_node_graph(self, node, coords):

        data_df = node.data
        # Condition
        str_condition = node.prev_condition + '\n' if node.prev_condition else ''
        print("str_condition: ", str_condition)
        print()
        # Impurity
        str_method = self.criterion
        if str_method == 'entropy':
            impurity = calculate_entropy(data_df.values)
        elif str_method == 'gini':
            impurity = calculate_gini(data_df.values)
        elif str_method == 'mce':
            impurity = calculate_mce(data_df.values)
        else:
            raise ValueError

        print("str_method: ", str_method)
        print()
        # Number of samples
        str_samples = str(len(data_df))
        print("str_samples: ", str_samples)
        print()
        # Classes
        #print(type(node.classification))
        str_predicted_class = str(node.classification) + '\n' if str(node.classification) else ''
        print("str_predicted_class: ", str_predicted_class)
        print()
        np_classes = np.unique(data_df[data_df.columns[-1]].to_numpy())
        print(np_classes)
      
        str_actual_classes = ',\n'.join(list(np.array2string(np.unique(np_classes))))
        print("str_actual_classes: ", str_actual_classes)
        
        # Plot the text with bound
        (x, y) = coords
        node_text = str_condition + str_method + ' = ' + str(round(impurity,4)) + '\n' + 'samples = ' + str_samples + '\n' + 'class = ' + str_predicted_class + 'Actual classes = ' + str_actual_classes
        plt.text(x, y, node_text, color='black', ha='center', va='center')

        # If there are son nodes
        x_offset = 0.3
        y_offset = 0.1
        line_y_offset = 0.015
        if node.left:
            coords_left = (x-x_offset, y-y_offset)  # Coordinates of the left son node
            coords_right = (x+x_offset, y-y_offset)  # Coordinates of the right son node
            line_to_sons = ([x-x_offset, x, x+x_offset], [y-y_offset+line_y_offset, y-line_y_offset, y-y_offset+line_y_offset])
            # Plot connection lines
            plt.plot(line_to_sons[0], line_to_sons[1], color='black', linewidth=0.5)

            # Recursive part
            self._view_node_graph(node.left, coords_left)
            self._view_node_graph(node.right, coords_right)

        

    def _view_graph(self, saveflag=False, savename='Decision Tree'):

        plt.figure()
        self._view_node_graph(self.root, (0,0)) # Plot from the root at (0,0)
        plt.axis('off')
        
        if saveflag:
            plt.savefig(savename + '.pdf', bbox_inches='tight', orientation ='landscape')
            plt.savefig(savename + '.jpg', bbox_inches='tight')
        plt.show()
    
    def print_info(self):

        print(          
            tabulate(
                [
                    ['Data head', self.root.data.head() if self.root else None],
                    ['Criterion', self.criterion],
                    ['Minimum size of the node data', self.min_samples],
                    ['Maximum depth of the tree', self.max_depth],
                    ['Post_pruning', self.post_prune],
                    ['Features', [feature for feature in self.features]],
                    ['All classes', list(np.unique(self.root.data[self.root.data.columns[-1]].to_numpy()))]
                ], headers=['Attributes', 'Values'], tablefmt='fancy_grid'
            )
        )
        

    def predict(self, test_data_df):

        # Only one row of sample
        if len(test_data_df) == 1: 
            class_name = self._predict_example(test_data_df, self.root)
            return class_name
        else:   # Multiple rows
            predicted_classes = []

            # Iterate over all samples and store the classes in a list
            for i_row in range(len(test_data_df)):
                test_data_example = test_data_df[i_row:i_row+1]
                predicted_classes.append(self._predict_example(test_data_example, self.root))
            return predicted_classes
      


    def _predict_example(self, data_df, node):

        # If there are son nodes for further expanding
        if node.left:   # Yes
            feature_name = node.left.prev_feature
            split_thresh = node.left.prev_thresh

            # Recursive part
            if data_df.iloc[0][feature_name] <= split_thresh: # Go to left son
                return self._predict_example(data_df, node.left)
            else:   # Go to right son
                return self._predict_example(data_df, node.right)
        
        else: # No expanding
            return node.classification

import random

def train_test_split(df, test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
     
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df

random.seed(1)  # For reproduction
train_data, test_data = train_test_split(Wine_dataset, test_size=0.3)

classification_method = input("choose classification methods from entropy, gini and mce: ")
print(classification_method)

dt = DesicionTree(criterion=classification_method, post_prune=False)
dt.feed(train_data)

dt.view(method='t', saveflag=True)  # View in text

extended_test_data = test_data.copy()   # Deep copy to avoid shared reference
predicted_classes = dt.predict(extended_test_data)  # Predict

extended_test_data['predicted'] = predicted_classes # Add a column of predicted

y_true = extended_test_data['quality'].to_numpy()
y_predicted = extended_test_data['predicted'].to_numpy()

def confusion_matrix(true_class, predicted_class):

  # extract all unique classes from the train y class
  unique_classes = np.unique(true_class)
  # print('unique', unique_classes)

  # initialize a matrix with zero values that will be the final confusion matrix
  confusion_matrix = np.zeros((len(unique_classes), len(unique_classes)))

  for i in range(len(unique_classes)):
    for j in range(len(unique_classes)):
      confusion_matrix[i, j] = np.sum((true_class == unique_classes[i]) & (predicted_class == unique_classes[j]))

  return confusion_matrix

y_true = extended_test_data['quality'].to_numpy()
y_predicted = extended_test_data['predicted'].to_numpy()

print(confusion_matrix(y_true, y_predicted))

def accuracy(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    return accuracy
print(accuracy(y_true, y_predicted))

