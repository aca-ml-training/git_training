#random_forest.py
from decision_tree import DecisionTree
from collections import Counter
import numpy as np

class RandomForest(object):
    """
    RandomForest a class, that represents Random Forests.

    :param num_trees: Number of trees in the random forest
    :param max_tree_depth: maximum depth for each of the trees in the forest.
    :param ratio_per_tree: ratio of points to use to train each of
        the trees.
    """
    def __init__(self, num_trees, max_tree_depth, ratio_per_tree=0.5, classifier = DecisionTree(100)):
        self.num_trees = num_trees
        self.classifier = classifier
        self.ratio_per_tree = ratio_per_tree
        self.max_tree_depth = max_tree_depth
        self.trees = None

    def fit(self, X, Y):
        self.trees = []
        data = np.hstack((X, Y))
        for i in range(self.num_trees):
            rand = data[np.random.randint(data.shape[0], size=data.shape[0]*self.ratio_per_tree), :]
            rand.tolist()
            
            self.trees.append(self.classifier.fit(rand))
        return self.trees 

    def predict(self, X, model):
        
        self.X = X
        self.model = model
        temp = []
        conf = []
        Y = []

        for i in range(len(model)):
            temp.append((self.classifier.predict(X, model[i], []))  )
        for i in range(len(temp[0])):

            count = Counter([row[i] for row in temp])
            maximum = max(np.array(list(count.values())))
            Y.append(next((j for j, k in count.items() if k == maximum)))
          
      
        return (Y, conf)
