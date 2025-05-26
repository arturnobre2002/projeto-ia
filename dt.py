import numpy as np
import math

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None): # Additional optional arguments can be added, but the default value needs to be provided
        dataset_entropy=self.entropy(y)
        total=len(y)



        pass

    def predict(self, x): # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        # Implement this
        pass
    
    def entropy(self,y):
        n_bomb=0
        n_fruit=0
        for val in y:
            if val == 1:
                n_bomb += 1
            else: 
                n_fruit += 1
        
        prob_bomb=n_bomb/len(y)
        prob_fruit=n_fruit/len(y)
        entropy=-(prob_bomb*math.log2(prob_bomb)+prob_fruit*math.log2(prob_fruit))
        return entropy
    
    def split_by_attribute(self, X, y, attr_index):
        groups = {}  # exemplo: {'circle': [lista de y], 'curved': [lista de y], ...}
        for i in range(len(X)):
            val = X[i][attr_index]
            if val not in groups:
                groups[val] = []
            groups[val].append(y[i])
        return groups
    
    def information_gain(self, total, dataset_entropy, groups):
        entropia_ponderada = 0
        for group in groups:
            sub_y = groups[group]
            peso = len(sub_y) / total
            entropia_ponderada += peso * self.entropy(sub_y)

        ig = dataset_entropy - entropia_ponderada
        return ig





def train_decision_tree(X, y):
    # Replace with your configuration
    return DecisionTree(X, y)

