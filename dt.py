import numpy as np
import math

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None): # Additional optional arguments can be added, but the default value needs to be provided
        dataset_entropy=self.entropy(y)
        total=len(y)


        if all(val == y[0] for val in y):
            self.label = y[0]
            return

        melhor_attr = -1
        melhor_ganho = -1
        melhor_grupos = None

        for attr_index in range(len(X[0])):
            grupos = self.split_by_attribute(X, y, attr_index)
            ganho = self.information_gain(total, dataset_entropy, grupos)
            if ganho >= melhor_ganho:
                melhor_ganho = ganho
                melhor_attr = attr_index
                melhor_grupos = grupos


        self.attribute = melhor_attr
        self.branches = {}

        for valor in melhor_grupos:
            sub_X = []
            sub_y = []
            for i in range(len(X)):
                if X[i][melhor_attr] == valor:
                    sub_X.append(X[i])
                    sub_y.append(y[i])
            self.branches[valor] = DecisionTree(sub_X, sub_y)

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
        pbomb_log2=0
        pfruit_log2=0
        if prob_bomb>0:
            pbomb_log2=math.log2(prob_bomb)

        if prob_fruit>0:
            pfruit_log2=math.log2(prob_fruit)

        
        
        entropy=-(prob_bomb*math.log2(prob_bomb)+prob_fruit*math.log2(prob_fruit))
        return entropy
    
    def split_by_attribute(self, X, y, attr_index): #grupos de um atributo com attr_index p exemplo atrr_index=0 entao o atributo é o Nome
        groups = {}  # exemplo: {'circle': [lista de y], 'curved': [lista de y], ...}
        for i in range(len(X)):
            val = X[i][attr_index]
            if val not in groups:
                groups[val] = []
            groups[val].append(y[i])
        return groups
    
    def information_gain(self, total, dataset_entropy, groups): #grupos de um atributo
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

