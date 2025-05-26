import numpy as np
import math

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None,depth=0): # Additional optional arguments can be added, but the default value needs to be provided
        dataset_entropy=self.entropy(y)
        total=len(y)
        

        if all(val == y[0] for val in y):
            self.label = y[0]
            return
        
        if max_depth is not None and depth >= max_depth:
            self.label = self.most_common(y)
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


        if melhor_ganho < threshold:
            self.label = self.most_common(y)
            return
        
        self.attribute = melhor_attr
        self.branches = {}

        for valor in melhor_grupos:
            sub_X = []
            sub_y = []
            for i in range(len(X)):
                if X[i][melhor_attr] == valor:
                    sub_X.append(X[i])
                    sub_y.append(y[i])
            self.branches[valor] = DecisionTree(sub_X, sub_y, threshold, max_depth, depth + 1)


        

    def predict(self, x): # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        # Se estamos numa folha (tem atributo 'label')
        if hasattr(self, 'label'):
            return self.label

        # Caso contrário, estamos num nó → usar atributo para decidir o caminho
        valor = x[self.attribute]

        # Se esse valor existe nos ramos → seguir para esse ramo
        if valor in self.branches:
            return self.branches[valor].predict(x)

        # Valor nunca visto no treino → devolve classe padrão (ex: 1)
        return 1
    
    def entropy(self,y):
        n_fruit = 0
        n_bomb = 0
        for val in y:
            if val == 1:
                n_fruit += 1
            else:
                n_bomb += 1
        
        prob_bomb=n_bomb/len(y)
        prob_fruit=n_fruit/len(y)
        pbomb_log2=0
        pfruit_log2=0
        if prob_bomb>0:
            pbomb_log2=math.log2(prob_bomb)

        if prob_fruit>0:
            pfruit_log2=math.log2(prob_fruit)

        
        
        entropy=-(prob_bomb*pbomb_log2+prob_fruit*pfruit_log2)
        return entropy
    
    def most_common(self,y):
        n_fruit = 0
        n_bomb = 0
        for val in y:
            if val == 1:
                n_fruit += 1
            else:
                n_bomb += 1

        if n_bomb>n_fruit:
            return -1
        else:
            return 1


    
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
    return DecisionTree(X, y, threshold=0.7, max_depth=3)

