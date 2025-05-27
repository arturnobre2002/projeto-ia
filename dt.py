import numpy as np
import math

class DecisionTree:

    def __init__(self, X, y, threshold=1.0, max_depth=None,depth=0): # Additional optional arguments can be added, but the default value needs to be provided
        dataset_entropy=self.entropy(y) # entropia do dataset
        total=len(y) # tamanho do dataset
        

        if all(val == y[0] for val in y): # se o dataset e homogeneo
            self.leaf = y[0] # nasce uma folha com esse valor (1 ou -1)
            return
        
        if max_depth is not None and depth >= max_depth: # se atingiu a max_depth
            self.leaf = self.most_common(y) # poe uma folha com o valor mais comum (1 ou -1)
            return

        best_attr = -1 # melhor atributo vai ser o que tem maior ig
        best_ig = -1 # melhor ig
        best_groups = None # grupos do melhor atributo

        for attr_index in range(len(X[0])): # percorre as colunas de X, ou seja, os atributos
            groups = self.split_by_attribute(X, y, attr_index) # vai buscar os grupos desse atributo
            ig = self.information_gain(total, dataset_entropy, groups) # calcula o ig desse atributo
            if ig >= best_ig: # se o ig for melhor que o ig do atributo anterior, atualiza
                best_ig = ig
                best_attr = attr_index
                best_groups = groups

        # se o melhor ig for menor que o threshold (valor minimo de ig para esse atributo se tornar numa decisao)
        if best_ig < threshold:
            self.leaf = self.most_common(y) # adiciona uma folha com o valor mais comum e retorna
            return
        
        self.attribute = best_attr # se o ig for suficiente introduz o atributo na arvore 
        self.branches = {} # inicializa um dicionario de ramos para esse atributo

        for val in best_groups: # percorre o grupo desse atributo
            sub_X = [] # para cada grupo vai buscar o subset respetivo
            sub_y = []
            for i in range(len(X)):
                if X[i][best_attr] == val:
                    sub_X.append(X[i])
                    sub_y.append(y[i])
            self.branches[val] = DecisionTree(sub_X, sub_y, threshold, max_depth, depth + 1) # adiciona no dicionario de ramos do atributo um ramo em que o valor do grupo aponta para uma nova arvore de decisao que recebe o subset 


        

    def predict(self, x): # (e.g. x = ['apple', 'green', 'circle'] -> 1 or -1)
        # o hasattr ve se tem atributo 'leaf', significa que estamos numa folha
        if hasattr(self, 'leaf'):
            return self.leaf

        # se nao, estamos num no. entao vai buscar o valor a x do atributo do no em que estamos. ex: se self.attribute=1 estamos no atributo cor e ele vai buscar 'green'
        val = x[self.attribute]

        # se o valor existe nos ramos faz uma chamada recursiva para esse ramo. isto acontece ate chegar a uma folha, ai devolve o valor da folha
        if val in self.branches:
            return self.branches[val].predict(x)

        # se o valor nunca foi visto no treino, devolve -1. no caso do nosso treino nunca vai acontecer
        return -1
    
    # funcao auxiliar que calcula a entropia de um dado dataset
    def entropy(self,y):
        n_fruit = 0 # contagem de frutas
        n_bomb = 0 # contagem de bombas
        for val in y:
            if val == 1:
                n_fruit += 1
            else:
                n_bomb += 1
        
        prob_bomb = n_bomb/len(y) # probabilidade de ser bomba
        prob_fruit = n_fruit/len(y) # probabilidade de ser fruta
        pbomb_log2 = 0
        pfruit_log2 = 0
        # evita fazer log2(0)
        if prob_bomb > 0: 
            pbomb_log2 = math.log2(prob_bomb)

        if prob_fruit > 0:
            pfruit_log2 = math.log2(prob_fruit)

        
        entropy = -(prob_bomb * pbomb_log2 + prob_fruit * pfruit_log2)
        return entropy
    # funcao auxiliar para ver se ha mais bombas ou mais frutas num dataset. util para escolher folhas 
    def most_common(self,y):
        n_fruit = 0
        n_bomb = 0
        for val in y:
            if val == 1:
                n_fruit += 1
            else:
                n_bomb += 1

        if n_bomb > n_fruit:
            return -1
        else:
            return 1


    
    def split_by_attribute(self, X, y, attr_index): # grupos de um atributo com attr_index p exemplo atrr_index=0 entao o atributo é o Nome
        groups = {}  # ex: {'apple': [lista de y], 'orange': [lista de y], ...}
        for i in range(len(X)): # percorre as linhas de X
            val = X[i][attr_index] # vai buscar o valor a linha i do atributo com attr_index
            if val not in groups: # se ainda nao registou o grupo, adiciona a groups
                groups[val] = []
            groups[val].append(y[i]) # adiciona y[i] (respetivo y) a lista de ys desse grupo
        return groups
    
    # funcao para calcular o ig de um atributo
    def information_gain(self, total, dataset_entropy, groups): # groups = grupos de um atributo
        weighted_entropy = 0 # weighted_entropy = (#s1/#dataset * entropy(s1) + ... + #si/#dataset * entropy(si)) # i = nr de valores/grupos de um atributo
        for group in groups:
            sub_y = groups[group]
            weight = len(sub_y) / total
            weighted_entropy += weight * self.entropy(sub_y)

        ig = dataset_entropy - weighted_entropy
        return ig
    





def train_decision_tree(X, y):
    # Replace with your configuration
    return DecisionTree(X, y, threshold=0.1, max_depth=None)

