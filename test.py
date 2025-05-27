import csv
from dt import train_decision_tree

def load_train_dataset(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        headers = next(reader)
        feature_names = headers[1:-1]  # Skip ID, exclude label
        X, y = [], []
        for row in reader:
            X.append(row[1:-1])  # Skip ID column
            y.append(int(row[-1])) # Last column is the label
    return feature_names, X, y

def train_fruit_classifier(filename):
    f, X, y = load_train_dataset(filename)
    dt = train_decision_tree(X, y)
    return dt

def load_test_dataset(filename):
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader)  # pular cabeçalho
        X, y = [], []
        for row in reader:
            X.append(row[1:-1])
            y.append(int(row[-1]))
    return X, y

def accuracy(y_true, y_pred):
    return sum(yt == yp for yt, yp in zip(y_true, y_pred)) / len(y_true)

def main():
    dt = train_fruit_classifier('train.csv')  
    
    X_test, y_test = load_test_dataset('test.csv')  
    y_pred = [dt.predict(x) for x in X_test]  

    acc = accuracy(y_test, y_pred)
    print(f'Acurácia no conjunto de teste: {acc*100:.2f}%')

    for i, (pred, real) in enumerate(zip(y_pred, y_test), 1):
        print(f'Exemplo {i}: Predição={pred}, Real={real}')

if __name__ == '__main__':
    main()


