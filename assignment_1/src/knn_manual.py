import numpy as np 
from collections import Counter


def euclidean_distance(a,b):
    return np.sqrt(np.sum(( a - b ) ** 2))

def KNN_predict(x_train,y_train,x_test,k = 5):
    predictions = []
    print("meow")
    print("Type of y_train:", type(y_train))
    print("First few indices of y_train (if Series):", getattr(y_train, "index", "N/A")[:10])
    print("First few values of y_train:", np.array(y_train)[:10])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    for x in x_test:
        distances = np.array([euclidean_distance(x, x_t) for x_t in x_train])
        k_indices = np.argsort(distances)[:k]
        k_labels  = [y_train[i] for i in k_indices]
        predictions.append(Counter(k_labels).most_common(1)[0][0])
    return np.array(predictions)
