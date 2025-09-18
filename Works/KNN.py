import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import pandas
from typing import NoReturn, Tuple, List


# Task 1

def read_cancer_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к cancer датасету.

    Returns
    -------
    X : np.array
        Матрица признаков опухолей.
    y : np.array
        Вектор бинарных меток, 1 соответствует доброкачественной опухоли (M), 
        0 --- злокачественной (B).

    
    """
    dataframe = pandas.read_csv(path_to_csv, delimiter = ',')
    arr = np.random.choice(dataframe.shape[0], dataframe.shape[0], replace = False)
    dataframe = dataframe.iloc[arr]
    dataframe = dataframe.T
    tuple = [dataframe[(dataframe.iloc[:,0] != 'M') & (dataframe.iloc[:,0] != 'B')].T, dataframe[(dataframe.iloc[:,0] == 'M') | (dataframe.iloc[:,0] == 'B')].T]
    tuple[1] = (tuple[1] == 'M').astype(int)
    return tuple[0].values, tuple[1].values

def read_spam_dataset(path_to_csv: str) -> Tuple[np.array, np.array]:
    """
     
    Parameters
    ----------
    path_to_csv : str
        Путь к spam датасету.

    Returns
    -------
    X : np.array
        Матрица признаков сообщений.
    y : np.array
        Вектор бинарных меток, 
        1 если сообщение содержит спам, 0 если не содержит.
    
    """
    dataframe = pandas.read_csv(path_to_csv, delimiter = ',')
    size = dataframe.shape[0]
    arr = np.random.choice(size, size, replace = False)
    dataframe = dataframe.iloc[arr]
    dataframe = dataframe.T
    size = dataframe.shape[0]
    tuple = [dataframe[0:(size - 1)].T, dataframe[(size - 1):size].T]
    return tuple[0].values, tuple[1].values
    
# Task 2

def train_test_split(X: np.array, y: np.array, ratio: float) -> Tuple[np.array, np.array, np.array, np.array]:
    """

    Parameters
    ----------
    X : np.array
        Матрица признаков.
    y : np.array
        Вектор меток.
    ratio : float
        Коэффициент разделения.

    Returns
    -------
    X_train : np.array
        Матрица признаков для train выборки.
    y_train : np.array
        Вектор меток для train выборки.
    X_test : np.array
        Матрица признаков для test выборки.
    y_test : np.array
        Вектор меток для test выборки.

    """
    size = X.shape[0]
    r = int(ratio * size)
    index = np.arange(0, size)
    train = np.random.choice(size, r, replace = False)
    test = np.setdiff1d(index, train)
    return X[train], y[train], X[test], y[test]
    
# Task 3

def get_precision_recall_accuracy(y_pred: np.array, y_true: np.array) -> Tuple[np.array, np.array, float]:
    """

    Parameters
    ----------
    y_pred : np.array
        Вектор классов, предсказанных моделью.
    y_true : np.array
        Вектор истинных классов.

    Returns
    -------
    precision : np.array
        Вектор с precision для каждого класса.
    recall : np.array
        Вектор с recall для каждого класса.
    accuracy : float
        Значение метрики accuracy (одно для всех классов).

    """
    if len(y_pred.shape) == 1:
        size = y_pred.shape[0]
    else:    
        size = max(y_pred.shape[0], y_pred.shape[1])
    y_pred = y_pred.reshape(1, size)
    y_true = y_true.reshape(1, size)
    classes = np.unique(np.array(y_pred.tolist() + y_true.tolist())).astype(int)
    conc = np.concatenate((y_pred, y_true))
    conc = conc.T
    precision = np.zeros((1, len(classes)), float)
    recall = np.zeros((1, len(classes)), float)
    for k in classes:
        tp = conc[(conc[:,0] == k) & (conc[:,0] == conc[:,1])].shape[0]
        fp = conc[(conc[:,0] == k) & (conc[:,0] != conc[:,1])].shape[0]
        fn = conc[(conc[:,1] == k) & (conc[:,0] != conc[:,1])].shape[0]
        precision[0][k] = tp / (tp + fp)
        recall[0][k] = tp / (tp + fn)
    accuracy = conc[conc[:,0] == conc[:,1]].shape[0] / size   
    return precision.T, recall.T, accuracy
    
# Task 4

class KDTree:
    def __init__(self, X: np.array, leaf_size: int = 40):
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которому строится дерево.
        leaf_size : int
            Минимальный размер листа
            (то есть, пока возможно, пространство разбивается на области, 
            в которых не меньше leaf_size точек).

        Returns
        -------

        """      
        self.points = X
        self.leaf_size = leaf_size
        self.left = None
        self.right = None
    
    def query(self, X: np.array, k: int = 1) -> List[List]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно найти ближайших соседей.
        k : int
            Число ближайших соседей.

        Returns
        -------
        list[list]
            Список списков (длина каждого списка k): 
            индексы k ближайших соседей для всех точек из X.

        """
        
        tree = self.build_tree(self.points, 0, self.points.shape[1], self.leaf_size)
        best = []
        distance = []
        def dist(point1, point2):
            distance = 0
            size = len(point1)
            for i in range(size - 1):
                distance += (point1[i] - point2[i]) ** 2
            return distance
        
        def search(node, point, size, depth, k, best, distance):
            if node is None:
                return [], []
            axis = depth % size
            next = None
            opposite = None
            if node.left is not None or node.right is not None:
                if point[axis] < node.med:
                    next = node.left
                    opposite = node.right
                else:
                    next = node.right
                    opposite = node.left
                search(next, point, size, depth + 1, k, best, distance)
                if len(best) < k or abs(point[axis] - node.med) ** 2 < distance[min(k, len(distance)) - 1]:
                    search(opposite, point, size, depth + 1, k, best, distance)
            else:
                for p in node.points:
                    d = dist(point, p)
                    indx = 0
                    for i in range(len(distance)):
                        if i >= k:
                            indx = 1
                            break
                        if d < distance[i]:
                            distance.insert(i, d)
                            best.insert(i, p)
                            indx = 1
                            break
                    if indx == 0:
                        distance.append(d)
                        best.append(p)
                
                
        index = np.arange(X.shape[0])
        X = np.column_stack((X, index))
        b = []
        for point in X:
            best = []
            distance = []
            sz = len(point)
            search(tree, point, X.shape[1] - 1, 0, k, best, distance)
            b_ind = []
            for i in range(k):
                b_ind += [best[i][sz - 1]]
            b += [b_ind]
        return b 
                
                

    def build_tree(self, points, depth, size, leaf_size):
        if depth == 0:
            index = np.array(np.arange(0, points.shape[0])).reshape(1, points.shape[0])
            points = points.T
            points = np.concatenate((points, index))
            points = points.T
        ax = depth % size
        if points.shape[0] == 0:
            return None
        if points.shape[0] <= 2 * leaf_size:
            tree = KDTree(points, leaf_size)
            return tree 
        med = np.median(points[:,ax])
        arr1 = points[points[:,ax] < med]
        arr2 = points[points[:,ax] >= med]
        if arr1.shape[0] < leaf_size or arr2.shape[0] < leaf_size:
            return KDTree(points, leaf_size)
        else:
            point = np.array([0] * points.shape[1])
            node = KDTree(point, leaf_size)
            node.med = med
            node.left = KDTree.build_tree(self, arr1, depth + 1, size, leaf_size)
            node.right = KDTree.build_tree(self, arr2, depth + 1, size, leaf_size)
            return node
        
# Task 5

class KNearest:
    def __init__(self, n_neighbors: int = 5, leaf_size: int = 30):
        """

        Parameters
        ----------
        n_neighbors : int
            Число соседей, по которым предсказывается класс.
        leaf_size : int
            Минимальный размер листа в KD-дереве.

        """        
        self.neighbors = n_neighbors
        self.leaf_size = leaf_size
        self.tree = None
        self.classes = None
        self.size = None
    
    def fit(self, X: np.array, y: np.array) -> NoReturn:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, по которым строится классификатор.
        y : np.array
            Метки точек, по которым строится классификатор.

        """        
        self.tree = KDTree(X, self.leaf_size)
        self.classes = y.astype(int).tolist()
        self.size = len(np.unique(y).tolist())
        
    def predict_proba(self, X: np.array) -> List[np.array]:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        list[np.array]
            Список np.array (длина каждого np.array равна числу классов):
            вероятности классов для каждой точки X.
            

        """
    
        lst = self.tree.query(X, self.neighbors)
        answer = []
        for i in lst:
            prob = np.array([float(0)] * self.size)
            for j in i:
                cls = self.classes[int(j)]
                prob[cls] += 1
            prob /= self.size
            answer += [prob]
        return answer    
        
    def predict(self, X: np.array) -> np.array:
        """

        Parameters
        ----------
        X : np.array
            Набор точек, для которых нужно определить класс.
        
        Returns
        -------
        np.array
            Вектор предсказанных классов.
            

        """
        return np.argmax(self.predict_proba(X), axis=1)
