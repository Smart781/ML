from sklearn.datasets import make_blobs, make_moons
import numpy as np
import pandas
import random
from typing import Callable, Union, NoReturn, Optional, Dict, Any, List

# Task 1

def gini(x: np.ndarray) -> float:
    """
    Считает коэффициент Джини для массива меток x.
    """
    c = np.unique(x, return_counts=True)[1]
    p = c / c.sum()
    return np.sum(p * (1 - p))
    
    
def entropy(x: np.ndarray) -> float:
    """
    Считает энтропию для массива меток x.
    """
    c = np.unique(x, return_counts=True)[1]
    p = c / c.sum()
    return -np.sum(p * np.log2(p))

def gain(left_y: np.ndarray, right_y: np.ndarray, criterion: Callable) -> float:
    """
    Считает информативность разбиения массива меток.

    Parameters
    ----------
    left_y : np.ndarray
        Левая часть разбиения.
    right_y : np.ndarray
        Правая часть разбиения.
    criterion : Callable
        Критерий разбиения.
    """
    node = np.concatenate((left_y, right_y))
    return criterion(node) - (len(left_y) / len(node)) * criterion(left_y) - (len(right_y) / len(node)) * criterion(right_y)


# Task 2

from typing import Union

class DecisionTreeLeaf:
    """

    Attributes
    ----------
    y : Тип метки (напр., int или str)
        Метка класса, который встречается чаще всего среди элементов листа дерева
    """
    def __init__(self, ys):
        val, counts = np.unique(ys, return_counts=True)
        self.y = val[np.argmax(counts)]

class DecisionTreeNode:
    """

    Attributes
    ----------
    split_dim : int
        Измерение, по которому разбиваем выборку.
    split_value : float
        Значение, по которому разбираем выборку.
    left : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] < split_value.
    right : Union[DecisionTreeNode, DecisionTreeLeaf]
        Поддерево, отвечающее за случай x[split_dim] >= split_value. 
    """
    def __init__(self, split_dim: int, split_value: float, 
                 left: Union['DecisionTreeNode', DecisionTreeLeaf], 
                 right: Union['DecisionTreeNode', DecisionTreeLeaf]):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right
        
# Task 3

class DecisionTreeClassifier:
    """
    Attributes
    ----------
    root : Union[DecisionTreeNode, DecisionTreeLeaf]
        Корень дерева.

    (можете добавлять в класс другие аттрибуты).

    """
    def __init__(self, criterion : str = "gini", 
                 max_depth : Optional[int] = None, 
                 min_samples_leaf: int = 1):
        """
        Parameters
        ----------
        criterion : str
            Задает критерий, который будет использоваться при построении дерева.
            Возможные значения: "gini", "entropy".
        max_depth : Optional[int]
            Ограничение глубины дерева. Если None - глубина не ограничена.
        min_samples_leaf : int
            Минимальное количество элементов в каждом листе дерева.

        """
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Строит дерево решений по обучающей выборке.

        Parameters
        ----------
        X : np.ndarray
            Обучающая выборка.
        y : np.ndarray
            Вектор меток классов.
        """
        self.root = self.build_tree(X, y, 0)

    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Union[DecisionTreeNode, DecisionTreeLeaf]:
        if depth == self.max_depth or len(set(y)) == 1 or len(y) <= self.min_samples_leaf:
            return DecisionTreeLeaf(y)
        best_ig = -1
        best_feature = None
        best_border = None
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            if len(values) > 200:
                r = len(values) // 100
            else:
                r = 1
            for i in range(0, len(values), r):
                border = values[i]
                left_X = X[:, feature] < border
                right_X = X[:, feature] >= border
                if len(y[left_X]) < self.min_samples_leaf or len(y[right_X]) < self.min_samples_leaf:
                    continue
                if len(left_X) > 0 and len(right_X) > 0:
                    if self.criterion == "gini":
                        ig = gain(y[left_X], y[right_X], gini)
                    else:
                        ig = gain(y[left_X], y[right_X], entropy)
                    if ig > best_ig:
                        best_ig = ig
                        best_feature = feature
                        best_border = border
        if best_feature is None:
            return DecisionTreeLeaf(y)                
        left_X = X[:, best_feature] < best_border
        right_X = X[:, best_feature] >= best_border
        left_tree = self.build_tree(X[left_X], y[left_X], depth + 1)
        right_tree = self.build_tree(X[right_X], y[right_X], depth + 1)  
        return DecisionTreeNode(best_feature, best_border, left_tree, right_tree)
        
    def predict_proba(self, X: np.ndarray) ->  List[Dict[Any, float]]:
        """
        Предсказывает вероятность классов для элементов из X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        List[Dict[Any, float]]
            Для каждого элемента из X возвращает словарь 
            {метка класса -> вероятность класса}.
        """
        answer = []
        for x in X:
            dict = self.search(x, self.root)
            answer += [dict]
        return answer

    def search(self, x: np.ndarray, value) -> Dict[Any, float]:
        if isinstance(value, DecisionTreeLeaf):
            return {value.y: 1.0}

        if x[value.split_dim] < value.split_value:
            return self.search(x, value.left)
        else:
            return self.search(x, value.right)        
    
    def predict(self, X : np.ndarray) -> list:
        """
        Предсказывает классы для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Элементы для предсказания.
        
        Return
        ------
        list
            Вектор предсказанных меток для элементов X.
        """
        proba = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in proba]
    
# Task 4
task4_dtc = DecisionTreeClassifier(max_depth=6)

