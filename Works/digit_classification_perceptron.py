import numpy as np
from sklearn.model_selection import train_test_split
import copy
from typing import NoReturn


# Task 1

class Perceptron:
    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.iterations = iterations
        self.vals = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает простой перцептрон. 
        Для этого сначала инициализирует веса перцептрона,
        а затем обновляет их в течении iterations итераций.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        n, m = X.shape
        self.vals = np.unique(y)
        mask0 = (y == self.vals[0])
        mask1 = (y == self.vals[1])
        y[mask0] = -1
        y[mask1] = 1
            
        ones = np.ones((1, n))
        X = X.T
        X = np.concatenate((ones, X))
        X = X.T
        self.w = np.array([0] * (m + 1), dtype=float)
        for i in range(self.iterations):
            z = np.sign(X.dot(self.w))
            mask = (y != z)
            X_mask = X[mask]
            y_mask = y[mask]
            X_mark = X_mask * y_mask[:, np.newaxis]
            self.w += np.sum(X_mark, axis=0)
                    
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        ones = np.ones((1, X.shape[0]))
        X = X.T
        X = np.concatenate((ones, X))
        X = X.T
        ans = X.dot(self.w)  
        ans = np.sign(ans).astype(int)
        mask0 = (ans == -1)
        mask1 = (ans == 1)
        ans[mask0] = self.vals[0]
        ans[mask1] = self.vals[1]
        return ans     
    
# Task 2

class PerceptronBest:

    def __init__(self, iterations: int = 100):
        """
        Parameters
        ----------
        iterations : int
        Количество итераций обучения перцептрона.

        Attributes
        ----------
        w : np.ndarray
        Веса перцептрона размерности X.shape[1] + 1 (X --- данные для обучения), 
        w[0] должен соответстовать константе, 
        w[1:] - коэффициентам компонент элемента X.

        Notes
        -----
        Вы можете добавлять свои поля в класс.
        
        """

        self.w = None
        self.iterations = iterations
        self.vals = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает перцептрон.

        Для этого сначала инициализирует веса перцептрона, 
        а затем обновляет их в течении iterations итераций.

        При этом в конце обучения оставляет веса, 
        при которых значение accuracy было наибольшим.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, на котором обучается перцептрон.
        y: np.ndarray
            Набор меток классов для данных.
        
        """
        n, m = X.shape
        self.vals = np.unique(y)
        mask0 = (y == self.vals[0])
        mask1 = (y == self.vals[1])
        y[mask0] = -1
        y[mask1] = 1
            
        w_best = np.array([0] * (m + 1), dtype=float)
        best_accuracy = 0
        ones = np.ones((1, n))
        X = X.T
        X = np.concatenate((ones, X))
        X = X.T
        self.w = np.array([0] * (m + 1), dtype=float)
        for i in range(self.iterations):
            z = np.sign(X.dot(self.w))
            mask = (y != z)
            accuracy = n - y[mask].shape[0]
            if accuracy > best_accuracy:
                w_best = np.copy(self.w)
                best_accuracy = accuracy   
            X_mark = X[mask] * y[mask][:, np.newaxis]
            self.w += np.sum(X_mark, axis=0)
        z = np.sign(X.dot(self.w))
        mask = (y != z)
        accuracy = n - y[mask].shape[0]
        if accuracy > best_accuracy:
            w_best = np.copy(self.w)
            best_accuracy = accuracy
        self.w = np.copy(w_best)
            
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает метки классов.
        
        Parameters
        ----------
        X : np.ndarray
            Набор данных, для которого необходимо вернуть метки классов.
        
        Return
        ------
        labels : np.ndarray
            Вектор индексов классов 
            (по одной метке для каждого элемента из X).
        
        """
        ones = np.ones((1, X.shape[0]))
        X = X.T
        X = np.concatenate((ones, X))
        X = X.T
        ans = X.dot(self.w)  
        ans = np.sign(ans).astype(int)
        mask0 = (ans == -1)
        mask1 = (ans == 1)
        ans[mask0] = self.vals[0]
        ans[mask1] = self.vals[1]
        return ans     
    
# Task 3

def transform_images(images: np.ndarray) -> np.ndarray:
    """
    Переводит каждое изображение в вектор из двух элементов.
        
    Parameters
    ----------
    images : np.ndarray
        Трехмерная матрица с черное-белыми изображениями.
        Её размерность: (n_images, image_height, image_width).

    Return
    ------
    np.ndarray
        Двумерная матрица с преобразованными изображениями.
        Её размерность: (n_images, 2).
    """
    result = np.zeros((images.shape[0], 2)) # Первая координата - интенсивность, вторая - симметрия относительно горизонтали
    for i in range(images.shape[0]):
        image = images[i]
        result[i][0] = np.sum(image) / (image.shape[0] * image.shape[1])
        sym_image = np.flipud(image)
        mark = (sym_image == image)
        result[i][1] = np.sum(mark.astype(int)) / (image.shape[0] * image.shape[1])
    return result    