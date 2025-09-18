from sklearn.neighbors import KDTree
from sklearn.datasets import make_blobs, make_moons
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
import copy
import cv2
from collections import deque
from typing import NoReturn
from queue import PriorityQueue

# Task 1

class KMeans:
    def __init__(self, n_clusters: int, init: str = "random", 
                 max_iter: int = 300):
        """
        
        Parameters
        ----------
        n_clusters : int
            Число итоговых кластеров при кластеризации.
        init : str
            Способ инициализации кластеров. Один из трех вариантов:
            1. random --- центроиды кластеров являются случайными точками,
            2. sample --- центроиды кластеров выбираются случайно из  X,
            3. k-means++ --- центроиды кластеров инициализируются 
                при помощи метода K-means++.
        max_iter : int
            Максимальное число итераций для kmeans.
        
        """
        
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = []
        
    def fit(self, X: np.array, y = None) -> NoReturn:
        """
        Ищет и запоминает в self.centroids центроиды кластеров для X.
        
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit обязаны принимать 
            параметры X и y, даже если y не используется).
        
        """
        def dist(point1, point2):
            distance = 0
            size = point1.shape[0]
            for i in range(size):
                distance += (point1[i] - point2[i]) ** 2
            return distance
            
        def cluster(point, centroids):
            centroid = 0
            distance = dist(point, centroids[0])
            for i in range(len(centroids)):
                d = dist(point, centroids[i])
                if d < distance:
                    distance = d
                    centroid = i
            return centroid
            
        def clusters(X, centroids):
            answer = []
            for point in X:
                answer.append(cluster(point, centroids))
            return answer    
            

        if self.init == "random":
            self.centroids = np.random.uniform(np.min(X), np.max(X), size=(self.n_clusters,X.shape[1]))
        elif self.init == "sample":
            arr = np.random.choice(X.shape[0], self.n_clusters, replace = False)
            self.centroids = X[arr]
        elif self.init == "k-means++":
            self.centroids = []
            distance = []
            ind = random.randint(0, X.shape[0])
            self.centroids += [X[ind]]
            centroid = 0
            for point in X:
                distance += [dist(self.centroids[-1], point)]
                if distance[-1] > distance[centroid]:
                    centroid = len(distance) - 1
            self.centroids += [X[centroid]]
            for i in range(2, self.n_clusters):
                centroid = 0
                for j in range(X.shape[0]):
                    distance[j] = min(distance[j], dist(self.centroids[i - 1], X[j]))
                    if distance[centroid] < distance[j]:
                        centroid = j
                self.centroids += [X[centroid]]
            self.centroids = np.array(self.centroids)
        cls = clusters(X, self.centroids)
        cls = np.array(cls)
        steps = 0
        while True:
            ind_cluster = [X[cls == i] for i in range(self.n_clusters)]
            for i in range(len(self.centroids)):
                if len(ind_cluster[i]) != 0:
                    self.centroids[i] = np.mean(ind_cluster[i], axis = 0)
            new_cls = clusters(X, self.centroids)
            new_cls = np.array(new_cls)
            if np.any(cls - new_cls) == False:
                break
            cls = new_cls    
            steps += 1
            if steps == self.max_iter:
                break
        answer = []        
        cls = np.unique(clusters(X, self.centroids))
        for i in cls:
            answer.append(self.centroids[i])
        self.centroids = np.array(answer)
            
                    
                    
                
            
    
    def predict(self, X: np.array) -> np.array:
        """
        Для каждого элемента из X возвращает номер кластера, 
        к которому относится данный элемент.
        
        Parameters
        ----------
        X : np.array
            Набор данных, для элементов которого находятся ближайшие кластера.
        
        Return
        ------
        labels : np.array
            Вектор индексов ближайших кластеров 
            (по одному индексу для каждого элемента из X).
        
        """

        def dist(point1, point2):
            distance = 0
            size = point1.shape[0]
            for i in range(size):
                distance += (point1[i] - point2[i]) ** 2
            return distance
            
        answer = []
        for point in X:
            centroid = 0
            distance = dist(point, self.centroids[0])
            for i in range(len(self.centroids)):
                d = dist(point, self.centroids[i])
                if d < distance:
                    distance = d
                    centroid = i
            answer.append(centroid)
        return np.array(answer)    
            
                    
    
# Task 2

class DBScan:
    def __init__(self, eps: float = 0.5, min_samples: int = 5, 
                 leaf_size: int = 40, metric: str = "euclidean"):
        """
        
        Parameters
        ----------
        eps : float, min_samples : int
            Параметры для определения core samples.
            Core samples --- элементы, у которых в eps-окрестности есть 
            хотя бы min_samples других точек.
        metric : str
            Метрика, используемая для вычисления расстояния между двумя точками.
            Один из трех вариантов:
            1. euclidean 
            2. manhattan
            3. chebyshev
        leaf_size : int
            Минимальный размер листа для KDTree.

        """
        self.eps = eps
        self.min_samples = min_samples
        self.leaf_size = leaf_size
        self.metric = metric
        
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """

        def core_dfs(x, color, ind, clusters):
            if clusters[x] != -1:
                return
            clusters[x] = color
            for i in ind[x]:
                if core_points[i] == True:
                    core_dfs(i, color, ind, clusters)   
        
        tree = KDTree(X, self.leaf_size, metric = self.metric)
        ind, dist = tree.query_radius(X, r = self.eps, return_distance = True)
        core_points = [False] * X.shape[0]
        clusters = [-1] * X.shape[0]
        distance = [-1] * X.shape[0]
        for i in range(X.shape[0]):
            if len(ind[i]) > self.min_samples:
                core_points[i] = True      
        mark = 0
        for i in range(X.shape[0]):
            if core_points[i] == True and clusters[i] == -1:
                core_dfs(i, mark, ind, clusters)
                mark += 1        
        for i in range(X.shape[0]):
            if core_points[i] == True:
                for j in range(len(ind[i])):
                    if core_points[ind[i][j]] != True:
                        if distance[ind[i][j]] == -1:
                            distance[ind[i][j]] = dist[i][j]
                            clusters[ind[i][j]] = clusters[i]
                        elif distance[ind[i][j]] < dist[i][j]:
                            distance[ind[i][j]] = dist[i][j]
                            clusters[ind[i][j]] = clusters[i]                  
        return np.array(clusters)                

# Task 3

class AgglomerativeClustering:
    def __init__(self, n_clusters: int = 16, linkage: str = "average"):
        """
        
        Parameters
        ----------
        n_clusters : int
            Количество кластеров, которые необходимо найти (то есть, кластеры 
            итеративно объединяются, пока их не станет n_clusters)
        linkage : str
            Способ для расчета расстояния между кластерами. Один из 3 вариантов:
            1. average --- среднее расстояние между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            2. single --- минимальное из расстояний между всеми парами точек, 
               где одна принадлежит первому кластеру, а другая - второму.
            3. complete --- максимальное из расстояний между всеми парами точек,
               где одна принадлежит первому кластеру, а другая - второму.
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def fit_predict(self, X: np.array, y = None) -> np.array:
        """
        Кластеризует элементы из X, 
        для каждого возвращает индекс соотв. кластера.
        Parameters
        ----------
        X : np.array
            Набор данных, который необходимо кластеризовать.
        y : Ignored
            Не используемый параметр, аналогично sklearn
            (в sklearn считается, что все функции fit_predict обязаны принимать 
            параметры X и y, даже если y не используется).
        Return
        ------
        labels : np.array
            Вектор индексов кластеров
            (Для каждой точки из X индекс соотв. кластера).

        """

        min_distance = PriorityQueue()
        size = X.shape[0]
        length = X.shape[0]
        distance = np.zeros((size, size))
        cluster = np.arange(size)
        col = [1] * size
        ind = [1] * size
        for i in range(size):
            r_min = np.inf
            index = -1
            for j in range(i + 1, size):
                r = np.linalg.norm(X[i] - X[j])
                distance[i][j] = r
                distance[j][i] = r
                if r < r_min:
                    r_min = r
                    index = j
            min_distance.put((r_min, i, index))
        dc = {}        
        while size > self.n_clusters:        
            while True:
                r, i, j = min_distance.get()
                if ind[cluster[i]] != -1 and ind[cluster[j]] != -1 and distance[i][j] == r:
                    a = cluster[i]
                    b = cluster[j]
                    break      
            r_min = np.inf
            index = -1        
            for i in range(length):
                if i != a and i != b and ind[cluster[i]] != -1:
                    if self.linkage == "average":
                        d = (distance[a][i] * col[a] + distance[b][i] * col[b]) / (col[a] + col[b])
                        distance[a][i] = d
                        distance[i][a] = d
                    elif self.linkage == "single":
                        d = min(distance[a][i], distance[b][i])
                        distance[a][i] = d
                        distance[i][a] = d
                    elif self.linkage == "complete":
                        d = max(distance[a][i], distance[b][i])
                        distance[a][i] = d
                        distance[i][a] = d
                    if d < r_min:
                        r_min = d
                        index = i              
                        
            col[a] += col[b]
            size -= 1
            dc[b] = a
            ind[b] = -1
            min_distance.put((r_min, a, index))
        dic = {}
        for i in range(X.shape[0]):
            nac = cluster[i]
            con = cluster[i]
            while True:
                if con not in dc or dc[con] == con:
                    break
                con = dc[con]
            dc[nac] = con
            cluster[i] = con
        mark = 0
        for i in range(X.shape[0]):
            c = cluster[i]
            if c not in dic:
                dic[c] = mark
                cluster[i] = mark
                mark += 1
            else:
                cluster[i] = dic[c]
        return np.array(cluster)