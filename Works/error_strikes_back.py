import numpy as np
import copy
from typing import List, NoReturn
import torch
from torch import nn
import torch.nn.functional as F


# Task 1

class Module:
    """
    Абстрактный класс. Его менять не нужно. Он описывает общий интерфейс взаимодествия со слоями нейронной сети.
    """

    def forward(self, x):
        pass

    def backward(self, d):
        pass

    def update(self, alpha):
        pass


class Linear(Module):
    """
    Линейный полносвязный слой.
    """

    def __init__(self, in_features: int, out_features: int):
        """
        Parameters
        ----------
        in_features : int
            Размер входа.
        out_features : int
            Размер выхода.

        Notes
        -----
        W и b инициализируются случайно.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.W = np.random.normal(0, 1, (out_features, in_features)) / np.sqrt(in_features + out_features + 1)
        self.b = np.random.normal(0, 1, (1, out_features)) / np.sqrt(in_features + out_features + 1)
        self.d = None
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = Wx + b.

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.
            То есть, либо x вектор с in_features элементов,
            либо матрица размерности (batch_size, in_features).

        Return
        ------
        y : np.ndarray
            Выход после слоя.
            Либо вектор с out_features элементами,
            либо матрица размерности (batch_size, out_features)

        """
        self.x = x
        return x.dot(self.W.T) + self.b

    def backward(self, d: np.ndarray) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        self.d = d
        return d.reshape(-1, self.out_features).dot(self.W)

    def update(self, alpha: float) -> NoReturn:
        """
        Обновляет W и b с заданной скоростью обучения.

        Parameters
        ----------
        alpha : float
            Скорость обучения.
        """
        self.W -= alpha * (self.x.reshape(self.in_features, -1).dot(self.d.reshape(-1, self.out_features))).T
        self.b -= alpha * self.d.sum(axis=0, keepdims=True)


class ReLU(Module):
    """
    Слой, соответствующий функции активации ReLU. Данная функция возвращает новый массив, в котором значения меньшие 0 заменены на 0.
    """

    def __init__(self):
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Возвращает y = max(0, x).

        Parameters
        ----------
        x : np.ndarray
            Входной вектор или батч.

        Return
        ------
        y : np.ndarray
            Выход после слоя (той же размерности, что и вход).

        """
        self.x = x
        return x * (x > 0)

    def backward(self, d) -> np.ndarray:
        """
        Cчитает градиент при помощи обратного распространения ошибки.

        Parameters
        ----------
        d : np.ndarray
            Градиент.
        Return
        ------
        np.ndarray
            Новое значение градиента.
        """
        return d * (self.x > 0)


# Task 2

class MLPClassifier:
    def __init__(self, modules: List[Module], epochs: int = 40, alpha: float = 0.01, batch_size: int = 32):
        """
        Parameters
        ----------
        modules : List[Module]
            Cписок, состоящий из ранее реализованных модулей и
            описывающий слои нейронной сети.
            В конец необходимо добавить Softmax.
        epochs : int
            Количество эпох обучения.
        alpha : float
            Cкорость обучения.
        batch_size : int
            Размер батча, используемый в процессе обучения.
        """
        self.modules = modules
        self.epochs = epochs
        self.alpha = alpha
        self.batch_size = batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Обучает нейронную сеть заданное число эпох.
        В каждой эпохе необходимо использовать cross-entropy loss для обучения,
        а так же производить обновления не по одному элементу, а используя батчи (иначе обучение будет нестабильным и полученные результаты будут плохими.

        Parameters
        ----------
        X : np.ndarray
            Данные для обучения.
        y : np.ndarray
            Вектор меток классов для данных.
        """
        for i in range(self.epochs):
            indexes = np.arange(0, X.shape[0] - 1)
            while True:
                if indexes.shape[0] == 0:
                    break
                if indexes.shape[0] > self.batch_size:
                    arr = np.random.choice(indexes, self.batch_size, replace=False)
                else:
                    arr = indexes
                x_in = X[arr]
                for module in self.modules:
                    x_in = module.forward(x_in)
                exps = np.exp(x_in - np.max(x_in, axis=1, keepdims=True))
                grad = exps / np.sum(exps, axis=1, keepdims=True)
                m = arr.shape[0]
                grad[range(m), np.array(y)[arr]] -= 1
                grad /= m
                for j in range(len(self.modules) - 1, -1, -1):
                    grad = self.modules[j].backward(grad)
                    self.modules[j].update(self.alpha)
                indexes = np.setdiff1d(np.unique(indexes), np.unique(arr))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Предсказывает вероятности классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Предсказанные вероятности классов для всех элементов X.
            Размерность (X.shape[0], n_classes)

        """
        x_in = X
        for module in self.modules:
            x_in = module.forward(x_in)
        exps = np.exp(x_in - np.max(x_in, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def predict(self, X) -> np.ndarray:
        """
        Предсказывает метки классов для элементов X.

        Parameters
        ----------
        X : np.ndarray
            Данные для предсказания.

        Return
        ------
        np.ndarray
            Вектор предсказанных классов

        """
        p = self.predict_proba(X)
        return np.argmax(p, axis=1)


# Task 3

classifier_moons = MLPClassifier(
    modules=[
        Linear(2, 70),
        ReLU(),
        Linear(70, 70),
        ReLU(),
        Linear(70, 2)
    ], epochs=100)  # Нужно указать гиперпараметры

classifier_blobs = MLPClassifier([
    Linear(2, 50),
    ReLU(),
    Linear(50, 50),
    ReLU(),
    Linear(50, 3)
], epochs=100)  # Нужно указать гиперпараметры


# Task 4

class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

    def load_model(self):
        """
        Используйте torch.load, чтобы загрузить обученную модель
        Учтите, что файлы решения находятся не в корне директории, поэтому необходимо использовать следующий путь:
        `__file__[:-7] +"model.pth"`, где "model.pth" - имя файла сохраненной модели `
        """
        path = __file__[:-7] + "model.pth"
        self.load_state_dict(torch.load(path))
        self.eval()

    def save_model(self):
        """
        Используйте torch.save, чтобы сохранить обученную модель
        """
        path = __file__[:-7] + "model.pth"
        torch.save(self.state_dict(), path)


def calculate_loss(X: torch.Tensor, y: torch.Tensor, model: TorchModel):
    """
    Cчитает cross-entropy.

    Parameters
    ----------
    X : torch.Tensor
        Данные для обучения.
    y : torch.Tensor
        Метки классов.
    model : Model
        Модель, которую будем обучать.

    """
    cel = nn.CrossEntropyLoss()
    y_pred = model(X)
    loss = cel(y_pred, y)
    return loss