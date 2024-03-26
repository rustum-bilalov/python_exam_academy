#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # индекс признака, по которому разбиваем
        self.threshold = threshold  # порог разбиения
        self.left = left  # левое поддерево (меньше порога)
        self.right = right  # правое поддерево (больше или равно порогу)
        self.value = value  # значение в листовом узле (среднее значение целевой переменной в листе)

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth  # максимальная глубина дерева
        self.min_samples_split = min_samples_split  # минимальное количество образцов для разделения узла

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)
        
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        n_values = y.shape[0]

        # условия останова: если достигнута максимальная глубина или число образцов меньше минимального для разделения
        if depth == self.max_depth or n_values < self.min_samples_split:
            return Node(value=np.mean(y))

        # поиск наилучшего разделения
        best_feature_index, best_threshold = self._find_best_split(X, y)

        # условие останова: если не удается найти разделение
        if best_feature_index is None or best_threshold is None:
            return Node(value=np.mean(y))

        # разделение данных
        left_indices = np.where(X[:, best_feature_index] < best_threshold)[0]
        right_indices = np.where(X[:, best_feature_index] >= best_threshold)[0]

        # создание поддеревьев
        left = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature_index, threshold=best_threshold, left=left, right=right)

    def _find_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_mse = float('inf')
        best_feature_index = None
        best_threshold = None

        # перебор всех признаков
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            thresholds = np.unique(feature_values)

            # перебор всех уникальных значений признака в качестве порога разделения
            for threshold in thresholds:
                left_indices = np.where(feature_values < threshold)[0]
                right_indices = np.where(feature_values >= threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                # расчет среднеквадратичной ошибки для данного разделения
                left_mean = np.mean(y[left_indices])
                right_mean = np.mean(y[right_indices])
                mse = np.mean((y[left_indices] - left_mean) ** 2) + np.mean((y[right_indices] - right_mean) ** 2)

                # обновление лучшего разделения, если текущее лучше
                if mse < best_mse:
                    best_mse = mse
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def predict(self, X):
        return np.array([self._predict_tree(x, self.root) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] < node.threshold:
            return self._predict_tree(x, node.left)
        else:
            return self._predict_tree(x, node.right)

