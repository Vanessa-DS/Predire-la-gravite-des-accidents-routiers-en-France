import pandas as pd
import numpy as np
import torch.nn as nn

from sklearn.base import BaseEstimator, TransformerMixin


class TransfoHour(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transfo = X.copy()
        X_transfo[f'{self.column_name}_sin'] = np.sin(np.multiply(X_transfo[self.column_name],(2.*np.pi/24)))
        X_transfo[f'{self.column_name}_cos'] = np.cos(np.multiply(X_transfo[self.column_name],(2.*np.pi/24)))
        X_transfo = X_transfo.drop(self.column_name, axis=1)
        return X_transfo
    
    def get_feature_names_out(self):
        pass
    
class TransfoMonth(BaseEstimator, TransformerMixin):
    def __init__(self, column_name):
        self.column_name = column_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transfo = X.copy()
        X_transfo[f'{self.column_name}_sin'] = np.sin((X_transfo[self.column_name]-1)*(2.*np.pi/12))
        X_transfo[f'{self.column_name}_cos'] = np.cos((X_transfo[self.column_name]-1)*(2.*np.pi/12))
        X_transfo = X_transfo.drop(self.column_name, axis=1)
        return X_transfo

    def get_feature_names_out(self):
        pass
    
class Multiclass(nn.Module):
    def __init__(self, activation=nn.ReLU, dropout_rate=0.1, n_neurones = 70):
        super().__init__()
        self.layer1 = nn.Linear(37, n_neurones)
        self.layer2 = activation()
        self.layer3 = nn.Linear(n_neurones, 2*n_neurones)
        self.layer4 = activation()
        self.layer5 = nn.Dropout(p=dropout_rate)
        self.layer6 = nn.Linear(2*n_neurones, n_neurones)
        self.layer7 = activation()
        self.layer8 = nn.Linear(n_neurones, int(0.5*n_neurones))
        self.layer9 = activation()
        self.layer10 = nn.Dropout(p=dropout_rate)
        self.layer11= nn.Linear(int(0.5*n_neurones), int(0.25*n_neurones))
        self.layer12= activation()
        self.layer13= nn.Linear(int(0.25*n_neurones), 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        return nn.functional.softmax(x, dim=1)