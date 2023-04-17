######################################
# Telco Churn Prediction
######################################

#####################################
# Business Problem
#####################################
# Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli geliştirilmesi beklenmektedir.

#####################################
# Dataset History
#####################################
# Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve
# İnternet hizmetleri sağlayan hayali bir telekom şirketi hakkında bilgi içerir.
# Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu gösterir.

import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, validation_curve, RandomizedSearchCV
from skompiler import skompile

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


###################################
# TASKS
###################################

###################################
# Görev 1 : Keşifçi Veri Analizi
###################################

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
# Adım 5: Aykırı gözlem var mı inceleyiniz.
# Adım 6: Eksik gözlem var mı inceleyiniz.

###################################
# Görev 2 : Feature Engineering
###################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# Adım 2: Yeni değişkenler oluşturunuz.
# Adım 3: Encoding işlemlerini gerçekleştiriniz.
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.

###################################
# Görev 3 : Modelleme
###################################

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve
# bulduğunuz hiparparametreler ile modeli tekrar kurunuz.