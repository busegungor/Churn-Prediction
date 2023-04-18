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
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
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
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%.3f" %x)
pd.set_option("display.width", 500)
warnings.simplefilter(action="ignore", category=Warning)
df = pd.read_csv("/Users/busegungor/PycharmProjects/TelcoChurn/Telco-Customer-Churn.csv")
###################################
# Görev 1 : Keşifçi Veri Analizi
###################################
def general_picture(dataframe):
    print("Shape".center(70, "~"))
    print(dataframe.shape)
    print("Variables".center(70, "~"))
    print(dataframe.columns)
    print("Types".center(70, "~"))
    print(dataframe.dtypes)
    print("NA".center(70, "~"))
    print(dataframe.isnull().sum())
    print("Head".center(70, "~"))
    print(dataframe.head())
    print("Describe".center(70, "~"))
    print(dataframe.describe().T)
general_picture(df)

# Adım 2: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df["TotalCharges"] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df["Churn"] = [0 if x == "No" else 1 for x in df["Churn"]]

# Adım 1: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cats = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cats
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cats]
    print(f"Observation: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_car: {len(num_but_cats)}")
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)
# Adım 3: Numerik ve kategorik değişkenlerin veri içindeki dağılımını gözlemleyiniz.
def category_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("###########################")
    print(f"{col_name} : {dataframe[col_name].unique()}")
    print("###########################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=90)
        plt.figure(figsize=(14, 14))
        plt.show(block=True)

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        category_summary(df, col, plot=True)
    else:
        category_summary(df, col, plot=True)

def number_summary(dataframe, numberical_col, plot=False):
    quantiles = [0, 0.05, 0.95, 0.99, 1]
    print(dataframe[numberical_col].describe(quantiles).T)
    if plot:
        dataframe[numberical_col].hist(bins=15,ec='white')
        plt.xlabel(numberical_col)
        plt.title(f"Frequency of {numberical_col}")
        plt.show(block=True)

for col in num_cols:
    number_summary(df, col, plot=True)
# Adım 4: Kategorik değişkenler ile hedef değişken incelemesini yapınız.
def target_summary_with_cat(dataframe, target, categorical_col):
    if target == categorical_col:
        print("no")
    else:
        print(pd.DataFrame({"Target_Mean": dataframe.groupby(categorical_col)[target].mean()}))

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)

def target_summary_with_num(dataframe, target, numerical_col):
    if target == numerical_col:
        print("no")
    else:
        print(pd.DataFrame({"Target_Mean": dataframe.groupby(numerical_col)[target].mean()}))

for col in num_cols:
    target_summary_with_num(df, "Churn", col)

plots = {1: [111], 2: [121, 122], 3: [131, 132, 133], 4: [221, 222, 223, 224], 5: [231, 232, 233, 234, 235], 6: [231, 232, 233, 234, 235, 236]}
def boxplot(x, y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize=(7*columns, 7*rows))

    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.boxplot(x=x, y=j, data=df[[x, j]], palette="Blues", linewidth=1)
        ax.set_title(j)
    return plt.show()
boxplot("Churn", ["tenure", "MonthlyCharges", "TotalCharges"], df)

plt.style.use("fivethirtyeight")
def countplot(x, y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize=(7 * columns, 7 * rows))

    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=df, palette="Blues", linewidth=1, edgecolor="black")
        ax.set_title(j)
    return plt.show()
countplot("Churn", ["InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "MultipleLines"], df)

def countplot(x, y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize=(7 * columns, 7 * rows))

    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=df, palette="Blues", linewidth=1, edgecolor="black")
        ax.set_title(j)
    return plt.show()
countplot("Churn", ["Contract", "PaperlessBilling", "PaymentMethod"], df)

def countplot(x, y, df):
    rows = int(str(plots[len(y)][0])[0])
    columns = int(str(plots[len(y)][0])[1])
    plt.figure(figsize=(7 * columns, 7 * rows))

    for i, j in enumerate(y):
        plt.subplot(plots[len(y)][i])
        ax = sns.countplot(x=j, hue=x, data=df, palette="Blues", linewidth=1, edgecolor="black")
        ax.set_title(j)
    return plt.show()
countplot("Churn", ["Partner", "SeniorCitizen", "Dependents"], df)

# Adım 5: Aykırı gözlem var mı inceleyiniz.
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    # catch outliers
    q1 = dataframe[col_name].quantile(q1)
    q3 = dataframe[col_name].quantile(q3)
    interquartile_range = q3 - q1
    up_limit = q3 + 1.5 * interquartile_range
    low_limit = q1 - 1.5 * interquartile_range
    return low_limit, up_limit
def check_outlier(dataframe, col_name):
    # kod akışı sırasında outlier var mı yok mu kontrol et varsa bir görev yapmak istediğimizde yoksa bir görev yapmak istediğimizde yön verir.
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, check_outlier(df, col))
# Adım 6: Eksik gözlem var mı inceleyiniz.
df.isnull().sum() # TotalCharges        11

###################################
# Görev 2 : Feature Engineering
###################################

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.
# There is a little missing values which are 11. Prefer to dropna.
df.dropna(inplace=True)
# According to low limit and up limit, there are no values.
# Adım 2: Yeni değişkenler oluşturunuz.
df["Servicequality"] = (df[["InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "MultipleLines", "PhoneService", "StreamingTV", "StreamingMovies"]] == "Yes").sum(axis=1)
df.loc[(df["Servicequality"] <= 1), "Servicediff"] = "less"
df.loc[(df["Servicequality"] > 1) & (df["Servicequality"] <= 3), "Servicediff"] = "normal"
df.loc[(df["Servicequality"] > 3) & (df["Servicequality"] <= 5), "Servicediff"] = "high"
df.loc[(df["Servicequality"] > 5) & (df["Servicequality"] <= 8), "Servicediff"] = "abnormal"

df["Charges"] = df["MonthlyCharges"] / df["TotalCharges"]*100

df.loc[(df["tenure"] <= 9), "belonging"] = "less"
df.loc[(df["tenure"] > 9) & (df["tenure"] <= 29), "belonging"] = "normal"
df.loc[(df["tenure"] > 29) & (df["tenure"] <= 55), "belonging"] = "high"
df.loc[(df["tenure"] > 55) & (df["tenure"] <= 72), "belonging"] = "abnormal"


cat_cols, num_cols, cat_but_car = grab_col_names(df)
for col in num_cols:
    print(col, check_outlier(df, col))

df.columns = [col.lower() for col in df.columns]
df = df.drop("customerid", axis=1)
# Adım 3: Encoding işlemlerini gerçekleştiriniz.

# def rare_analyser(dataframe, target, cat_cols):
#     for col in cat_cols:
#         print(col, ":", len(dataframe[col].value_counts()))
#         print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
#                             "RATIO": dataframe[col].value_counts() / len(dataframe),
#                             "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
#
# rare_analyser(df, "churn", cat_cols)
#
# def rare_encoder(dataframe, rare_perc):
#     temp_df = dataframe.copy() # dataframe'in kopyası oluşturuldu.
#
#     rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O" and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
#     # rare değere sahip değişkenler seçildi.
#
#     for var in rare_columns:
#         # rare değişkenlerde gez.
#         tmp = temp_df[var].value_counts() / len(temp_df) # tem_df dataframe'i içerisinden ilgili değişkeni seçiyoruz. value_counts'ları alarak gözlem sayısına bölüyoruz. tmp adında yeni bir dataframe oluşturuyoruz.
#         rare_labels = tmp[tmp < rare_perc].index # bu dataframe de belirlediğimiz rare_perc oranının altında olan değerlerin index'ini rare_labels'a ata.
#         temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var]) # ana dataframe'de bu rare_labels varsa(isin) olduğu yere (where) "Rare" ata yoksa aynen bırak.
#
#     return temp_df
#
# df = rare_encoder(df, 0.01)
# rare_analyser(df, "churn", cat_cols)

feautes_ohe = ["multiplelines", "internetservice", "onlinesecurity", "onlinebackup", "deviceprotection", "techsupport", "streamingtv", "streamingmovies", "contract", "paymentmethod", "servicediff", "belonging"]
df = pd.get_dummies(df, columns=feautes_ohe, drop_first=True, dtype=int)

def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)
# Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
###################################
# Görev 3 : Modelleme
###################################

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz.
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve
# bulduğunuz hiparparametreler ile modeli tekrar kurunuz.