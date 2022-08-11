import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
import warnings
warnings.simplefilter(action="ignore")
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
# !pip install missingno
import missingno as msno
from sklearn import preprocessing

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("datasets/hitters.csv")
df.head()

def check_df(df, head=5):
    print("##################### Shape #####################")
    print(df.shape)

    print("##################### Types #####################")
    print(df.dtypes)

    print("##################### Head #####################")
    print(df.head(head))

    print("##################### Tail #####################")
    print(df.tail(head))

    print("##################### is null? #####################")
    print(df.isnull().sum())

    print("##################### Quantiles #####################")
    print(df.quantile([0, 0.25, 0.50, 0.75, 0.99, 1]).T)
    print(df.describe().T)

check_df(df)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.
    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optional
                kategorik fakat kardinal değişkenler için sınıf eşik değeri
    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi
    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))
    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Numerical Columns
df[num_cols].head()

# Categoric Columns
df[cat_cols].head()

# Kategorik değişken analizi
def cat_summary(df, col_name, plot=False):
    print(pd.DataFrame({col_name: df[col_name].value_counts(),
                        "Ratio": 100 * df[col_name].value_counts() / len(df)}))
    print("###################################")
    if plot:
        sns.countplot(x=df[col_name], data=df)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Numerik değişken analizi
def num_summary(df, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(df[numerical_col].describe(quantiles).T)

    if plot:
        df[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col, plot=True)

def target_summary_with_cat(df, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": df.groupby(categorical_col)[target].mean(),
                        "Count": df[categorical_col].value_counts(),
                        "Ratio": 100 * df[categorical_col].value_counts() / len(df)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Salary", col)

# Maaş hedef değişkeni bazında kategorik değişken görselleştirme
for col in cat_cols:
    graph=pd.crosstab(index=df['Salary'],columns=df[col]).plot.bar(figsize=(7,4), rot=0)
    plt.show()

# Numerik değişkene göre hedef değişken ortalaması

def target_summary_with_num(df, target, numerical_col):
    print(df.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Salary", col)


# DATA PREPROCESSING
# Outliers (Aykırı Değerler)
# Missing Values (Eksik Değerler)
# Feature Extraction (Özellik Çıkarımı)
# Encoding (Label Encoding, One-Hot Encoding, Rare Encoding)
# Feature Scaling (Özellik Ölçeklendirme)



# 1) Aykırı gözlem analizi (Outliers)
def outlier_thresholds(df, col_name, q1=0.10, q3=0.90):
    quartile1 = df[col_name].quantile(q1)
    quartile3 = df[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low_limit,up_limit = outlier_thresholds(df, "Salary")
print("Low Limit : {0}  Up Limit : {1}".format(low_limit,up_limit))

def check_outlier(df, col_name):
    low_limit, up_limit = outlier_thresholds(df, col_name)
    if df[(df[col_name] > up_limit) | (df[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col, ": ", check_outlier(df, col))


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))

#CHits :  True
#CHmRun :  True
#CWalks :  True

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

# Now, every variables are full, there is no outliers


# 2. Missing Values

df.isnull().sum()
#Salary       59

def missing_values_table(df, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]

    n_miss = df[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (df[na_columns].isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

import missingno as msno
msno.bar(df ,color="dodgerblue", sort="ascending");

df.dropna(inplace=True)

# we checked again and there is no missing value
missing_values_table(df)


#Korelasyon analizi

df[num_cols].corr()
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

# As we saw in the outputs, 59 amount of Salary values are NaN

dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)

scaler = RobustScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns=dff.columns)

imputer = KNNImputer(n_neighbors=5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns=dff.columns)
dff.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)
dff.head()

dff.isnull().sum()
# Salary         0

sns.boxplot(x = dff["Salary"])
plt.show()


# 3) Feature Extraction
df1 = df.copy()
df1.head()

# RUN RATE
df1["Run_Rate"]= df["HmRun"] / df["Runs"] + 0.00001
df1["AtBat_Hits_Rate"] = df["AtBat"] / df["Hits"] + 0.00001
df1["RBI_Rates"] = df["RBI"] / df["CRBI"] + 0.00001
df1["Yearly_Walks_Rate"] = df["Walks"] / df["Years"] + 0.00001
df1["Yearly_RBI"] = df["RBI"] / df["Years"] + 0.00001
df1["PutOuts_Rate"] = df["PutOuts"] / df["Years"] + 0.00001
df1["Hit_Rate"] = df["CAtBat"] / df["CHits"] + 0.00001
df1["Error_Mean"] = df["Errors"] / df["CWalks"] * 100


# 4) Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df1 = one_hot_encoder(df1, cat_cols, drop_first=True)
df1.head()

# 5) Feature Scaling

cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in ["Salary"]]
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()


# BASE MODEL PROCESS

y = df1["Salary"]
X = df1.drop("Salary", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=46)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

reg_model.intercept_
reg_model.coef_

# Train RMSE

y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# Output: 210.56

# TRAIN R-Square

reg_model.score(X_train, y_train)
# Output: 0.76

# Test RMSE

y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# Output: 343.50

# Test R-Square

reg_model.score(X_test, y_test)
# Output: 0.71

# 10 katlı Cross Validation RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

