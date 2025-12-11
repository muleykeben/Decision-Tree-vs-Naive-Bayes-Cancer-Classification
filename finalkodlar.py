# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# =============================================================================
# Gerekli kütüphaneleri çağıralım ve veriyi okutalım 
# =============================================================================
import numpy as np
import pandas as pd

file_path = '/Users/cm/Desktop/Verı Madencılıgı/Final/cancer_data.xlsx'
data = pd.read_excel(file_path)


# =============================================================================
# VERİ HAKKINDA BİLGİ
#Veri hakkında bilgi: diagnosis değişkeni kategorik bağımlı değişkendir. İki düzeyi vardır: M=1, B=0 
# M kötü huylu kanseri temsil ederken B iyi huylu kanseri temsil eder. 
# =============================================================================


# =============================================================================
# Analiz için gereksiz olan değişkenleri veriden çıkaralım 
# =============================================================================

data = data.drop(columns = ['id','Column1'])

# =============================================================================
# Verideki değişkenler ve değişkenlerin türleri hakkında bilgi edinelim
# =============================================================================

data.info()
data.describe()

# =============================================================================
# Veriyi daha kolay görselleştirebilmek için 3 kısma ayırarak analize başlayalım
# =============================================================================

data1 = data.iloc[:,1:11]
data2 = data.iloc[:,11:21]
data3 = data.iloc[:,21:-1]
data1['diagnosis'] = data.diagnosis
data2['diagnosis'] = data.diagnosis
data3['diagnosis'] = data.diagnosis

# =============================================================================
# data1 için grafiklerimizi çizelim
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
fig,axs = plt.subplots(ncols =2 ,nrows =5 ,figsize = (35,40),dpi= 100)
axs = axs.flatten()
for i,col in enumerate(data1.drop(columns = 'diagnosis').columns) :
    sns.boxplot(x = col,y='diagnosis',data = data1 , ax=axs[i])
    axs[i].set_xlabel(col,fontsize=35)
    axs[i].tick_params(axis= 'x',labelsize = 35)
    axs[i].tick_params(axis= 'y',labelsize = 25)
    
plt.tight_layout()
plt.show()

fig,axs = plt.subplots(ncols =2 ,nrows =5 ,figsize = (35,40),dpi= 100)
axs = axs.flatten()
for i,col in enumerate(data1.drop(columns = 'diagnosis').columns) :
    sns.violinplot(x = col,y='diagnosis',data = data1, ax=axs[i])
    axs[i].set_xlabel(col,fontsize=35)
    axs[i].tick_params(axis= 'x',labelsize = 35)
    axs[i].tick_params(axis= 'y',labelsize = 25)
    
plt.tight_layout()
plt.show()

# =============================================================================
# data2 için grafikleri çizelim 
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
fig,axs = plt.subplots(ncols =2 ,nrows =5 ,figsize = (35,40),dpi= 100)
axs = axs.flatten()
for i,col in enumerate(data2.drop(columns = 'diagnosis').columns) :
    sns.boxplot(x = col,y='diagnosis',data = data2 , ax=axs[i])
    axs[i].set_xlabel(col,fontsize=35)
    axs[i].tick_params(axis= 'x',labelsize = 35)
    axs[i].tick_params(axis= 'y',labelsize = 25)
    
plt.tight_layout()
plt.show()

fig,axs = plt.subplots(ncols =2 ,nrows =5 ,figsize = (35,40),dpi= 100)
axs = axs.flatten()
for i,col in enumerate(data2.drop(columns = 'diagnosis').columns) :
    sns.violinplot(x = col,y='diagnosis',data = data2, ax=axs[i])
    axs[i].set_xlabel(col,fontsize=35)
    axs[i].tick_params(axis= 'x',labelsize = 35)
    axs[i].tick_params(axis= 'y',labelsize = 25)
    
plt.tight_layout()
plt.show()

# =============================================================================
# data3 için grafikleri çizelim 
# =============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
fig,axs = plt.subplots(ncols =2 ,nrows =5 ,figsize = (35,40),dpi= 100)
axs = axs.flatten()
for i,col in enumerate(data3.drop(columns = 'diagnosis').columns) :
    sns.boxplot(x = col,y='diagnosis',data = data3 , ax=axs[i])
    axs[i].set_xlabel(col,fontsize=35)
    axs[i].tick_params(axis= 'x',labelsize = 35)
    axs[i].tick_params(axis= 'y',labelsize = 25)
    
plt.tight_layout()
plt.show()

fig,axs = plt.subplots(ncols =2 ,nrows =5 ,figsize = (35,40),dpi= 100)
axs = axs.flatten()
for i,col in enumerate(data3.drop(columns = 'diagnosis').columns) :
    sns.violinplot(x = col,y='diagnosis',data = data3, ax=axs[i])
    axs[i].set_xlabel(col,fontsize=35)
    axs[i].tick_params(axis= 'x',labelsize = 35)
    axs[i].tick_params(axis= 'y',labelsize = 25)
    
plt.tight_layout()
plt.show()

# =============================================================================
# Bağımsız değişkenler arasında ilişki olup olmadığını araştırmak için ısı haritasına bakalım
# =============================================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
corr_matrix = data.drop(['diagnosis'], axis=1).corr()
mask = np.tril(np.ones_like(corr_matrix, dtype=bool))
plt.figure(figsize=(30, 20))
sns.heatmap(corr_matrix, cmap='viridis', annot=True, mask=mask)
plt.xticks(fontsize=25, rotation=90)
plt.yticks(fontsize=25, rotation=0)
plt.show()

# =============================================================================
# Bağımlı değişkenin düzeylerinin frekansına bakalım 
# =============================================================================

sns.countplot(x='diagnosis',hue='diagnosis',data = data)
data['diagnosis'].value_counts()
plt.show()

# =============================================================================
# Veriyi standartlaştıralım ve eğitim ile test kümesini ayıralım
# =============================================================================

from sklearn.model_selection import train_test_split
from sklearn import preprocessing,metrics
scaler = preprocessing.StandardScaler()
X = data.drop(columns = 'diagnosis').values
X = scaler.fit_transform(X)
Y = data['diagnosis'].values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

# =============================================================================
# Bayes ve Karar ağacı modellerini kuralım ve Roc eğrisi çizelim
# =============================================================================

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

models = [GaussianNB(), DecisionTreeClassifier(random_state=42)]
model_names = ['NaiveBayes', 'DecisionTree']
auc_scores = []

#Roc eğrisi çizme
for model,name in zip(models,model_names): 
    model.fit(x_train,y_train)
    y_pred_prob = model.predict_proba(x_test)[:, 1]
    fpr,tpr,thresholds = roc_curve(y_test,y_pred_prob)
    auc_score = auc(fpr,tpr)
    auc_scores.append(auc_score)
    plt.plot(fpr, tpr, label='%s (AUC = %0.2f)' % (name, auc_score))
    
plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# =============================================================================
# Bayes ve Karar Ağacı modelleri için F1-Recall- precision skorları 
# =============================================================================

from sklearn.metrics import precision_score, recall_score, f1_score
f1_scores = []
recall = []
precision = []

for model, name in zip(models, model_names):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = f1_score(y_test, y_pred)
    r = recall_score(y_test,y_pred)
    p = precision_score(y_test,y_pred)
    f1_scores.append(f1)
    recall.append(r)
    precision.append(p)
    print('%s: F1-score = %0.3f' % (name, f1))
    print('%s: Precision = %0.3f' % (name, p))
    print('%s: Recall = %0.3f' % (name, r))
    print('\n')
    
average_f1_score = sum(f1_scores) / len(f1_scores)
print('Average F1-score:', average_f1_score)

# =============================================================================
# Yorum: bayes sınıflandırmalarının f1 scoru ve precision değeri karar ağacına göre daha büyük olduğu için bayes modeli seçilir.
# =============================================================================

# =============================================================================
# Bayes modeli(nb) için accuracy, f1 skoru, classification report
# =============================================================================

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, f1_score

nb_model = GaussianNB()
nb_model.fit(x_train, y_train.ravel())
nb_y_pred = nb_model.predict(x_test)

print(f"Accuracy  : {accuracy_score(y_test, nb_y_pred):.2f} \n")
print(f"F1 Score  : {f1_score(y_test, nb_y_pred):.2f} \n")
print(f"Classification Report :\n{classification_report(y_test, nb_y_pred)} \n")

# Bayes için confusion matrix
cm_nb = confusion_matrix(y_test, nb_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=nb_model.classes_)
disp.plot(cmap='Blues')

nb_y_train_pred = nb_model.predict(x_train)
nb_train_accuracy = accuracy_score(y_train, nb_y_train_pred)
print(f"Training Accuracy: {nb_train_accuracy:.2f}")

nb_y_test_pred = nb_model.predict(x_test)
nb_test_accuracy = accuracy_score(y_test, nb_y_test_pred)
print(f"Test Accuracy: {nb_test_accuracy:.2f}")


# =============================================================================
# Karar ağaçları(dt) modeli için accuracy, f1 skoru, classification report
# =============================================================================

dt_model= DecisionTreeClassifier(random_state=42)
dt_model.fit(x_train, y_train.ravel())
dt_y_pred = dt_model.predict(x_test)

print(f"Accuracy  : {accuracy_score(y_test, dt_y_pred):.2f} \n")
print(f"F1 Score  : {f1_score(y_test, dt_y_pred):.2f} \n")
print(f"Classification Report :\n{classification_report(y_test, dt_y_pred)} \n")

dt_y_train_pred = dt_model.predict(x_train)
dt_train_accuracy = accuracy_score(y_train, dt_y_train_pred)
print(f"Training Accuracy: {dt_train_accuracy:.2f}")

dt_y_test_pred = dt_model.predict(x_test)
dt_test_accuracy = accuracy_score(y_test, dt_y_test_pred)
print(f"Test Accuracy: {dt_test_accuracy:.2f}")


# =============================================================================
# Karar ağacı grafiğini çizdirmek
# =============================================================================

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Değişkenlerin isimlerini feature_names değişkeninde kaydettik ki karar ağacında gözüksün
feature_names = [
    'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 
    'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 
    'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 
    'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 
    'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 
    'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

plt.figure(figsize=(16, 10))  # Grafik boyutunu ayarlayın
plot_tree(
    dt_model,
    feature_names=feature_names,  # Özellik isimlerini ekleyin
    class_names=['Class_0', 'Class_1'],  # Sınıf isimleri (örneğin, benign/malign)
    filled=True,  # Renkli düğümler
    rounded=True  # Yuvarlatılmış düğümler
)
plt.title("Karar Ağacı")
plt.show()



# =============================================================================
# Karar ağaçları modelinde aşırı öğrenme gözlendiği için hiperparametre optimizasyonu yapalım
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt

# Veriyi tekrar eğitim ve test olarak böldük çünkü %20 ile karar ağaçlarını çok iyi açıklayamamıştı. 
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

param_grid = {
    'max_depth': [3, 4, 5, 6, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6]
}

# GridSearchCV ile model optimizasyonu
grid_search = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,  # 5-Fold Cross-Validation
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# En iyi model
best_model = grid_search.best_estimator_
print("En iyi hiperparametreler:", grid_search.best_params_)

# Test setinde performansı değerlendir
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Eğitim kümesinin accuracy değerlendirme
y_train_pred= best_model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2f}")


# =============================================================================
# Hiperparametre optimizasyonu yapılmış haliyle tekrardan karar ağacı grafiği ve confusion matrix çizme
# =============================================================================

plt.figure(figsize=(16, 10))  # Grafik boyutunu ayarlayın
plot_tree(
    best_model,
    feature_names=feature_names,  # Özellik isimlerini ekleyin
    class_names=['Class_0', 'Class_1'],  # Sınıf isimleri (örneğin, benign/malign)
    filled=True,  # Renkli düğümler
    rounded=True  # Yuvarlatılmış düğümler
)
plt.title("Karar Ağacı")
plt.show()

#Karar ağacı confusion matrix çizelim
cm_dt = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=nb_model.classes_)
disp.plot(cmap='Blues')


# =============================================================================
# Yorum: Sonuç olarak karar ağaçları test accuracy %95, Bayes sınıflandırıcısı accuracy ise %96 olarak bulunmuştur. 
#Bayes modeli Karar Ağaçları modelinden daha iyi bir model olduğu görülmüştür. 
# =============================================================================











