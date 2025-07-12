# 🧠 Decision Trees vs Naive Bayes: Cancer Classification

This project compares the performance of **Decision Tree** and **Naive Bayes** classifiers on a breast cancer dataset. The analysis was conducted as part of the *Statistical Data Mining (İST405)* course at Hacettepe University.

## 📊 About the Dataset

The dataset contains diagnostic features of breast cancer patients and was obtained from [Kaggle](https://www.kaggle.com/). Each observation represents a patient with a binary diagnosis label:

- `diagnosis`: Target variable (B = Benign, M = Malignant → converted to 0 and 1)
- Features include: `radius_mean`, `texture_mean`, `area_mean`, `smoothness_mean`, etc.
- A total of 30 numerical predictor variables are included.
- <img width="440" height="681" alt="image" src="https://github.com/user-attachments/assets/c5c853f3-7ba0-402c-8453-106c7d8849e6" />


## ⚙️ Data Preprocessing

- Irrelevant variables were removed.
- No missing values were present in the dataset.
- Outliers were **not removed**, as they may represent critical real-world cases in a medical context.
- All features were standardized.
- The data was split into training and testing sets.

## 📈 Data Visualization

- The features were grouped into three parts: mean values, standard errors, and worst-case values.
- Boxplots and violin plots were used to explore distributions.
- <img width="427" height="181" alt="image" src="https://github.com/user-attachments/assets/646543de-6903-40a2-bfa8-f4f152c19e53" />
<img width="742" height="851" alt="image" src="https://github.com/user-attachments/assets/a5ba6065-4731-4739-9872-5b3deb647d48" />
<img width="409" height="143" alt="image" src="https://github.com/user-attachments/assets/467d770b-349d-4a5d-9123-ad1311ffbbb9" />
<img width="861" height="982" alt="image" src="https://github.com/user-attachments/assets/cd6f2110-320e-4099-97a3-075bf0d64d91" />
<img width="389" height="294" alt="image" src="https://github.com/user-attachments/assets/b401aade-660a-480f-8b48-811402719b4a" />
<img width="870" height="996" alt="image" src="https://github.com/user-attachments/assets/18d9655a-3088-43d1-8878-16e99e42e226" />
<img width="843" height="968" alt="image" src="https://github.com/user-attachments/assets/dda55ca3-27ec-4430-837f-d936a63ceff7" />
<img width="521" height="393" alt="image" src="https://github.com/user-attachments/assets/40f69ab8-bcea-46a3-84ff-91e5c434b60d" />
<img width="943" height="1081" alt="image" src="https://github.com/user-attachments/assets/b6a5b626-fa55-469e-ab9f-36ba1ea4723e" />
<img width="876" height="1002" alt="image" src="https://github.com/user-attachments/assets/3335303e-29d5-42b7-8539-ed19fabb8042" />
- A heatmap was generated to check for correlations among predictors.
<img width="665" height="214" alt="image" src="https://github.com/user-attachments/assets/6c3c482d-6cc5-4ef7-941f-6ea5f6309701" />
<img width="945" height="736" alt="image" src="https://github.com/user-attachments/assets/eb13830a-2f34-4420-bd0b-c54cce48cbd8" />

<img width="528" height="71" alt="image" src="https://github.com/user-attachments/assets/9f1bca4b-9820-43c3-9c8e-58f31d2200ac" />
<img width="574" height="125" alt="image" src="https://github.com/user-attachments/assets/3e2054fb-ef26-4b69-b4c3-73f5926a9a57" />


## 🤖 Model Building

### 1. **Naive Bayes Classifier**
- ROC curve showed strong classification performance.
- Outperformed the decision tree in all evaluation metrics.
- Accuracy: **96%** on the test set.
- No signs of overfitting.

### 2. **Decision Tree Classifier**
- Initial model showed signs of overfitting.
- Hyperparameter tuning (pruning) was applied to improve performance.
- Accuracy after tuning: **95%**
- Reduced tree complexity and overfitting.
- 
<img width="510" height="400" alt="image" src="https://github.com/user-attachments/assets/925f5f71-da70-4551-b26f-50f291a80132" />
<img width="452" height="333" alt="image" src="https://github.com/user-attachments/assets/2613a33b-52a9-425c-866b-5e046826c9fd" />

<img width="467" height="339" alt="image" src="https://github.com/user-attachments/assets/fa10a1f9-a49e-410f-8aea-1a28c5a5a2ae" />
<img width="327" height="223" alt="image" src="https://github.com/user-attachments/assets/ea4a4006-eb7a-40c2-9ba1-4ddddcadb806" />
<img width="681" height="352" alt="image" src="https://github.com/user-attachments/assets/73dc5cf9-4add-44c3-a680-3b255e473713" />
<img width="326" height="213" alt="image" src="https://github.com/user-attachments/assets/e4f25106-5314-4851-a5e6-c58bc3da496e" />
<img width="315" height="119" alt="image" src="https://github.com/user-attachments/assets/a28850ce-d0a9-4ceb-99fc-44ad4ccbe56e" />
<img width="465" height="247" alt="image" src="https://github.com/user-attachments/assets/8d972497-7447-4e80-9c25-84d8c11e77db" />
<img width="355" height="243" alt="image" src="https://github.com/user-attachments/assets/ee36e7b9-6096-47c6-815d-53d3ae776a60" />
<img width="460" height="306" alt="image" src="https://github.com/user-attachments/assets/7a1a401a-865f-4066-8118-dfc775a6ff41" />
<img width="945" height="607" alt="image" src="https://github.com/user-attachments/assets/0d6e22e9-16f4-4f0f-a18a-48d3d99cdf5f" />
<img width="612" height="569" alt="image" src="https://github.com/user-attachments/assets/1df3ce0d-25d0-4aae-9057-fa89066bf194" />
<img width="684" height="285" alt="image" src="https://github.com/user-attachments/assets/5b104bb5-1b5f-4a43-ae9f-23a180ed57e8" />
<img width="267" height="102" alt="image" src="https://github.com/user-attachments/assets/dd556642-daaf-4cc9-9f39-f5e615048eaa" />

## 🧪 Evaluation Results

| Model              | Accuracy | Notes                                      |
|-------------------|----------|--------------------------------------------|
| Naive Bayes        | 96%      | More reliable predictions, lower variance  |
| Decision Tree      | 95%      | Improved after hyperparameter optimization |

Overall, the Naive Bayes classifier performed slightly better than the Decision Tree model.

<img width="945" height="607" alt="image" src="https://github.com/user-attachments/assets/cc511560-3c1f-4cf7-a0cc-98ebcf36a658" />

YORUM: Sonuç olarak Bayes Sınıflandırıcıları ve Karar Ağaçları modelleri karşılaştırıldığında %96’lık doğruluk oranı ile Bayes Sınıflandırıcılarının, %95’lik doğruluk oranı ile Karar Ağaçları modelinden daha iyi ve daha az hatalı bir model olduğu gözlemlenmiştir. 
## 📚 References

- [Python Graph Gallery – Heatmaps](https://python-graph-gallery.com/heatmap/)
- [Python Graph Gallery – Violin Plots](https://python-graph-gallery.com/violin-plot/)
- [Google Developers – ROC and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=tr)

## ✍️ Authors

- Müleyke Önal – [@your_github_username](https://github.com/your_github_username)  
- Cem Molla

---

