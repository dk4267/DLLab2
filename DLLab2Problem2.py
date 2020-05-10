import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import Normalizer
import seaborn as sns
import matplotlib.pyplot as plt

#read in the data
dataset = pd.read_csv("heart.csv", header=0)
ydata = dataset['target']
xdata = dataset.drop(['target'], axis=1)

#normalize data
scaler = Normalizer().fit(xdata)
normalizedX = scaler.transform(xdata)

#build and fit the model
model = LogisticRegression(solver='saga', C=1.0, random_state=0)
model.fit(normalizedX, ydata)

#find accuracy scores, build confusion matrix
p_pred = model.predict_proba(normalizedX)
y_pred = model.predict(normalizedX)
score = model.score(normalizedX, ydata)
conf_m = confusion_matrix(ydata, y_pred)
report = classification_report(ydata, y_pred)
print(score)
print(conf_m)
print(report)

#plot confusion matrix, which sort of shows loss - sorry for the bad formatting
plt.figure(figsize=(9, 9))
sns.heatmap(conf_m, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()

#initial accuracy: 0.64
#Change C value from 1.0 to 10.0: accuracy = 0.70
#change class weight from 'none' to 'balanced': accuracy = 0.66
#change solver to "saga": accuracy = 0.70
