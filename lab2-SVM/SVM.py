import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [-1] * 20 + [1] * 20


df = pd.DataFrame(X,columns=['x1','x2'])
df['y'] = Y
df.head()

sns.scatterplot(df,x='x1',y='x2',hue='y')

svm = SVC(kernel='linear',random_state=42)
svm.fit(X,Y)

support_vectors = svm.support_vectors_
print("support_vectors:")
print(support_vectors)

coef = svm.coef_
intercept = svm.intercept_
print("coef:")
print(coef)

sns.scatterplot(df,x='x1',y='x2',hue='y')
sns.scatterplot(x=support_vectors[:,0],y=support_vectors[:,1])

k = coef[0,0]/(-coef[0,1])
b = intercept/(-coef[0,1])
hyperplane_x = [i-5 for i in range(9)]
hyperplane_y = [k*(i-5)+b for i in range(9)]
plt.plot(hyperplane_x,hyperplane_y)

