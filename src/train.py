"""
Smile Detector — logistic regression classifier on grayscale face images.
"""

import glob
from PIL import Image
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import pickle

# ---------------------------------------------------------------------------
# 1. Data loading
# ---------------------------------------------------------------------------

import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
smile_path    = glob.glob(os.path.join(BASE_DIR, "data", "smile",     "*.jpg"))
no_smile_path = glob.glob(os.path.join(BASE_DIR, "data", "non_smile", "*.jpg"))

data=[]
output=[]

for i in smile_path:
    img=Image.open(i)
    img=img.convert('L')
    img=img.resize((64,64))
    img=np.array(img).flatten()
    data.append(img)
    output.append(1)


for i in no_smile_path:
    img=Image.open(i)
    img=img.convert('L')
    img=img.resize((64,64))
    img=np.array(img).flatten()
    data.append(img)
    output.append(0)

X=np.array(data)
y=np.array(output)


# ---------------------------------------------------------------------------
# 2. Model training
# ---------------------------------------------------------------------------

X,y=shuffle(X,y,random_state=42)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()

X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.transform(X_test)

regression=LogisticRegression()

regression.fit(X_train_scale,y_train)
y_predict=regression.predict(X_test_scale)

accuracy = accuracy_score(y_test,y_predict)

print("Baseline accuracy:", accuracy)

# ---------------------------------------------------------------------------
# 3.Hyperparameter search & Best model train
# --------------------------------------------------------------------------- 

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

params = {
    'C':        [0.01, 0.1, 1, 10, 100],
    'l1_ratio': [0, 1],   # 0 = l2, 1 = l1
    'solver':   ['liblinear'],
}

best_paras=GridSearchCV(regression,params,n_jobs=-1,cv=5)
best_paras.fit(X_train_scale,y_train)

best_paras.best_params_

best_paras.best_score_

best_model = LogisticRegression(C=0.01, l1_ratio=0)

best_model.fit(X_train_scale,y_train)
y_best=best_model.predict(X_test_scale)

print("Best model accuracy:", accuracy_score(y_test, y_best))

# ---------------------------------------------------------------------------
# 5.Save model & scaler
# --------------------------------------------------------------------------- 

MODELS_DIR  = os.path.join(BASE_DIR, "models")
model_path  = os.path.join(MODELS_DIR, "regression.pkl")
scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
 
with open(model_path, "wb") as f:
    pickle.dump(best_model, f)
 
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)
 
print(f"\n── Saved models ──────────────────────")
print(f"  Model  → {model_path}")
print(f"  Scaler → {scaler_path}")

# ---------------------------------------------------------------------------
# 4.Test
# --------------------------------------------------------------------------- 

test_img_path = os.path.join(BASE_DIR, "data", "test", "Abner_Martinez_0001.jpg")
img = Image.open(test_img_path) 
img=img.resize((64,64))
img=img.convert('L')
img_arr=np.array(img).flatten()
scale_arr=scaler.transform(img_arr.reshape(1,-1))
result = best_model.predict(scale_arr)
print("Prediction:", "Smile" if result[0] == 1 else "No Smile")