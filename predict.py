import pandas as pd
from sklearn.svm import SVC
import util
from featurize import add_features

# Gather training data
lines_train = [add_features(line) for line in util.parse_training_data()]
lines_test = [add_features(line) for line in util.parse_testing_data()]

df_train = pd.DataFrame.from_dict(lines_train)
df_test = pd.DataFrame.from_dict(lines_test)

# Get rid of columns not needed
del df_train["?"]
del df_train["chinese"]
del df_train["human"]
del df_train["h_tree"]
del df_train["q_tree"]
del df_train["bleu"]
del df_test["?"]
del df_test["chinese"]
del df_test["human"]
del df_test["h_tree"]
del df_test["q_tree"]
del df_test["bleu"]

# Map labels to numbers
label_map = {"H" : 0, "M" : 1}
df_train["label"] = df_train["label"].map(label_map)

feature_col_names = list(df_train.columns.values)
feature_col_names.remove('label')

predicted_class_names = ['label']

X_train = df_train[feature_col_names].values
y_train = df_train[predicted_class_names].values
X_test = df_test[feature_col_names].values

svc_model = SVC()
svc_model.fit(X_train, y_train.ravel())
svc_predict_test = svc_model.predict(X_test)

for i in range(len(lines_test)):
    print(lines_test[i]["chinese"])
    print(lines_test[i]["human"])
    print(lines_test[i]["?"])
    print(lines_test[i]["bleu"])
    if svc_predict_test[i] == 0:
        print("H")
    else:
        print("M")
    print()