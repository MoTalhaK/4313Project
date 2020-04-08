import pandas as pd
import bow
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer

f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall

tp_list = []
fp_list = []
fn_list = []
tn_list = []

num_if = []


def get_if(data):
    for x in data:
        num_if.append(x[1])


def get_for(data):
    for x in data:
        num_if.append(x[2])


def learn(file_train, file_test):
    global f1_list, p_list, r_list
    global tp_list, fp_list, fn_list, tn_list

    # load the training data as a matrix
    dataset = pd.read_csv(file_train, header=0)

    # separate the data from the target attributes
    train_data = dataset.drop('500_Buggy?', axis=1)

    # remove unnecessary features
    train_data = train_data.drop('change_id', axis=1)
    train_data = train_data.drop('412_full_path', axis=1)
    train_data = train_data.drop('411_commit_time', axis=1)

    # the lables of training data. `label` is the title of the  last column in your CSV files
    train_target = dataset.iloc[:, -1]

    # load the testing data
    dataset2 = pd.read_csv(file_test, header=0)

    # separate the data from the target attributes
    test_data = dataset2.drop('500_Buggy?', axis=1)

    # remove unnecessary features
    test_data = test_data.drop('change_id', axis=1)
    test_data = test_data.drop('412_full_path', axis=1)
    test_data = test_data.drop('411_commit_time', axis=1)

    # the lables of test data
    test_target = dataset2.iloc[:, -1]

    gnb = GaussianNB()
    # classifier = Pipeline([
    #     ("features", FeatureUnion([
    #         ("num_if", Pipeline([
    #             ("count", FunctionTransformer(get_if, validate=False)),
    #         ])),
    #         ("num_for", Pipeline([
    #             ("count", FunctionTransformer(get_for, validate=False)),
    #         ]))
    #     ])),
    #     ("clf", GaussianNB())])

    test_pred = gnb.fit(train_data, train_target).predict(test_data)
    conf_matrix = confusion_matrix(test_target, test_pred, labels=[0, 1])
    TP = conf_matrix[0][0]
    FP = conf_matrix[0][1]
    FN = conf_matrix[1][0]
    TN = conf_matrix[1][1]

    tp_list.append(TP)
    fp_list.append(FP)
    fn_list.append(FN)
    tn_list.append(TN)


# jackrabbit
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/jackrabbit/" + str(i) + "/train.csv", "./data/jackrabbit/" + str(i) + "/test.csv")

print("Average precision, recall, and f1-score for 'jackrabbit'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(F1)
p_list.append(precision)
r_list.append(recall)

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()

# jdt
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/jdt/" + str(i) + "/train.csv", "./data/jdt/" + str(i) + "/test.csv")

print("\nAverage precision, recall, and f1-score for 'jdt'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(F1)
p_list.append(precision)
r_list.append(recall)

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()

# lucene
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/lucene/" + str(i) + "/train.csv", "./data/lucene/" + str(i) + "/test.csv")

print("\nAverage precision, recall, and f1-score for 'lucene'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(F1)
p_list.append(precision)
r_list.append(recall)

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()

# xorg
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/xorg/" + str(i) + "/train.csv", "./data/xorg/" + str(i) + "/test.csv")

print("\nAverage precision, recall, and f1-score for 'xorg'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(F1)
p_list.append(precision)
r_list.append(recall)

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()
print(f1_list)
