import numpy as np
import csv as csv
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

f1_avg, p_avg, r_avg = 0, 0, 0
avg_f1_all, avg_p_all, avg_r_all = 0, 0, 0
precision_avg = 0
recall_avg = 0
f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall


def learn(file_train, file_test):
    global f1_list, p_list, r_list
    f1_num = 0
    p_num = 0
    r_num = 0
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
    test_pred = gnb.fit(train_data, train_target).predict(test_data)

    print(classification_report(test_target, test_pred, labels=[0, 1]))
    # print(round(f1_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2))
    f1_num = round(f1_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    p_num = round(precision_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    r_num = round(recall_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    f1_list.append(f1_num)
    p_list.append(p_num)
    r_list.append(r_num)


# jackrabbit
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/jackrabbit/" + str(i) + "/train.csv", "./data/jackrabbit/" + str(i) + "/test.csv")

print("Average precision, recall, and f1-score for 'jackrabbit'\n")
# average f1-score, precision, and recall for "jackrabbit" project
f1_avg = sum(f1_list) / len(f1_list)
p_avg = sum(p_list) / len(p_list)
r_avg = sum(r_list) / len(r_list)
print("F1-Score =", round(f1_avg, 2))
print("Precision =", round(p_avg, 2))
print("Recall =", round(r_avg, 2))
avg_f1_all = avg_f1_all + f1_avg
avg_p_all = avg_p_all + p_avg
avg_r_all = avg_r_all + r_avg
print(f1_list)

f1_avg, p_avg, r_avg = 0, 0, 0
f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall

# jdt
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/jdt/" + str(i) + "/train.csv", "./data/jdt/" + str(i) + "/test.csv")

print("\nAverage precision, recall, and f1-score for 'jdt'\n")
# average f1-score, precision, and recall for "jdt" project
f1_avg = sum(f1_list) / len(f1_list)
p_avg = sum(p_list) / len(p_list)
r_avg = sum(r_list) / len(r_list)
print("F1-Score =", round(f1_avg, 2))
print("Precision =", round(p_avg, 2))
print("Recall =", round(r_avg, 2))
avg_f1_all = avg_f1_all + f1_avg
avg_p_all = avg_p_all + p_avg
avg_r_all = avg_r_all + r_avg
print(f1_list)

f1_avg, p_avg, r_avg = 0, 0, 0
f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall

# lucene
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/lucene/" + str(i) + "/train.csv", "./data/lucene/" + str(i) + "/test.csv")

print("\nAverage precision, recall, and f1-score for 'lucene'\n")
# average f1-score, precision, and recall for "lucene" project
f1_avg = sum(f1_list) / len(f1_list)
p_avg = sum(p_list) / len(p_list)
r_avg = sum(r_list) / len(r_list)
print("F1-Score =", round(f1_avg, 2))
print("Precision =", round(p_avg, 2))
print("Recall =", round(r_avg, 2))
avg_f1_all = avg_f1_all + f1_avg
avg_p_all = avg_p_all + p_avg
avg_r_all = avg_r_all + r_avg
print(f1_list)

f1_avg, p_avg, r_avg = 0, 0, 0
f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall

# xorg
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("./data/xorg/" + str(i) + "/train.csv", "./data/xorg/" + str(i) + "/test.csv")

print("\nAverage precision, recall, and f1-score for 'xorg'\n")
# average f1-score, precision, and recall for "xorg" project
f1_avg = sum(f1_list) / len(f1_list)
p_avg = sum(p_list) / len(p_list)
r_avg = sum(r_list) / len(r_list)
print("F1-Score =", round(f1_avg, 2))
print("Precision =", round(p_avg, 2))
print("Recall =", round(r_avg, 2))
avg_f1_all = avg_f1_all + f1_avg
avg_p_all = avg_p_all + p_avg
avg_r_all = avg_r_all + r_avg
print(f1_list)

f1_avg, p_avg, r_avg = 0, 0, 0
f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall

print("\nAverage precision, recall, and f1-score across all the data sets\n")
avg_f1_all = (avg_f1_all / 4)
avg_p_all = (avg_p_all / 4)
avg_r_all = (avg_r_all / 4)
print("F1-Score =", round(avg_f1_all, 2))
print("Precision =", round(avg_p_all, 2))
print("Recall =", round(avg_r_all, 2))

print(f1_list)
