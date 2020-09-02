import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall

tp_list = []
fp_list = []
fn_list = []
tn_list = []


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
    train_target = dataset.loc[:, '500_Buggy?']

    # load the testing data
    dataset2 = pd.read_csv(file_test, header=0)

    # separate the data from the target attributes
    test_data = dataset2.drop('500_Buggy?', axis=1)

    # remove unnecessary features
    test_data = test_data.drop('change_id', axis=1)
    test_data = test_data.drop('412_full_path', axis=1)
    test_data = test_data.drop('411_commit_time', axis=1)

    # the lables of test data
    test_target = dataset2.loc[:, '500_Buggy?']

    gnb = GaussianNB()

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
    learn("../data/jackrabbit/" + str(i) + "/train_bow.csv", "../data/jackrabbit/" + str(i) + "/test_bow.csv")

print("Average precision, recall, and f1-score for 'jackrabbit'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(round(F1, 2))
p_list.append(round(precision, 2))
r_list.append(round(recall, 2))

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()

# jdt
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("../data/jdt/" + str(i) + "/train_bow.csv", "../data/jdt/" + str(i) + "/test_bow.csv")

print("\nPrecision, recall, and f1-score for 'jdt'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(round(F1, 2))
p_list.append(round(precision, 2))
r_list.append(round(recall, 2))

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()

# lucene
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("../data/lucene/" + str(i) + "/train_bow.csv", "../data/lucene/" + str(i) + "/test_bow.csv")

print("\nPrecision, recall, and f1-score for 'lucene'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(round(F1, 2))
p_list.append(round(precision, 2))
r_list.append(round(recall, 2))

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()

# xorg
for i in range(0, 6):
    # print("Conducting tests on set " + str(i))
    learn("../data/xorg/" + str(i) + "/train_bow.csv", "../data/xorg/" + str(i) + "/test_bow.csv")

print("\nPrecision, recall, and f1-score for 'xorg'\n")

total_tp = sum(tp_list)
total_fp = sum(fp_list)
total_fn = sum(fn_list)
total_tn = sum(tn_list)

precision = total_tp / (total_tp + total_fp)
recall = total_tp / (total_tp + total_fn)
F1 = (2 * precision * recall) / (precision + recall)
print("F1-Score =", round(F1, 2))
f1_list.append(round(F1, 2))
p_list.append(round(precision, 2))
r_list.append(round(recall, 2))

tp_list.clear()
fp_list.clear()
fn_list.clear()
tn_list.clear()

df = pd.DataFrame(list(zip(p_list, r_list, f1_list)),
                  columns=["Precision", "Recall", "F1-Score"])
print(df)
