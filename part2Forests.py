import warnings
import sys
import pandas as pd
import docx
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

f = open('r_forests_best.txt', 'w')
sys.stdout = f
f1_avg, p_avg, r_avg = 0, 0, 0
# warnings.filterwarnings("ignore")
precision_avg = 0
recall_avg = 0
f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall
avg_f1_all_list = []
avg_p_all_list = []
avg_r_all_list = []
curr = 0


def learn_dt(file_train, file_test, n_estimators, criterion,
             max_depth, min_samples_split, min_samples_leaf,
             min_weight_fraction_leaf, max_features, max_leaf_nodes,
             min_impurity_decrease, min_impurity_split, bootstrap,
             oob_score, n_jobs, random_state, verbose, warm_start,
             class_weight, ccp_alpha, max_samples):
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

    r_forests = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion,
                                       max_depth=max_depth, min_samples_split=min_samples_split,
                                       min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                       max_leaf_nodes=max_leaf_nodes,
                                       min_impurity_decrease=min_impurity_decrease,
                                       min_impurity_split=min_impurity_split, bootstrap=bootstrap,
                                       oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose,
                                       warm_start=warm_start,
                                       class_weight=class_weight, ccp_alpha=ccp_alpha, max_samples=max_samples)

    test_pred = r_forests.fit(train_data, train_target).predict(test_data)
    f1_num = round(f1_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    # print(f1_num)
    p_num = round(precision_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    r_num = round(recall_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    f1_list.append(f1_num)
    p_list.append(p_num)
    r_list.append(r_num)


def param_test(criterion, n_estimators, max_feat, max_d):
    global avg_f1_all_list, avg_p_all_list, avg_r_all_list
    avg_f1_all, avg_p_all, avg_r_all = 0, 0, 0
    global f1_avg, p_avg, r_avg
    global curr
    for i in range(0, 6):
        learn_dt("./data/jackrabbit/" + str(i) + "/train.csv", "./data/jackrabbit/" + str(i) + "/test.csv",
                 n_estimators=n_estimators, criterion=criterion, max_depth=max_d, min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=max_feat, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    f1_avg = sum(f1_list) / len(f1_list)
    p_avg = sum(p_list) / len(p_list)
    r_avg = sum(r_list) / len(r_list)

    avg_f1_all = avg_f1_all + f1_avg
    avg_p_all = avg_p_all + p_avg
    avg_r_all = avg_r_all + r_avg

    f1_avg, p_avg, r_avg = 0, 0, 0
    f1_list.clear()
    p_list.clear()
    r_list.clear()

    # jdt
    for i in range(0, 6):
        # print("Conducting tests on set " + str(i))
        learn_dt("./data/jdt/" + str(i) + "/train.csv", "./data/jdt/" + str(i) + "/test.csv",
                 n_estimators=n_estimators, criterion=criterion, max_depth=max_d, min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=max_feat, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    f1_avg = sum(f1_list) / len(f1_list)
    p_avg = sum(p_list) / len(p_list)
    r_avg = sum(r_list) / len(r_list)
    avg_f1_all = avg_f1_all + f1_avg
    avg_p_all = avg_p_all + p_avg
    avg_r_all = avg_r_all + r_avg

    f1_avg, p_avg, r_avg = 0, 0, 0
    f1_list.clear()
    p_list.clear()
    r_list.clear()

    # lucene
    for i in range(0, 6):
        learn_dt("./data/lucene/" + str(i) + "/train.csv", "./data/lucene/" + str(i) + "/test.csv",
                 n_estimators=n_estimators, criterion=criterion, max_depth=max_d, min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=max_feat, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    f1_avg = sum(f1_list) / len(f1_list)
    p_avg = sum(p_list) / len(p_list)
    r_avg = sum(r_list) / len(r_list)

    avg_f1_all = avg_f1_all + f1_avg
    avg_p_all = avg_p_all + p_avg
    avg_r_all = avg_r_all + r_avg

    f1_avg, p_avg, r_avg = 0, 0, 0
    f1_list.clear()
    p_list.clear()
    r_list.clear()

    # xorg
    for i in range(0, 6):
        # print("Conducting tests on set " + str(i))
        learn_dt("./data/xorg/" + str(i) + "/train.csv", "./data/xorg/" + str(i) + "/test.csv",
                 n_estimators=n_estimators, criterion=criterion, max_depth=max_d, min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.0, max_features=max_feat, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=42, verbose=0,
                 warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

    f1_avg = sum(f1_list) / len(f1_list)
    p_avg = sum(p_list) / len(p_list)
    r_avg = sum(r_list) / len(r_list)

    avg_f1_all = avg_f1_all + f1_avg
    avg_p_all = avg_p_all + p_avg
    avg_r_all = avg_r_all + r_avg

    f1_avg, p_avg, r_avg = 0, 0, 0
    f1_list.clear()
    p_list.clear()
    r_list.clear()

    avg_f1_all = (avg_f1_all / 4)
    avg_p_all = (avg_p_all / 4)
    avg_r_all = (avg_r_all / 4)

    avg_f1_all_list.append(round(avg_f1_all, 2))
    avg_p_all_list.append(round(avg_p_all, 2))
    avg_r_all_list.append(round(avg_r_all, 2))


doc = docx.Document("test.docx")


def to_doc(d_frame):
    # global doc
    t = doc.add_table(d_frame.shape[0] + 1, d_frame.shape[1])

    for j in range(df.shape[-1]):
        t.cell(0, j).text = df.columns[j]

    # add the rest of the data frame
    for i in range(df.shape[0]):
        for j in range(df.shape[-1]):
            t.cell(i + 1, j).text = str(df.values[i, j])
    doc.save("test.docx")


n_est = [1, 2, 5, 8, 10, 20, 30, 50, 80, 100]
n_depth = [1, 2, 5, 8, 10, 20, 30, 50, 80, 100]
n_feat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#
print("Using Gini")
for a in n_est:
    param_test("gini", a, None, None)
df = pd.DataFrame(list(zip(n_est, avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["n_estimators", "Precision", "Recall", "F1-Score"])
to_doc(df)
print(df)
avg_f1_all_list.clear()
avg_p_all_list.clear()
avg_r_all_list.clear()
print()
print("Using Entropy")
for a in n_est:
    param_test("entropy", a, None, None)
df = pd.DataFrame(list(zip(n_est, avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["n_estimators", "Precision", "Recall", "F1-Score"])

to_doc(df)
print(df)
avg_f1_all_list.clear()
avg_p_all_list.clear()
avg_r_all_list.clear()
print()
print("Using n_estimators = 5")
for a in n_depth:
    param_test("entropy", 5, None, a)
df = pd.DataFrame(list(zip(n_depth, avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["max_depth", "Precision", "Recall", "F1-Score"])

to_doc(df)
print(df)
avg_f1_all_list.clear()
avg_p_all_list.clear()
avg_r_all_list.clear()
print()
print("Using n_estimators = 5 and max_depth = 10")
for a in n_feat:
    param_test("entropy", 5, a, 10)
df = pd.DataFrame(list(zip(n_feat, avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["max_features", "Precision", "Recall", "F1-Score"])

to_doc(df)
print(df)
avg_f1_all_list.clear()
avg_p_all_list.clear()
avg_r_all_list.clear()

param_test("entropy", 5, 6, 10)
df = pd.DataFrame(list(zip(avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["Precision", "Recall", "F1-Score"])
to_doc(df)
print(df)
# print(max(avg_f1_all_list))
f.close()
