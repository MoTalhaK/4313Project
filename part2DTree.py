import warnings
import sys
import pandas as pd
import docx
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV

f = open('dt_out.txt', 'w')
sys.stdout = f
f1_avg, p_avg, r_avg = 0, 0, 0
warnings.filterwarnings("ignore")
precision_avg = 0
recall_avg = 0
f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall
avg_f1_all_list = []
avg_p_all_list = []
avg_r_all_list = []


def learn_dt(file_train, file_test, criterion, split,
             max_depth, min_samp_split, min_samp_leaf, min_weight_fraction,
             max_feat, rand_state, max_leaf_nodes, min_impurity_dec,
             class_w, ccp_alpha):
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

    decision_t = DecisionTreeClassifier(criterion=criterion, splitter=split,
                                        max_depth=max_depth, min_samples_split=min_samp_split,
                                        min_samples_leaf=min_samp_leaf, min_weight_fraction_leaf=min_weight_fraction,
                                        max_features=max_feat, random_state=rand_state, max_leaf_nodes=max_leaf_nodes,
                                        min_impurity_decrease=min_impurity_dec, class_weight=class_w,
                                        ccp_alpha=ccp_alpha)
    test_pred = decision_t.fit(train_data, train_target).predict(test_data)
    f1_num = round(f1_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    p_num = round(precision_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    r_num = round(recall_score(test_target, test_pred, labels=[0, 1], average='weighted'), 2)
    f1_list.append(f1_num)
    p_list.append(p_num)
    r_list.append(r_num)


def param_test(criterion, split, max_depth, max_feat):
    global avg_f1_all_list, avg_p_all_list, avg_r_all_list
    avg_f1_all, avg_p_all, avg_r_all = 0, 0, 0
    global f1_avg, p_avg, r_avg
    # jackrabbit
    for i in range(0, 6):
        learn_dt("./data/jackrabbit/" + str(i) + "/train.csv", "./data/jackrabbit/" + str(i) + "/test.csv",
                 criterion=criterion, split=split, max_depth=max_depth, min_samp_split=2,
                 min_samp_leaf=1, min_weight_fraction=0.0, max_feat=max_feat, rand_state=45,
                 max_leaf_nodes=None, min_impurity_dec=0.0, class_w=None, ccp_alpha=0.0)

    # average f1-score, precision, and recall for "jackrabbit" project
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
        learn_dt("./data/jdt/" + str(i) + "/train.csv", "./data/jdt/" + str(i) + "/test.csv",
                 criterion=criterion, split=split, max_depth=max_depth, min_samp_split=2,
                 min_samp_leaf=1, min_weight_fraction=0.0, max_feat=max_feat, rand_state=45,
                 max_leaf_nodes=None, min_impurity_dec=0.0, class_w=None, ccp_alpha=0.0)

    # average f1-score, precision, and recall for "jdt" project
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
                 criterion=criterion, split=split, max_depth=max_depth, min_samp_split=2,
                 min_samp_leaf=1, min_weight_fraction=0.0, max_feat=max_feat, rand_state=45,
                 max_leaf_nodes=None, min_impurity_dec=0.0, class_w=None, ccp_alpha=0.0)

    # average f1-score, precision, and recall for "lucene" project
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
        learn_dt("./data/xorg/" + str(i) + "/train.csv", "./data/xorg/" + str(i) + "/test.csv",
                 criterion=criterion, split=split, max_depth=max_depth, min_samp_split=2,
                 min_samp_leaf=1, min_weight_fraction=0.0, max_feat=max_feat, rand_state=45,
                 max_leaf_nodes=None, min_impurity_dec=0.0, class_w=None, ccp_alpha=0.0)

    # average f1-score, precision, and recall for "xorg" project
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
    print()


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


n_depth = [1, 2, 5, 8, 10, 20, 30, 50, 80, 100]
n_feat = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

print("Using 'gini'")
for a in n_depth:
    param_test("gini", "best", a, None)
df = pd.DataFrame(list(zip(n_depth, avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["max_depth", "Precision", "Recall", "F1-Score"])
to_doc(df)
print(df)
avg_f1_all_list.clear()
avg_p_all_list.clear()
avg_r_all_list.clear()

print("Using 'entropy'")
for a in n_depth:
    param_test("entropy", "best", a, None)
df = pd.DataFrame(list(zip(n_depth, avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["max_depth", "Precision", "Recall", "F1-Score"])
to_doc(df)
print(df)
avg_f1_all_list.clear()
avg_p_all_list.clear()
avg_r_all_list.clear()

print("Using Criteria = 'entropy', splitter = 'best', and max_depth = 20 ")
for a in n_feat:
    param_test("entropy", "best", 20, a)
df = pd.DataFrame(list(zip(n_feat, avg_p_all_list, avg_r_all_list, avg_f1_all_list)),
                  columns=["max_features", "Precision", "Recall", "F1-Score"])
to_doc(df)
print(df)
avg_f1_all_list.clear()
avg_p_all_list.clear()
avg_r_all_list.clear()

f.close()
