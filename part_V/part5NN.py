import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K

f1_list = []  # list for f1-scores
p_list = []  # list for precision
r_list = []  # list for recall


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def learn(file_train, file_test):
    dataset = pd.read_csv(file_train, header=0)

    # separate the data from the target attributes
    train_data = dataset.drop('500_Buggy?', axis=1)

    # remove unnecessary features
    train_data = train_data.drop('change_id', axis=1)
    train_data = train_data.drop('412_full_path', axis=1)
    train_data = train_data.drop('411_commit_time', axis=1)

    # the lables of training data
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

    model = Sequential()
    model.add(Dense(12, input_dim=18, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1_m, precision_m, recall_m])

    # fit the model
    model.fit(train_data, train_target, epochs=150, batch_size=10)
    _, accuracy, f1_score, precision, recall = model.evaluate(train_data, train_target)
    f1_list.append(f1_score)
    p_list.append(precision)
    r_list.append(recall)
    print("Accuracy: %.2f" % (accuracy * 100))
    print()
    print("F1-Score: %.2f" % (f1_score * 100))
    print()
    print("Precision: %.2f" % (precision * 100))
    print()
    print("Recall: %.2f" % (recall * 100))


def param_test_jr():
    for i in range(0, 6):
        learn("../data/jackrabbit/" + str(i) + "/train_bow.csv", "../data/jackrabbit/" + str(i) + "/test_bow.csv")


param_test_jr()
f1_avg = sum(f1_list) / len(f1_list)
p_avg = sum(p_list) / len(p_list)
r_avg = sum(r_list) / len(r_list)
print("f1: %.2f" % (f1_avg * 100))
print()
print("precision: %.2f" % (p_avg * 100))
print()
print("recall: %.2f" % (r_avg * 100))
print()

f1_list.clear()
p_list.clear()
r_list.clear()