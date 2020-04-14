from nltk.corpus import stopwords
import nltk
# nltk.download("stopwords")
import string
import numpy
import sys
import pandas as pd
from os import listdir
import os
import shutil
from collections import Counter
from keras.preprocessing.text import Tokenizer


def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# load file
def load_file(f_name):
    f = open(f_name, "r", errors="ignore")
    first = ""
    second = ""
    file_text = []
    while True:
        line = f.readline()

        if line.startswith("+") and line.count("+") == 1:
            # first = line
            file_text.append(line)
        if line.startswith("-") and line.count("-") == 1:
            # second = line
            file_text.append(line)
        if not line:
            break
    f.close()
    return file_text


def clean_file(doc):
    tokens = ""
    list_tok = []
    all_stop_words = stopwords.words("english")
    all_stop_words.remove("for")
    all_stop_words.remove("while")
    all_stop_words.remove("if")
    for i in doc:
        tokens = i.split()
        list_tok.append(tokens)
    flat_list = [item for sublist in list_tok for item in sublist]
    # result = sorted(set(map(tuple, list_tok)), reverse=True)
    stop_words = {"int", "double", "boolean", "String", "class", "public", "void", "private", "protected", "+", "-"
        , "char", "extends", "final", "float", "implements", "import", "interface", "long"
        , "new", "package", "short", "super", "void", "synchronized", "abstract", "static", "*"
        , "*/", "/**", "+import", "-import", "}", "{", "//", "@param", "License", "file", "distributed"
        , "node", "The", "may", "@throws", "See", "Apache", "You", "License.", "@return", "e)", "@see"
        , "name", "{@link", "null)", "one", "use", "2.0", "Version", "path", "required", "either", "specific"
        , "copy", "permissions", "information", '"', "applicable", "except", "language", "(the", "additional"
        , "obtain", "OR", "software"}
    stop_words_eng = set(all_stop_words)
    tokens = [w for w in flat_list if not w in stop_words]
    tokens = [w for w in tokens if not w in stop_words_eng]
    # # result = map(list, sorted(set(map(tuple, tokens)), reverse=True))
    return tokens


def add_file_to_vocab(f_name, vocab):
    doc = load_file(f_name)
    tokens = clean_file(doc)
    vocab.update(tokens)


def process_files(directory, vocab):
    for f_name in listdir(directory):
        # print(f_name)
        path = directory + "/" + f_name
        add_file_to_vocab(path, vocab)


def save_list(lines, f_name):
    data = "\n".join(lines)
    file = open(f_name, "w")
    file.write(data)
    file.close()


# filename = "./data/jackrabbit/patch/jackrabbit-35621.patch"
# text = load_file(filename)
# print(text)
# tok = clean_file(text)
# print(tok)

# vocab = Counter()
# # add all files to vocab
# process_files("./data/jackrabbit/patch", vocab)
# # size of vocab
# print(len(vocab))
# # top words
# print(vocab.most_common(50))
# rev_vocab = dict()
# for key, value in vocab.items():
#     if key == "for":
#         rev_vocab.update({key: value})
#         # print(key, value)
#     if key == "if":
#         rev_vocab.update({key: value})
#     if key == "else":
#         rev_vocab.update({key: value})
#     if key == "while":
#         rev_vocab.update({key: value})
#     if key == "switch":
#         rev_vocab.update({key: value})
# print(rev_vocab)
# save_list(rev_vocab, "vocab.txt")

# use vocab.txt to generate new tokens which we will then use to form our bag of words

def doc_to_line(f_name, vocab):
    file = load_file(f_name)
    tokens = clean_file(file)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def process_docs(directory, data, vocab):
    lines = list()
    dataset = pd.read_csv(data, header=0)
    # iterate through files in folder
    for filename in listdir(directory):
        first_index = filename.index("-")
        second_index = filename.index(".")
        patch_num = filename[first_index + 1: second_index]
        if (dataset.get("change_id") == int(patch_num)).any():
            # print("yes")
            path = directory + "/" + filename
            line = doc_to_line(path, vocab)
            lines.append(line)
    return lines


vocab_filename = "vocab.txt"
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
docs_rabbit = list()
docs_jdt = list()
docs_lucene = list()
docs_xorg = list()

jackrabbit0 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/0/train.csv", vocab)
jackrabbit1 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/1/train.csv", vocab)
jackrabbit2 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/2/train.csv", vocab)
jackrabbit3 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/3/train.csv", vocab)
jackrabbit4 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/4/train.csv", vocab)
jackrabbit5 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/5/train.csv", vocab)

jackrabbit_test0 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/0/test.csv", vocab)
jackrabbit_test1 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/1/test.csv", vocab)
jackrabbit_test2 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/2/test.csv", vocab)
jackrabbit_test3 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/3/test.csv", vocab)
jackrabbit_test4 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/4/test.csv", vocab)
jackrabbit_test5 = process_docs("./data/jackrabbit/patch", "./data/jackrabbit/5/test.csv", vocab)

jdt0 = process_docs("./data/jdt/patch", "./data/jdt/0/train.csv", vocab)
jdt1 = process_docs("./data/jdt/patch", "./data/jdt/1/train.csv", vocab)
jdt2 = process_docs("./data/jdt/patch", "./data/jdt/2/train.csv", vocab)
jdt3 = process_docs("./data/jdt/patch", "./data/jdt/3/train.csv", vocab)
jdt4 = process_docs("./data/jdt/patch", "./data/jdt/4/train.csv", vocab)
jdt5 = process_docs("./data/jdt/patch", "./data/jdt/5/train.csv", vocab)

jdt_test0_docs = process_docs("./data/jdt/patch", "./data/jdt/0/test.csv", vocab)
jdt_test1_docs = process_docs("./data/jdt/patch", "./data/jdt/1/test.csv", vocab)
jdt_test2_docs = process_docs("./data/jdt/patch", "./data/jdt/2/test.csv", vocab)
jdt_test3_docs = process_docs("./data/jdt/patch", "./data/jdt/3/test.csv", vocab)
jdt_test4_docs = process_docs("./data/jdt/patch", "./data/jdt/4/test.csv", vocab)
jdt_test5_docs = process_docs("./data/jdt/patch", "./data/jdt/5/test.csv", vocab)

lucene0 = process_docs("./data/lucene/patch", "./data/lucene/0/train.csv", vocab)
lucene1 = process_docs("./data/lucene/patch", "./data/lucene/1/train.csv", vocab)
lucene2 = process_docs("./data/lucene/patch", "./data/lucene/2/train.csv", vocab)
lucene3 = process_docs("./data/lucene/patch", "./data/lucene/3/train.csv", vocab)
lucene4 = process_docs("./data/lucene/patch", "./data/lucene/4/train.csv", vocab)
lucene5 = process_docs("./data/lucene/patch", "./data/lucene/5/train.csv", vocab)

lucene_test0_docs = process_docs("./data/lucene/patch", "./data/lucene/0/test.csv", vocab)
lucene_test1_docs = process_docs("./data/lucene/patch", "./data/lucene/1/test.csv", vocab)
lucene_test2_docs = process_docs("./data/lucene/patch", "./data/lucene/2/test.csv", vocab)
lucene_test3_docs = process_docs("./data/lucene/patch", "./data/lucene/3/test.csv", vocab)
lucene_test4_docs = process_docs("./data/lucene/patch", "./data/lucene/4/test.csv", vocab)
lucene_test5_docs = process_docs("./data/lucene/patch", "./data/lucene/5/test.csv", vocab)

xorg0 = process_docs("./data/xorg/patch", "./data/xorg/0/train.csv", vocab)
xorg1 = process_docs("./data/xorg/patch", "./data/xorg/1/train.csv", vocab)
xorg2 = process_docs("./data/xorg/patch", "./data/xorg/2/train.csv", vocab)
xorg3 = process_docs("./data/xorg/patch", "./data/xorg/3/train.csv", vocab)
xorg4 = process_docs("./data/xorg/patch", "./data/xorg/4/train.csv", vocab)
xorg5 = process_docs("./data/xorg/patch", "./data/xorg/5/train.csv", vocab)

xorg_test0_docs = process_docs("./data/xorg/patch", "./data/xorg/0/test.csv", vocab)
xorg_test1_docs = process_docs("./data/xorg/patch", "./data/xorg/1/test.csv", vocab)
xorg_test2_docs = process_docs("./data/xorg/patch", "./data/xorg/2/test.csv", vocab)
xorg_test3_docs = process_docs("./data/xorg/patch", "./data/xorg/3/test.csv", vocab)
xorg_test4_docs = process_docs("./data/xorg/patch", "./data/xorg/4/test.csv", vocab)
xorg_test5_docs = process_docs("./data/xorg/patch", "./data/xorg/5/test.csv", vocab)

tokenizer = Tokenizer()

docs_rabbit_test = list()
docs_jdt_test = list()
docs_lucene_test = list()
docs_xorg_test = list()

# tokenize the output obtained from 'process_docs' and
# get count of our specified vocabulary (if, for, else, while, switch) and
# return as a numpy matrix
def return_matrix(data):
    tokenizer.fit_on_texts(data)
    return tokenizer.texts_to_matrix(data, mode="count")


j_train0 = return_matrix(jackrabbit0)
j_train1 = return_matrix(jackrabbit1)
j_train2 = return_matrix(jackrabbit2)
j_train3 = return_matrix(jackrabbit3)
j_train4 = return_matrix(jackrabbit4)
j_train5 = return_matrix(jackrabbit5)

j_test0 = return_matrix(jackrabbit_test0)
j_test1 = return_matrix(jackrabbit_test1)
j_test2 = return_matrix(jackrabbit_test2)
j_test3 = return_matrix(jackrabbit_test3)
j_test4 = return_matrix(jackrabbit_test4)
j_test5 = return_matrix(jackrabbit_test5)

jdt_train0 = return_matrix(jdt0)
jdt_train1 = return_matrix(jdt1)
jdt_train2 = return_matrix(jdt2)
jdt_train3 = return_matrix(jdt3)
jdt_train4 = return_matrix(jdt4)
jdt_train5 = return_matrix(jdt5)

jdt_test0 = return_matrix(jdt_test0_docs)
jdt_test1 = return_matrix(jdt_test1_docs)
jdt_test2 = return_matrix(jdt_test2_docs)
jdt_test3 = return_matrix(jdt_test3_docs)
jdt_test4 = return_matrix(jdt_test4_docs)
jdt_test5 = return_matrix(jdt_test5_docs)

lucene_train0 = return_matrix(lucene0)
lucene_train1 = return_matrix(lucene1)
lucene_train2 = return_matrix(lucene2)
lucene_train3 = return_matrix(lucene3)
lucene_train4 = return_matrix(lucene4)
lucene_train5 = return_matrix(lucene5)

lucene_test0 = return_matrix(lucene_test0_docs)
lucene_test1 = return_matrix(lucene_test1_docs)
lucene_test2 = return_matrix(lucene_test2_docs)
lucene_test3 = return_matrix(lucene_test3_docs)
lucene_test4 = return_matrix(lucene_test4_docs)
lucene_test5 = return_matrix(lucene_test5_docs)

xorg_train0 = return_matrix(xorg0)
xorg_train1 = return_matrix(xorg1)
xorg_train2 = return_matrix(xorg2)
xorg_train3 = return_matrix(xorg3)
xorg_train4 = return_matrix(xorg4)
xorg_train5 = return_matrix(xorg5)

xorg_test0 = return_matrix(xorg_test0_docs)
xorg_test1 = return_matrix(xorg_test1_docs)
xorg_test2 = return_matrix(xorg_test2_docs)
xorg_test3 = return_matrix(xorg_test3_docs)
xorg_test4 = return_matrix(xorg_test4_docs)
xorg_test5 = return_matrix(xorg_test5_docs)

############################################################
jrtrain0 = pd.read_csv("./data/jackrabbit/0/train.csv")
jrtrain1 = pd.read_csv("./data/jackrabbit/1/train.csv")
jrtrain2 = pd.read_csv("./data/jackrabbit/2/train.csv")
jrtrain3 = pd.read_csv("./data/jackrabbit/3/train.csv")
jrtrain4 = pd.read_csv("./data/jackrabbit/4/train.csv")
jrtrain5 = pd.read_csv("./data/jackrabbit/5/train.csv")

jrtest0 = pd.read_csv("./data/jackrabbit/0/test.csv")
jrtest1 = pd.read_csv("./data/jackrabbit/1/test.csv")
jrtest2 = pd.read_csv("./data/jackrabbit/2/test.csv")
jrtest3 = pd.read_csv("./data/jackrabbit/3/test.csv")
jrtest4 = pd.read_csv("./data/jackrabbit/4/test.csv")
jrtest5 = pd.read_csv("./data/jackrabbit/5/test.csv")

jdtrain0 = pd.read_csv("./data/jdt/0/train.csv")
jdtrain1 = pd.read_csv("./data/jdt/1/train.csv")
jdtrain2 = pd.read_csv("./data/jdt/2/train.csv")
jdtrain3 = pd.read_csv("./data/jdt/3/train.csv")
jdtrain4 = pd.read_csv("./data/jdt/4/train.csv")
jdtrain5 = pd.read_csv("./data/jdt/5/train.csv")

jdtest0 = pd.read_csv("./data/jdt/0/test.csv")
jdtest1 = pd.read_csv("./data/jdt/1/test.csv")
jdtest2 = pd.read_csv("./data/jdt/2/test.csv")
jdtest3 = pd.read_csv("./data/jdt/3/test.csv")
jdtest4 = pd.read_csv("./data/jdt/4/test.csv")
jdtest5 = pd.read_csv("./data/jdt/5/test.csv")

luctrain0 = pd.read_csv("./data/lucene/0/train.csv")
luctrain1 = pd.read_csv("./data/lucene/1/train.csv")
luctrain2 = pd.read_csv("./data/lucene/2/train.csv")
luctrain3 = pd.read_csv("./data/lucene/3/train.csv")
luctrain4 = pd.read_csv("./data/lucene/4/train.csv")
luctrain5 = pd.read_csv("./data/lucene/5/train.csv")

luctest0 = pd.read_csv("./data/lucene/0/test.csv")
luctest1 = pd.read_csv("./data/lucene/1/test.csv")
luctest2 = pd.read_csv("./data/lucene/2/test.csv")
luctest3 = pd.read_csv("./data/lucene/3/test.csv")
luctest4 = pd.read_csv("./data/lucene/4/test.csv")
luctest5 = pd.read_csv("./data/lucene/5/test.csv")

xtrain0 = pd.read_csv("./data/xorg/0/train.csv")
xtrain1 = pd.read_csv("./data/xorg/1/train.csv")
xtrain2 = pd.read_csv("./data/xorg/2/train.csv")
xtrain3 = pd.read_csv("./data/xorg/3/train.csv")
xtrain4 = pd.read_csv("./data/xorg/4/train.csv")
xtrain5 = pd.read_csv("./data/xorg/5/train.csv")

xtest0 = pd.read_csv("./data/xorg/0/test.csv")
xtest1 = pd.read_csv("./data/xorg/1/test.csv")
xtest2 = pd.read_csv("./data/xorg/2/test.csv")
xtest3 = pd.read_csv("./data/xorg/3/test.csv")
xtest4 = pd.read_csv("./data/xorg/4/test.csv")
xtest5 = pd.read_csv("./data/xorg/5/test.csv")

# number of if-statements
num_if = list()
num_else = list()
num_for = list()
num_while = list()
num_switch = list()


# appened the the counts of 'if', 'for', 'else', 'while', and 'switch'
# to a new pandas data frame column
def add_to_csv(data, file):
    # num if
    for x in data:
        num_if.append(x[1])
    # num for
    for x in data:
        num_for.append(x[2])
    # num else
    for x in data:
        num_else.append(x[3])
    # num while
    for x in data:
        num_while.append(x[4])
    # num switch
    for x in data:
        num_switch.append(x[5])

    file["num_if"] = num_if
    file["num_for"] = num_for
    file["num_else"] = num_else
    file["num_while"] = num_while
    file["num_switch"] = num_switch
    num_if.clear()
    num_else.clear()
    num_for.clear()
    num_while.clear()
    num_switch.clear()
    return file


jtraincsv0 = add_to_csv(j_train0, jrtrain0)
jtraincsv1 = add_to_csv(j_train1, jrtrain1)
jtraincsv2 = add_to_csv(j_train2, jrtrain2)
jtraincsv3 = add_to_csv(j_train3, jrtrain3)
jtraincsv4 = add_to_csv(j_train4, jrtrain4)
jtraincsv5 = add_to_csv(j_train5, jrtrain5)

jtestcsv0 = add_to_csv(j_test0, jrtest0)
jtestcsv1 = add_to_csv(j_test1, jrtest1)
jtestcsv2 = add_to_csv(j_test2, jrtest2)
jtestcsv3 = add_to_csv(j_test3, jrtest3)
jtestcsv4 = add_to_csv(j_test4, jrtest4)
jtestcsv5 = add_to_csv(j_test5, jrtest5)

jdtraincsv0 = add_to_csv(jdt_train0, jdtrain0)
jdtraincsv1 = add_to_csv(jdt_train1, jdtrain1)
jdtraincsv2 = add_to_csv(jdt_train2, jdtrain2)
jdtraincsv3 = add_to_csv(jdt_train3, jdtrain3)
jdtraincsv4 = add_to_csv(jdt_train4, jdtrain4)
jdtraincsv5 = add_to_csv(jdt_train5, jdtrain5)

jdtestcsv0 = add_to_csv(jdt_test0, jdtest0)
jdtestcsv1 = add_to_csv(jdt_test1, jdtest1)
jdtestcsv2 = add_to_csv(jdt_test2, jdtest2)
jdtestcsv3 = add_to_csv(jdt_test3, jdtest3)
jdtestcsv4 = add_to_csv(jdt_test4, jdtest4)
jdtestcsv5 = add_to_csv(jdt_test5, jdtest5)

lucenetraincsv0 = add_to_csv(lucene_train0, luctrain0)
lucenetraincsv1 = add_to_csv(lucene_train1, luctrain1)
lucenetraincsv2 = add_to_csv(lucene_train2, luctrain2)
lucenetraincsv3 = add_to_csv(lucene_train3, luctrain3)
lucenetraincsv4 = add_to_csv(lucene_train4, luctrain4)
lucenetraincsv5 = add_to_csv(lucene_train5, luctrain5)

lucenetestcsv0 = add_to_csv(lucene_test0, luctest0)
lucenetestcsv1 = add_to_csv(lucene_test1, luctest1)
lucenetestcsv2 = add_to_csv(lucene_test2, luctest2)
lucenetestcsv3 = add_to_csv(lucene_test3, luctest3)
lucenetestcsv4 = add_to_csv(lucene_test4, luctest4)
lucenetestcsv5 = add_to_csv(lucene_test5, luctest5)

xorgtraincsv0 = add_to_csv(xorg_train0, xtrain0)
xorgtraincsv1 = add_to_csv(xorg_train1, xtrain1)
xorgtraincsv2 = add_to_csv(xorg_train2, xtrain2)
xorgtraincsv3 = add_to_csv(xorg_train3, xtrain3)
xorgtraincsv4 = add_to_csv(xorg_train4, xtrain4)
xorgtraincsv5 = add_to_csv(xorg_train5, xtrain5)

xorgtestcsv0 = add_to_csv(xorg_test0, xtest0)
xorgtestcsv1 = add_to_csv(xorg_test1, xtest1)
xorgtestcsv2 = add_to_csv(xorg_test2, xtest2)
xorgtestcsv3 = add_to_csv(xorg_test3, xtest3)
xorgtestcsv4 = add_to_csv(xorg_test4, xtest4)
xorgtestcsv5 = add_to_csv(xorg_test5, xtest5)


# helper function to output the modified csv files with bag-of-words feature set
def output_csv(file, fold, flag, type):
    path = ""
    if flag == "jr" and type == "train":
        path = "./data/jackrabbit/" + str(fold) + "/train_bow.csv"
    if flag == "jr" and type == "test":
        path = "./data/jackrabbit/" + str(fold) + "/test_bow.csv"
    if flag == "jdt" and type == "train":
        path = "./data/jdt/" + str(fold) + "/train_bow.csv"
    if flag == "jdt" and type == "test":
        path = "./data/jdt/" + str(fold) + "/test_bow.csv"
    if flag == "lucene" and type == "train":
        path = "./data/lucene/" + str(fold) + "/train_bow.csv"
    if flag == "lucene" and type == "test":
        path = "./data/lucene/" + str(fold) + "/test_bow.csv"
    if flag == "xorg" and type == "train":
        path = "./data/xorg/" + str(fold) + "/train_bow.csv"
    if flag == "xorg" and type == "test":
        path = "./data/xorg/" + str(fold) + "/test_bow.csv"
    file.to_csv(path, index=False)


# we now output the new files for each project which contain the
# new feature set
#
# new train and test files for 'jackrabbit' project
output_csv(jtraincsv0, 0, "jr", "train")
output_csv(jtraincsv1, 1, "jr", "train")
output_csv(jtraincsv2, 2, "jr", "train")
output_csv(jtraincsv3, 3, "jr", "train")
output_csv(jtraincsv4, 4, "jr", "train")
output_csv(jtraincsv5, 5, "jr", "train")

output_csv(jtestcsv0, 0, "jr", "test")
output_csv(jtestcsv1, 1, "jr", "test")
output_csv(jtestcsv2, 2, "jr", "test")
output_csv(jtestcsv3, 3, "jr", "test")
output_csv(jtestcsv4, 4, "jr", "test")
output_csv(jtestcsv5, 5, "jr", "test")

# new train and test files for 'jdt' project
output_csv(jdtraincsv0, 0, "jdt", "train")
output_csv(jdtraincsv1, 1, "jdt", "train")
output_csv(jdtraincsv2, 2, "jdt", "train")
output_csv(jdtraincsv3, 3, "jdt", "train")
output_csv(jdtraincsv4, 4, "jdt", "train")
output_csv(jdtraincsv5, 5, "jdt", "train")

output_csv(jdtestcsv0, 0, "jdt", "test")
output_csv(jdtestcsv1, 1, "jdt", "test")
output_csv(jdtestcsv2, 2, "jdt", "test")
output_csv(jdtestcsv3, 3, "jdt", "test")
output_csv(jdtestcsv4, 4, "jdt", "test")
output_csv(jdtestcsv5, 5, "jdt", "test")

# new train and test files for 'lucene' project
output_csv(lucenetraincsv0, 0, "lucene", "train")
output_csv(lucenetraincsv1, 1, "lucene", "train")
output_csv(lucenetraincsv2, 2, "lucene", "train")
output_csv(lucenetraincsv3, 3, "lucene", "train")
output_csv(lucenetraincsv4, 4, "lucene", "train")
output_csv(lucenetraincsv5, 5, "lucene", "train")

output_csv(lucenetestcsv0, 0, "lucene", "test")
output_csv(lucenetestcsv1, 1, "lucene", "test")
output_csv(lucenetestcsv2, 2, "lucene", "test")
output_csv(lucenetestcsv3, 3, "lucene", "test")
output_csv(lucenetestcsv4, 4, "lucene", "test")
output_csv(lucenetestcsv5, 5, "lucene", "test")

# new train and test files for 'xorg' project
output_csv(xorgtraincsv0, 0, "xorg", "train")
output_csv(xorgtraincsv1, 1, "xorg", "train")
output_csv(xorgtraincsv2, 2, "xorg", "train")
output_csv(xorgtraincsv3, 3, "xorg", "train")
output_csv(xorgtraincsv4, 4, "xorg", "train")
output_csv(xorgtraincsv5, 5, "xorg", "train")

output_csv(xorgtestcsv0, 0, "xorg", "test")
output_csv(xorgtestcsv1, 1, "xorg", "test")
output_csv(xorgtestcsv2, 2, "xorg", "test")
output_csv(xorgtestcsv3, 3, "xorg", "test")
output_csv(xorgtestcsv4, 4, "xorg", "test")
output_csv(xorgtestcsv5, 5, "xorg", "test")
