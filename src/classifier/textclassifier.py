"""
"""

import csv
import json
import os
import random

import pandas as pd
from classifier import classifier_util as cu
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelBinarizer
from util import logging_util as lu

random.seed(10)


def get_feature_and_label_vectors(input_matrix):
    X_train = input_matrix[:, 1 : len(input_matrix[0])]
    y_train = input_matrix[:, 0]
    return X_train, y_train


def get_classifer(alg="rf"):
    if alg == "rf":
        classifier = RandomForestClassifier(n_estimators=20, n_jobs=-1)
        return classifier
    elif alg == "lr":
        classifier = LogisticRegression(random_state=300)
        return classifier
    elif alg == "svmrb":
        classifier = svm.SVC()
        return classifier

    elif alg == "svm":
        classifier = svm.LinearSVC(
            class_weight="balanced",
            C=0.01,
            penalty="l2",
            loss="squared_hinge",
            multi_class="ovr",
        )
        return classifier

    else:
        print("algorithm not supported: {}".format(alg))
        return -1


def encode_labels(y_train_text):
    encoder = LabelBinarizer()
    y_train_vec = encoder.fit_transform(y_train_text)
    label_dictionary = dict(zip(encoder.classes_, range(len(encoder.classes_))))
    y_train = []
    for i in y_train_text:
        y_train.append(label_dictionary[i])

    return y_train, label_dictionary


"""
This method takes the feature matrix (as a pandas dataframe) and performs n-fold validation
to evaluate the ML algorithm

- feature_dataframe: an i x j matrix where each row i is an instance, each column j (except 0) is a feature,
  column 0 should be the label (classification target)
- feature_index: a dictionary where key = j (j <> 0 and is an int), value = text label for that feature
- instance_index: a dictionary where key = i, value= the text object for classification
- alg: permitted values: svml, svmrb, rf, lr
"""


def nfold_validation(
    feature_dataframe: pd.DataFrame,
    instance_index: dict,
    outfolder: str,
    alg="rf",
    nfold=10,
    setting_id="general",
):
    classifier = get_classifer(alg)
    if classifier == -1:
        return

    matrix = feature_dataframe.to_numpy()
    X_train, y_train_text = get_feature_and_label_vectors(matrix)
    y_train, label_dictionary = encode_labels(y_train_text)

    #oversample = RandomOverSampler(sampling_strategy="minority")
    # fit and apply the transform
    #X_over, y_over = oversample.fit_resample(X_train, y_train)
    #X_train = X_over
    #y_train = y_over
    # print("\t running nfold validation {}".format(datetime.datetime.now()))
    classifier.fit(X_train, y_train)
    lu.log.info("NFOLD %i %i %i", len(X_train), len(y_train), nfold)
    if len(X_train) < nfold:
        return []

    nfold_predictions = cross_val_predict(classifier, X_train, y_train, cv=nfold)
    # save scores to a file
    if setting_id is not None:
        setting_id = setting_id.replace(",", "_")
        # alg = alg + "_" + setting_id

    cu.save_scores(
        nfold_predictions,
        y_train,
        str(alg) + "_" + str(setting_id),
        2,
        outfolder,
        instance_index,
        label_dictionary,
    )

    return list(nfold_predictions)


"""
This method takes the feature matrix (as a pandas dataframe) and performs training on the training set only.
It will also output the trained model

- alg: permitted values: svml, svmrb, rf, lr
"""


def train(
    training_feature_matrix_as_dataframe: pd.DataFrame,
    outfolder: str,
    alg="rf",
    setting_id="general",
):
    classifier = get_classifer(alg)
    if classifier == -1:
        return
    matrix = training_feature_matrix_as_dataframe.to_numpy()
    X_train, y_train_text = get_feature_and_label_vectors(matrix)
    y_train, label_dictionary = encode_labels(y_train_text)
    #oversample = RandomOverSampler(sampling_strategy="minority")
    # fit and apply the transform
    #X_over, y_over = oversample.fit_resample(X_train, y_train)
    #X_train = X_over
    #y_train = y_over
    classifier.fit(X_train, y_train)
    # Calculating Feature weight
    feature_names = list(training_feature_matrix_as_dataframe.head())
    # delete the "label" name from the feature_names list
    feature_names.pop(0)
    feature_importance_dict = {}
    feature_importance = list(classifier.feature_importances_)
    for i in range(0, len(feature_names)):
        feature_importance_dict[feature_names[i]] = feature_importance[i]
    # save feature weight to json file
    with open(outfolder + "/feature_importance.json", "w") as fp:
        json.dump(feature_importance_dict, fp)

    model_file = os.path.join(outfolder, "{}-{}.m".format(alg, setting_id))
    cu.save_classifier_model(classifier, model_file)
    with open(outfolder + "/label_dictionary.json", "w") as fp:
        json.dump(label_dictionary, fp)

    return label_dictionary, model_file


"""
This method takes a pre-trained model, the input feature matrix for testing data (as a pandas dataframe) and performs 
prediction using the model

"""


def predict(
    testing_feature_matrix_as_dataframe: pd.DataFrame,
    outfolder: str,
    model_file: str,
    instance_index: dict,
    label_dictionary: dict,
    hold_out_eval=False,
):
    inv_map = {v: k for k, v in label_dictionary.items()}
    model = cu.load_classifier_model(model_file)
    model_file = model_file.replace("\\", "/")
    model_name = model_file[model_file.rindex("/") + 1 :].strip()

    y_test = []
    matrix = testing_feature_matrix_as_dataframe.to_numpy()
    if (
        hold_out_eval
    ):  # assuming the first column is label, we need to chunk the matrix to get the feature and label vectors
        X_test, y_test_text = get_feature_and_label_vectors(matrix)
        predictions = model.predict_proba(X_test)
        for y in y_test_text:
            y_test.append(label_dictionary[y])
    else:
        predictions = model.predict_proba(matrix)

    filename = os.path.join(outfolder, "prediction-{}.csv".format(model_name))
    file = open(filename, "w")

    prediction_labels = []
    prediction_indeces = []
    probablities_labels = []
    for p in predictions:
        prob = {}
        for i in range(0, len(p)):
            label = inv_map[i]
            prob[label] = p[i]
        probablities_labels.append(prob)

        pred = list(p).index(max(p))
        prediction_indeces.append(pred)
        pred = inv_map[pred]
        file.write(pred + "\n")
        prediction_labels.append(pred)
    file.close()

    if hold_out_eval:
        cu.save_scores(
            prediction_indeces,
            y_test,
            model_name,
            2,
            outfolder,
            instance_index,
            label_dictionary,
        )

    return prediction_labels, probablities_labels


if __name__ == "__main__":
    in_file = "/home/zz/Work/vamstar/wolf-textzoning/input/test_docs/training.csv"
    out_folder = "/home/zz/Work/vamstar/wolf-textzoning/input/test_docs"
    gs = pd.read_csv(
        in_file, header=0, delimiter="\t", quoting=csv.QUOTE_MINIMAL, encoding="utf-8",
    ).fillna("none")

    alg = "rf"
    feature_dataframe = []
    feature_dictionary = {}
    instance_index = {}
    index = 0
    for h in gs.head():
        feature_dictionary[index] = h
        index += 1

    for index, row in gs.iterrows():
        instance_index[index] = "item-" + str(index)
        feature_dataframe.append(list(row))

    print("trying n-fold validation")
    nfold_validation(
        pd.DataFrame(feature_dataframe, columns=list(gs.head())),
        feature_dictionary,
        instance_index,
        out_folder,
        alg=alg,
        nfold=5,
        setting_id="general",
    )

    print("trying training only")
    label_dictionary = train(
        pd.DataFrame(feature_dataframe, columns=list(gs.head())),
        out_folder,
        alg=alg,
        setting_id="general",
    )

    print("trying prediction only")

    feature_dataframe = []
    model_file = "/home/zz/Work/vamstar/wolf-textzoning/input/test_docs/{}-general.m".format(
        alg
    )
    for index, row in gs.iterrows():
        instance_index[index] = "item-" + str(index)
        feature_dataframe.append(list(row))
    predict(
        pd.DataFrame(feature_dataframe, columns=list(gs.head())),
        out_folder,
        model_file,
        instance_index,
        label_dictionary,
        hold_out_eval=True,
    )
    print("all done")
