import os
import pickle

import numpy as np
import pandas
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels


def prepare_score_string(p, r, f1, s, labels, target_names, digits):
    string = ",precision,recall,f1,total#\n"
    for i, label in enumerate(labels):
        string = string + '"'+str(target_names[i]) + '",'
        for v in (p[i], r[i], f1[i]):
            string = string + "{0:0.{1}f}".format(v, digits) + ","
        string = string + "{0}".format(s[i]) + "\n"
        # values += ["{0}".format(s[i])]
        # report += fmt % tuple(values)

    return string


def write_scores(
    predictoins, truth: pandas.Series, digits, writer, label_mapping: dict
):
    labels = unique_labels(truth, predictoins)
    inv_map = {v: k for k, v in label_mapping.items()}

    target_names = [inv_map[l] for l in labels]
    p, r, f1, s = precision_recall_fscore_support(truth, predictoins, labels=labels)

    line = prepare_score_string(p, r, f1, s, labels, target_names, digits)
    pa, ra, f1a, sa = precision_recall_fscore_support(
        truth, predictoins, average="micro"
    )
    line += "avg_micro,"
    for v in (pa, ra, f1a):
        line += "{0:0.{1}f}".format(v, digits) + ","
    line += "{0}".format(np.sum(sa)) + "\n"
    pa, ra, f1a, sa = precision_recall_fscore_support(
        truth, predictoins, average="macro"
    )
    line += "avg_macro,"
    for v in (pa, ra, f1a):
        line += "{0:0.{1}f}".format(v, digits) + ","
    line += "{0}".format(np.sum(sa)) + "\n\n"
    # average

    writer.write(line)


def outputFalsePredictions(
    pred, truth, index, model_id, outfolder, instance_mapping: dict, label_mapping: dict
):
    inv_map = {v: k for k, v in label_mapping.items()}
    subfolder = outfolder + "/errors"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    filename = "errors" + ".csv"  #

    filename = os.path.join(subfolder, filename)
    file = open(filename, "w", encoding="utf-8")
    file.write("index,row_no_in_original_data_file,prediction,truth,wrong/correct\n")

    #try:
    for p, t, i in zip(pred, truth, index):
        if p == t:
            line = (
                str(i) + "," + str(instance_mapping[i]) + ',"' + str(inv_map[p])+ '","' + str(inv_map[t]) + '",ok\n'
            )
            file.write(line)
        else:
            line = (
                    str(i)
                    + ","
                    + str(instance_mapping[i])
                    + ',"'
                    + str(inv_map[p])
                    + '","' + str(inv_map[t])
                    + '",wrong\n'
            )
            file.write(line)
    file.close()
    #except:
    #    print()


def outputPredictions(
    pred, index, outfolder, instance_mapping: dict, label_mapping: dict
):
    inv_map = {v: k for k, v in label_mapping.items()}

    subfolder = outfolder + "/results"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)

    filename = "results" + ".csv"

    filename = os.path.join(subfolder, filename)

    file = open(filename, "w", encoding="utf-8")
    # writing the header of the table
    line = "ID" + "," + "Data" + "," + "Label"
    file.write(line + "\n")

    for p, i in zip(pred, index):

        # if i == 136:
        # print(i)
        line = str(i) + ',"' + str(inv_map[p]) + '"\n'

        # line = str(i) + "," + instance_mapping[i] + "," + str(inv_map[p]) + "\n"

        file.write(line)

    file.close()


def save_scores(
    predictions,
    ground_truth,
    alg_id,
    digits,
    outfolder,
    instance_mapping: dict,
    label_mapping: dict,
):
    if type(predictions) is not list:
        pred = predictions.tolist()
    else:
        pred = predictions
    truth = list(ground_truth)
    index = [i for i in range(0, len(predictions))]
    # saving the error analysis results
    outputFalsePredictions(
        pred, truth, index, alg_id, outfolder, instance_mapping, label_mapping
    )
    # saving the results of the classified tables
    outputPredictions(pred, index, outfolder, instance_mapping, label_mapping)

    # saving the evaluation results [Precision , Recall and F-measure]
    subfolder = outfolder + "/scores"
    try:
        os.stat(subfolder)
    except:
        os.mkdir(subfolder)
    filename = os.path.join(subfolder, "scores_all_models.csv")
    writer = open(filename, "a+")
    writer.write(alg_id + "\n")
    if predictions is not None:
        writer.write(" N-FOLD AVERAGE :\n")
        write_scores(predictions, ground_truth, digits, writer, label_mapping)

    writer.close()


def save_classifier_model(model, outfile):
    try:
        if model:
            with open(outfile, "wb") as model_file:
                pickle.dump(model, model_file)
    except AttributeError:
        print("Saving model failed. Perhaps not supported.")


def load_classifier_model(classifier_pickled=None):
    if classifier_pickled:
        with open(classifier_pickled, "rb") as model:
            classifier = pickle.load(model)
        return classifier
