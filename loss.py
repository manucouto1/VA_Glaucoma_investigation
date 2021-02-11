
import numpy as np
import chain as ch

def true_positives(g_truth, mask):
    return np.count_nonzero(np.logical_and(g_truth, mask))

def false_positives(g_truth, mask):
    return np.count_nonzero(np.logical_and(g_truth == False, mask))

def true_negatives(g_truth, mask):
    return np.count_nonzero(np.logical_or(g_truth, mask) == False)

def false_negatives(g_truth, mask):
    return np.count_nonzero(np.logical_and(g_truth, mask==False))

def sensitivity(g_truth, mask):
    TP = true_positives(g_truth, mask)
    FN = false_negatives(g_truth, mask)
    return TP/(TP+FN)

def specificity(g_truth, mask):
    FP = true_positives(g_truth, mask)
    TN = true_negatives(g_truth, mask)
    return TN/(TN+FP)

def jaccard(g_truth, mask):
    TP = true_positives(g_truth, mask)
    FN = false_negatives(g_truth, mask)
    FP = false_positives(g_truth, mask)
    return TP/(TP+FN+FP)

def DICE(g_truth, mask):
    TP = true_positives(g_truth, mask)
    FN = false_negatives(g_truth, mask)
    FP = false_positives(g_truth, mask)
    return 2 * TP/((TP+FN)+(TP+FP))

def run_all(tool, f_test, op="disc"):
    masks = []
    for key, value in tool.data.items():
        image = value["img"]
        g_truth = value[op]
        (mask ,_ ,_) = f_test(image, op)
        masks.append((key, mask, g_truth))

    return (masks, op)

def metrics(masks, tester):
    average = 0
    v_list = []
    n = 0
    (lista, op) = masks

    for (key, mask, g_truth) in lista:
        value = tester(g_truth > 0, mask)
        average += value
        n+=1
        v_list.append(value)
        print(tester.__name__, " value of ",key," ",op," : ", value)

    print("average:",average/n*100,"%")
    return (tester.__name__,v_list)



