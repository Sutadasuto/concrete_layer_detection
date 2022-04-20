import cv2
import math
import numpy as np
import os
import warnings

from scipy import stats
# from skimage.feature import greycomatrix as glcm


def calculate_dsc(y, y_pred, smooth=1):
    intersection = y_pred * y
    return (2 * np.sum(intersection) + smooth) / (np.sum(y_pred) + np.sum(y) + smooth)


def calculate_confusion_matrix(y, y_pred):
    confusion_matrix = np.zeros((2, 2))
    matrix = np.array([['true_positives', 'false_positives'], ['false_negatives', 'true_negatives']])
    h, w = y.shape
    tp = y_pred * y
    fp = y_pred - tp
    fn = np.maximum(0, (1 - y_pred) - (1 - y))

    confusion_matrix[0, 0] = np.sum(tp)
    confusion_matrix[0, 1] = np.sum(fp)
    confusion_matrix[1, 0] = np.sum(fn)
    confusion_matrix[1, 1] = h * w - np.sum(tp) - np.sum(fp) - np.sum(fn)

    return confusion_matrix, matrix


def calculate_PRF(confusion_matrix):
    if not confusion_matrix[0, 0] + confusion_matrix[0, 1] == 0:
        precision = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    else:
        if confusion_matrix[1, 0] == 0:
            warnings.warn(
                'The sum of true and false positives is 0 (no pixel predicted as crack). However, no false negative is present: setting precision to 1.')
            precision = 1
        else:
            warnings.warn(
                'The sum of true and false positives is 0 (no pixel predicted as crack). False negatives are present: setting precision to 0.')
            precision = 0
    if not confusion_matrix[0, 0] + confusion_matrix[1, 0] == 0:
        recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    else:
        warnings.warn(
            'The sum of true positives and false negatives is 0 (no crack pixel exists in the groundtruth). Setting recall to 1.')
        recall = 1
    if precision == 0 and recall == 0:
        warnings.warn('Precision and recall equal to 0. Setting f-score to 0 (totally wrong prediction).')
        f = 0
    else:
        f = 2 * precision * recall / (precision + recall)
    return precision, recall, f


# def calculate_Hr(x, y_pred):
#     Rj = y_pred
#     Sj = np.sum(Rj)
#     Vj, Lj = np.unique(x[np.where(Rj == 1.0)], return_counts=True)
#
#     Hr = 0
#     for m in range(len(Vj)):
#         Hr += (Lj[m] / Sj) * math.log(Lj[m] / Sj)
#     Hr *= -1
#     return Hr
#
#
# def calculate_approximate_Hr2(x, y_pred):
#     # Calculate an approximation of the second order entropy inside a mask.
#     # To calculate the intra-mask co-occurrence matrix, the input image is converted to uint16. The image pixels outside
#     # the mask are assigned 256 as intensity and the co-occurrence matrix is calculated with intensities in the range
#     # [0, 256]. Then, the last row and column from the matrix are suppressed.
#     # The co-occurrence probability matrix is calculated with a (radius 1) 8-pixels neighborhood
#
#     intensities = 2**8
#     if x.dtype != np.uint8:
#         x = (255*x).astype(np.uint8)
#     x_16b = x.astype(np.uint16)
#     x_16b[np.where(y_pred == 0)] = intensities
#     cm = glcm(x_16b, [1], [i*np.pi/4 for i in range(8)], intensities+1)
#     cm = cm[:-1, :-1]  # Crop the row and column corresponding to intensity 256
#     cm = np.sum(cm[...,0,:], axis=-1) # Join the 8 directions into a single 2D matrix
#     p = cm.astype(np.float64)/np.sum(cm)
#
#     Hr2 = 0
#     for i in range(intensities):
#         for j in range(intensities):
#             if p[i, j] > 0:
#                 Hr2 += p[i, j] * math.log(p[i, j])
#     Hr2 *= -1
#     return Hr2
#
#
# def calculate_kolmogorov_smirnov_statistic(x, y_pred, alpha):
#     bkgd = x[y_pred == 0]
#     crack = x[y_pred != 0]
#     if len(crack) == 0 or len(bkgd) == 0:
#         warnings.warn('One of the classes is not populated. Thus, the whole image is classified as crack or background.'
#                       ' Since the network did not make a difference between the two distributions (classes), they are'
#                       ' assumed to be the same one. Therefore the distance between them is assumed to be 0.')
#         statistic, p_value = (0.0, 0.0)
#     else:
#         statistic, p_value = stats.ks_2samp(bkgd, crack)
#     if p_value > alpha:
#         warnings.warn(
#             'The p-value {:.4f} is greater than the alpha {:.4f}. We cannot reject the null hypothesis in favor of '
#             'the alternative, thus we assume the Kolmogorov-Smirnov statistic to be zero (the distributions are '
#             'identical).')
#         statistic = 0.0
#     return statistic


def calculate_tolerant_scores(predictions_path, gt_path, tolerance=2):

    root_folder = os.path.split(predictions_path)[0]
    results_folder = os.path.join(root_folder, "evaluation-tolerance%s" % tolerance)
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    score_names = ['f-score', 'precision', 'recall']
    string_list = [','.join(['image'] + score_names)]

    image_names = sorted([f for f in os.listdir(gt_path)
                          if not f.startswith(".") and f.endswith(".png")],
                         key=lambda f: f.lower())
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*tolerance + 1, 2*tolerance + 1))
    scores = np.zeros((len(image_names), 3))
    for i, name in enumerate(image_names):
        print("\rImages processed: %s/%s" % (i, len(image_names)), end='')
        confusion_matrix = np.zeros((2, 2))
        score_dict = {}
        current_string_list = [os.path.join(predictions_path.split()[-1], name)] + ['' for n in score_names]

        gt = cv2.imread(os.path.join(gt_path, name), cv2.IMREAD_GRAYSCALE).astype(np.float)
        gt = np.where(gt / 255.0 >= 0.5, 1.0, 0.0)
        pred = cv2.imread(os.path.join(predictions_path, name), cv2.IMREAD_GRAYSCALE).astype(np.float)
        pred = np.where(pred / 255.0 >= 0.5, 1.0, 0.0)
        h, w = gt.shape

        if tolerance > 0:
            tolerant_gt = cv2.dilate(gt,se, borderType=cv2.BORDER_REFLECT_101)
        else:
            tolerant_gt = gt

        tp = pred*tolerant_gt
        fp = pred - tp
        fn = np.maximum(0, (1 - pred) - (1 - gt))

        confusion_matrix[0, 0] = np.sum(tp)
        confusion_matrix[0, 1] = np.sum(fp)
        confusion_matrix[1, 0] = np.sum(fn)
        confusion_matrix[1, 1] = h * w - np.sum(tp) - np.sum(fp) - np.sum(fn)

        score_dict['precision'], score_dict['recall'], score_dict['f-score'] = calculate_PRF(confusion_matrix)

        for key in score_dict.keys():
            scores[i, score_names.index(key)] = score_dict[key]
            current_string_list[score_names.index(key) + 1] = "{:.4f}".format(score_dict[key])
        string_list.append(",".join(current_string_list))

        tn = np.where((fn == 0) & (tp == 0) & (fp == 0))
        fn[tn] = 1.0
        tp[tn] = 1.0
        fp[tn] = 1.0
        output_image = np.concatenate((fn[..., None], tp[..., None], fp[..., None]), axis=-1)
        cv2.imwrite(os.path.join(results_folder, name), 255*output_image)

    print("\rImages processed: %s/%s" % (i + 1, len(image_names)))

    current_string_list = []
    summary_string_list = []
    average_scores = np.average(scores, axis=0)
    for i, score in enumerate(average_scores):
        current_string_list.append("{:.4f}".format(score))
        summary_string_list.append("{}: {:.4f}".format(score_names[i], score))
    current_string_list = ['average'] + current_string_list
    string_list.append(",".join(current_string_list))

    summary_string = "\n".join(summary_string_list)
    with open(os.path.join(results_folder, "scores_summary.txt"), 'w+') as f:
        f.write(summary_string)

    results_string = "\n".join(string_list)
    with open(os.path.join(results_folder, "scores.csv"), 'w+') as f:
        f.write(results_string)
