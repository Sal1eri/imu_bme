import numpy as np


#得到混淆矩阵
def _fast_hist(label_true, label_pred, n_class):
    """
     :param label_pred: numpy data, shape:[batch,h,w]
     :param label_trues:同上
     :param n_class:类别数
     Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """

    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)

    return hist


# 计算图像分割衡量系数
def label_accuracy_score(label_trues, label_preds, n_class):
    """
     :param label_preds: numpy data, shape:[batch,h,w]
     :param label_trues:同上
     :param n_class:类别数
     Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
      - recall
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    #
    acc = np.diag(hist).sum() / hist.sum()
    #   acc_cls类别像素准确率
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    #   这里覆盖了 变成了类别像素准确率的平均值
    mean_acc_cls = np.nanmean(acc_cls)
    #   类别交并比
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    #   平均交并比
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    #   fwavacc加权交并比
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    #   防止出现0/0的情况，这样会报错。所以加一个很小很小的值 并不影响总体的计算
    smooth = 1e-8

    # 类别dice
    dice = np.zeros(n_class)
    for i in range(n_class):
        dice[i] = (2 * np.diag(hist)[i] + smooth) / (hist.sum(axis=1)[i] + hist.sum(axis=0)[i] + smooth)

    mean_dice = np.nanmean(dice)

    # 计算召回率
    recall = np.diag(hist) / (hist.sum(axis=1) + smooth)
    mean_recall = np.nanmean(recall)

    return acc, mean_acc_cls, mean_iu, fwavacc, iu, acc_cls, dice, mean_dice, recall, mean_recall
