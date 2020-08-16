"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if (len(a.shape) != 3) or (len(b.shape) != 3):
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if (a.shape != b.shape):
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    a[a > 0] = 1
    b[b > 0] = 1

    intersection = np.sum(a*b)
    volumes = np.sum(a) + np.sum(b)

    if (volumes == 0):
        return -1

    return 2 * intersection / volumes

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if (len(a.shape) != 3) or (len(b.shape) != 3):
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if (a.shape != b.shape):
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    a[a > 0] = 1
    b[b > 0] = 1

    intersection = np.sum(a*b)
    volumes = np.sum(a) + np.sum(b)

    if (volumes == 0):
        return -1

    return intersection / (volumes - intersection)

def Sensitivity(gt, pred):
    """
    This will compute the sensitivity metric for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        gt {Numpy array} -- 3D array with first volume
        pred {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if (len(gt.shape) != 3) or (len(pred.shape) != 3):
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if (gt.shape != pred.shape):
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    gt[gt > 0] = 1
    pred[pred > 0] = 1

    tp = np.sum(gt[gt == pred])
    fn = np.sum(gt[gt != pred])

    if (fn + tp == 0):
        return -1

    return tp / (fn + tp)

def Specificity(gt, pred):
    """
    This will compute the sensitivity metric for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        gt {Numpy array} -- 3D array with first volume
        pred {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if (len(gt.shape) != 3) or (len(pred.shape) != 3):
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if (gt.shape != pred.shape):
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    gt[gt > 0] = 1
    pred[pred > 0] = 1

    tn = np.sum(gt[(gt == 0) & (pred == 0)])
    fp = np.sum(gt[(gt == 0) & (pred == 1)])

    if (fp + tn == 0):
        return -1

    return tn / (fp + tn)