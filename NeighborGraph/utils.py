import pickle


def readpkl(file):
    fr = open(file, 'rb')
    data = pickle.load(fr, encoding='latin1')
    return data


def bbox_ctr_dist(a, b):
    a_ct_x = (a[0] + a[2]) / 2.
    a_ct_y = (a[1] + a[3]) / 2.

    b_ct_x = (b[0] + b[2]) / 2.
    b_ct_y = (b[1] + b[3]) / 2.

    eculid_dist = ((a_ct_x - b_ct_x) ** 2 + (a_ct_y - b_ct_y) ** 2) ** 0.5

    return eculid_dist
