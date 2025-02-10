import torch
from sklearn.metrics import precision_recall_curve, auc, jaccard_score

def PR_AUC_sk(x,y):
    precision, recall, thresholds = precision_recall_curve(y,x)
    return torch.tensor(auc(recall,precision))

def IOU(x,y,threshold=0.5):
    x, y = x >= threshold, y >= threshold
    i = (x * y).sum()
    u = (x + y).sum()
    if u == 0:
        return 0
    return i/u

def PR_AUC(x,y):
    # AUC = int p dr
    x = x.numpy()
    y = y.numpy()
    xy = list(zip(x,y))
    xy = sorted(xy,key=lambda xy: -xy[0])
    tp = positive_pd = 0
    positive_gt = sum(y)
    prev_rec = 0
    prev_perc = 1
    auc = 0
    for x, y in xy:
        positive_pd += 1
        tp += y
        perc = tp/positive_pd
        rec = tp/positive_gt
        d_rec = rec - prev_rec
        auc += prev_perc * d_rec
        prev_rec = rec
        prev_perc = perc

    rec = 1
    d_rec = rec - prev_rec
    auc += prev_perc * d_rec

    return torch.tensor(auc)

def PR_AUC_fast(x,y,n_bins=1000):
    # AUC = int p dr
    x = x.numpy()
    y = y.numpy()

    recs = []
    precs = []

    bins = [0]*n_bins
    num_in_bins = [0]*n_bins
    for i in range(len(x)):
        j = min(int((1-x[i])*n_bins),n_bins-1)
        bins[j] += y[i]
        num_in_bins[j] += 1
    tp = positive_pd = 0
    positive_gt = sum(bins)
    prev_rec = 0
    prev_perc = 1
    auc = 0
    for n, bin in zip(num_in_bins,bins):
        if n == 0:
            continue
        positive_pd += n
        tp += bin
        perc = tp/positive_pd
        rec = tp/positive_gt
        d_rec = rec - prev_rec
        auc += prev_perc * d_rec
        recs.append(rec)
        precs.append(perc)
        prev_rec = rec
        prev_perc = perc
    
    rec = 1
    d_rec = rec - prev_rec
    auc += prev_perc * d_rec
    return torch.tensor(auc), torch.tensor(recs), torch.tensor(precs)
    '''
    positive_pd += 1
    tp += y
    perc = tp/positive_pd
    rec = tp/positive_gt
    d_rec = rec - prev_rec
    auc += (prev_perc+perc)/2 * d_rec
    prev_rec = rec
    prev_perc = perc
    '''

def dice_coef(x, y, smooth=1):
    return (2*(x*y).sum()+smooth) / (x.sum()+y.sum()+smooth)

if __name__ == '__main__':
    x = torch.rand(50000)
    y = torch.randint(0,2,(500000,))
    print(PR_AUC(x,y))
    print(PR_AUC_fast(x,y,1000))
