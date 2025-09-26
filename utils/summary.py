""" Summary utilities

Hacked together by / Copyright 2020 Ross Wightman
Modified by Janet Kok to update cross validation results
"""
import csv
import os
from collections import OrderedDict

def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


def update_summary(epoch, train_metrics, val_metrics, test_metrics, filename, write_header=False, log_wandb=False):
    rowd = OrderedDict(epoch=epoch)
    rowd.update([('train_' + k, v) for k, v in train_metrics.items()])
    rowd.update([('val_' + k, v) for k, v in val_metrics.items()])
    rowd.update([('test_' + k, v) for k, v in test_metrics.items()])

    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)

def update_cv(fold, cv_metrics,filename, write_header=False):
    rowd = OrderedDict(fold=fold)
    rowd.update([('test_' + k, v) for k, v in cv_metrics.items()])

    with open(filename, mode='a') as cf:
        dw = csv.DictWriter(cf, fieldnames=rowd.keys())
        if write_header:  # first iteration (epoch == 1 can't be used)
            dw.writeheader()
        dw.writerow(rowd)
