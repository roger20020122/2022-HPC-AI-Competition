
from numpy import log
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import DeepLearningBasedDNASequenceFastDecoding.deepLearningBasedDNASequenceFastDecoding.optim.trainDist as trainDist
import DeepLearningBasedDNASequenceFastDecoding.deepLearningBasedDNASequenceFastDecoding.optim.distUtil as distUtil
import os, sys

MEASURE_TIME = False
DIST = False

class EarlyStopper:
    def __init__(self,patient):
        self.patient = patient
        self.best = None
        self.count = 0
    def __call__(self,score):
        if self.best is None:
            self.best = score
        elif score > self.best:
            self.best = score
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patient

def objective(config):

    if not MEASURE_TIME:  
        config['kernel_size']+=1 # must be odd
        config['dropout'] = float(config['dropout'])
        config['pos_weight'] = float(config['pos_weight'])
        config['num_blocks'] = int(config['num_blocks'])
        config['initial_filter'] = int(config['initial_filter'])
        config['kernel_size'] = int(config['kernel_size'])
        config['scale_filter'] = float(config['scale_filter'])
        if DIST:
            distUtil.broadcast(config)
            print(f'Rank {distUtil.get_rank()} (master) got config {config}')
        config['exp_name'] = f'hyperopt_{config}'
    
    if MEASURE_TIME:
        config = {'dropout': 0.1535039249785174, 'initial_filter': 30, 'kernel_size': 5, 'num_blocks': 5, 'pos_weight': 1.0125812944977837, 'scale_filter': 1.1299710421481397} # Best

    earlyStopper = EarlyStopper(10)
    for pr_auc in trainDist.main(config,yeild_prauc = True):
        if earlyStopper(pr_auc):
            if DIST:
                distUtil.broadcast("stop")
                print(f'early stop')
            break
        else:
            if DIST:
                distUtil.broadcast("continue")
        
    return {
        'loss': - earlyStopper.best,
        'status': STATUS_OK,
    }
    
def main():
    if DIST:
        distUtil.setup_dist()
    if distUtil.is_master():
        trials = Trials()
        best = fmin(objective,
            space={
                'dropout':hp.uniform('dropout', 0, 0.5),
                'pos_weight':hp.loguniform('pos_weight', 0, 3),
                'num_blocks':hp.quniform('num_blocks', 3, 7, 1),
                'initial_filter':hp.quniform('initial_filter', 10, 30, 1),
                'kernel_size':hp.quniform('kernel_size', 4, 12, 2),
                'scale_filter':hp.loguniform('scale_filter',log(1),log(2)),
                },
            algo=tpe.suggest,
            max_evals=1000,
            trials=trials,
            show_progressbar=False,
            )
    else:
        while True:
            config = distUtil.broadcast()
            #print(f'Rank {dist_util.get_rank()} got config {config}')
            for _ in trainDist.main(config,yeild_prauc = True):
                if distUtil.broadcast() == "stop":
                    break

if __name__ == '__main__':
    main()