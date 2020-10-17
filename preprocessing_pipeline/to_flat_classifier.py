import pandas as import pd
import scipy.sparse




label_map=pd.read_csv('../data/intermediary-data/xbert_inputs/label_map.txt',sep='\t',header=None)
instance2label_trn=pd.DataFrame(scipy.sparse.load_npz('../data/intermediary-data/xbert_inputs/Y.trn.npz').todense())
instance2label_tst=pd.DataFrame(scipy.sparse.load_npz('../data/intermediary-data/xbert_inputs/Y.tst.npz').todense())
