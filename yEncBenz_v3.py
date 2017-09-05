#This is an ensemble classifier of GLM, RF, ExtraTree for kaggle Benz Competition
#Input: train.csv, test.csv
#Output: A series of predictions (in csv) by each classifiers and finally ensemble by a optimized weights

#references:
#y value trick from but using median
#https://www.kaggle.com/robertoruiz/a-magic-feature/code
#https://www.kaggle.com/c/mercedes-benz-greener-manufacturing/discussion/34180
#random_projection and some decomposition from
#https://www.kaggle.com/hakeem/stacked-then-averaged-models-0-5697
#kernel5686XGB
#https://www.kaggle.com/linux18/kernel-0-5686/code

import time
from random import choice, sample, shuffle, uniform, seed
from math import exp, expm1, log1p, log10, log2, sqrt, ceil, floor
#
import numpy as np
import pandas as pd
#features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.decomposition import TruncatedSVD, PCA, FastICA, NMF
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection 
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.cluster import KMeans, MiniBatchKMeans
#eval
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
#models
from sklearn.linear_model import SGDRegressor, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

target_id, target = 'ID', 'y'
###############################################################################

def read_data(path='./', train='train.csv', test='test.csv'):
    
    #train
    df_train = pd.read_csv(path + train)
    print("original data: X_train: {0}".format(df_train.shape), flush=True)
    #df_train['Xid'] = df_train[target_id].apply(log1p)
    #test
    df_test = pd.read_csv(path + test)
    print("original data: X_test: {0}".format(df_test.shape), flush=True)
    #df_test['Xid'] = df_test[target_id].apply(log1p)

    val_uqniues = {}

    feats = list(set(df_train.columns.tolist()).difference([target, target_id]))
    print('read in {} features'.format(len(feats)), flush=True)
    for c in feats:
        if df_train[c].dtype == 'object':
            candidates = df_train[c].tolist() + df_test[c].tolist()
            lbl = LabelEncoder()
            lbl.fit(candidates)
            df_train[c] = lbl.transform(df_train[c].values)
            df_test[c] = lbl.transform(df_test[c].values)
            val_uqniues[c] = set(candidates)
            print('{0} is object: {1:d} uniques'.format(c, len(set(candidates))), flush=True)

    return df_train, df_test, feats, val_uqniues

###############################################################################
def get_clf(clf_name='', params={}, random_seed=0):
    if clf_name == 'xt':
        return ExtraTreesRegressor(n_estimators=params.get('trees', 100), 
                                   criterion='mse', 
                                   max_depth=params.get('depth', 5), 
                                   min_samples_split=16, 
                                   min_samples_leaf=4, 
                                   min_weight_fraction_leaf=0.0, 
                                   max_features='auto', 
                                   max_leaf_nodes=params.get('leafs', 32), 
                                   bootstrap=True, 
                                   oob_score=True, 
                                   n_jobs=-1, 
                                   random_state=random_seed, 
                                   verbose=0, 
                                   warm_start=False)

    if clf_name == 'rf': 
        return RandomForestRegressor(n_estimators=params.get('trees', 100), 
                                     criterion='mse', 
                                     max_depth=params.get('depth', 5), 
                                     min_samples_split=16, 
                                     min_samples_leaf=4, 
                                     min_weight_fraction_leaf=0.0, 
                                     max_features='auto', 
                                     max_leaf_nodes=params.get('leafs', 32), 
                                     bootstrap=True, 
                                     oob_score=True, 
                                     n_jobs=-1, 
                                     random_state=random_seed, 
                                     verbose=0, 
                                     warm_start=False)

    if clf_name == 'gbdt':
        return GradientBoostingRegressor(loss='huber', 
                                         learning_rate=params.get('eta', 0.1), 
                                         n_estimators=params.get('trees', 100), 
                                         subsample=params.get('subsample', 0.7), 
                                         criterion='friedman_mse', 
                                         min_samples_split=16, 
                                         min_samples_leaf=4, 
                                         min_weight_fraction_leaf=0.0, 
                                         max_depth=params.get('depth', 8), 
                                         random_state=random_seed, 
                                         max_features=params.get('sub_f', 0.7), 
                                         alpha=0.9, 
                                         verbose=0, 
                                         max_leaf_nodes=params.get('leafs', 32), 
                                         warm_start=False)

    if clf_name == 'glm':
        return ElasticNet(alpha=params.get('a', 0.005), 
                          l1_ratio=params.get('l1r', 0.25), 
                          fit_intercept=True, 
                          normalize=False, 
                          precompute=False, 
                          max_iter=500, 
                          copy_X=True, 
                          tol=0.0001, 
                          warm_start=False, 
                          positive=False, 
                          random_state=random_seed, 
                          selection='cyclic')

    if clf_name == 'sdgr':
        return SGDRegressor(loss='huber', 
                            penalty='elasticnet', 
                            alpha=params.get('a', 0.005), 
                            l1_ratio=params.get('l1r', 0.10), 
                            fit_intercept=True, 
                            max_iter=1000, 
                            tol=0.0001, 
                            shuffle=True, 
                            verbose=0, 
                            epsilon=params.get('eps', 0.025), 
                            random_state=random_seed, 
                            learning_rate='invscaling', 
                            eta0=0.01, 
                            power_t=0.25, 
                            warm_start=False, 
                            average=False)


def tf_transform(data, method, stem='', opt_fit=False, f_sel=[]):
    if opt_fit:
        trans_data = method.fit_transform(data[f_sel])
                
    else:
        trans_data = method.transform(data[f_sel])
             
    cols = ['f_{}_{:03d}'.format(stem, nb+1) for nb in range(trans_data.shape[1])]
    return pd.concat([data, pd.DataFrame(trans_data, columns=cols)], axis=1)

#functions useful for further stacking
###############################################################################
def assign_folds(y, nr_splits=2, random_seed=0):
    train_sets, valid_sets = list(), list()
    fold_gen = KFold(n_splits=nr_splits, shuffle=True, random_state=random_seed)
    for train_indices, valid_indices in fold_gen.split(y, y):
        train_sets.append(train_indices)
        valid_sets.append(valid_indices)
    return train_sets, valid_sets

def collect_predictions(preds, collect, d=1., col='y_', opt_test=False):
    if opt_test:
        if col not in collect.columns:
            collect[col] = 0.
        collect[col] += preds / d
    else:        
        collect[col] = preds

def optimize_weights(trials=1000, y=None, df=None, f_sel=[]):
    #
    best_w = {}
    best_r2, best_rmse = -1.0 * float('inf'), float('inf')    
    
    #using only existing in f_sel
    f_sel = [f for f in f_sel if f in df.columns.tolist()]
    
    for i in range(trials):
        w = [uniform(0.15, 0.75) for f in f_sel]
        s = sum(w)
        weights = {f: w/s for f, w in zip(f_sel, w)}

        preds = (df[f_sel] * pd.Series(weights)).sum(1)
        score = r2_score(y, preds.values)
        rmse = sqrt(mean_squared_error(y, preds.values))
            
        if score > best_r2:
            best_r2, best_rmse = score, rmse
            best_w = {k: v for k, v in weights.items()}
            print('no {0:04d}: r2 = {1:.5f}, rmse = {2:.3f}, current best ({3})'.format(i, score, rmse, best_w), flush=True)

    print('Best r2 = {0:.5f}, rmse = {1:.3f}'.format(best_r2, best_rmse))
    return best_w, best_r2, best_rmse

#csv
###############################################################################
def write_csv(y_id, y_preds, stem=''):
    sub = pd.DataFrame({target_id: y_id, target: y_preds})
    sub.to_csv(stem, index=False)

###############################################################################

if __name__ == '__main__':

    options = {}
    nr_splits = 5
    seed_fold_gen = 904
    seed_val =170904
    seed_tf = 201706
    tmstmp = '{0}'.format(time.strftime("%Y-%m-%d-%H-%M"))
    
    #params
    cutoff = 40
    nb_comp = {'svd': 16, 'pca':4, 'ica':4, 'grp': 8, 'srp': 8, 'nmf': 16} #transformers
    n_clust = 4

    #load data
    df_train, df_test, feats, val_uqniues = read_data(path='../input/')
    #data
    train_X, test_X = df_train[feats], df_test[feats]
    #id
    train_id, test_id = df_train[target_id], df_test[target_id]
    #y, and if need capping for outlier
    train_y = df_train[target]
    raw_train_y = df_train[target].copy()

    #fold assignments
    train_sets, valid_sets = assign_folds(train_y, nr_splits=nr_splits, random_seed=seed_fold_gen)

    f_hcc = [k for k, v in val_uqniues.items() if len(v) >= cutoff]#'X0', 'X2'
    print('High cardinality (>= {}): {}'.format(cutoff, f_hcc))

    #regressor    
    reg_scikit = {}
    reg_scikit['sdgr_l2'] = get_clf(clf_name='sdgr', params={'a':0.0125, 'l1r': 0.002}, random_seed=seed_val)
    reg_scikit['glm_l1'] = get_clf(clf_name='glm', params={'a':0.5, 'l1r': 0.7}, random_seed=seed_val)
    reg_scikit['xt_sparse'] = get_clf(clf_name='xt', params={'trees': 800, 'leafs': 32, 'depth': 12}, random_seed=seed_val)
    reg_scikit['rf_dense'] = get_clf(clf_name='rf', params={'trees': 400, 'leafs': 128, 'depth': 16}, random_seed=seed_val)
    #reg_scikit['gbdt'] = get_clf(clf_name='gbdt', params={'trees': 560, 'leafs': 24, 'depth': 10}, random_seed=seed_val)
    #
    print('classifiers config:')
    for k, reg in reg_scikit.items():
        print('{0}={1}'.format(k, reg.get_params()), flush=True)

    #four known clusters
    clust = MiniBatchKMeans(n_clusters=n_clust, max_iter=1000, init_size=n_clust*10)
    #decompositions
    tfs = {}
    tfs['svd'] = TruncatedSVD(n_components=nb_comp['svd'], random_state=seed_tf)
    tfs['pca'] = PCA(n_components=nb_comp['pca'], random_state=seed_tf)
    tfs['ica'] = FastICA(n_components=nb_comp['ica'], max_iter=250, random_state=seed_tf)
    tfs['grp'] = GaussianRandomProjection(n_components=nb_comp['grp'], eps=0.1, random_state=seed_tf)
    tfs['srp'] = SparseRandomProjection(n_components=nb_comp['srp'], dense_output=True, random_state=seed_tf)
    tfs['nmf'] = NMF(n_components=nb_comp['nmf'], shuffle=True, init='random', random_state=seed_tf) 
    #embedding
    trees, depth, leafs = 25, 8, 32 #2 ** 8 = 256
    embed = RandomTreesEmbedding(n_estimators=trees, 
                                 max_depth=depth, 
                                 max_leaf_nodes=leafs, 
                                 min_samples_split=32, 
                                 min_samples_leaf=8,
                                 sparse_output=False, 
                                 n_jobs=-1, random_state=seed_val)
        
    #feats and data
    feats = list(set(df_train.columns.tolist()).difference([target, target_id]))
    train_X = df_train[feats]
    test_X = df_test[feats]

    #preds
    train_preds = pd.DataFrame()
    test_preds = pd.DataFrame()
    test_preds[target_id] = test_id

    #start cv
    print('\n{0:02d}-fold cv'.format(nr_splits))
    for nr_fold in range(nr_splits):
        print('eval fold {:02d}'.format(nr_fold), flush=True)

        #split data into subset
        X_train = train_X.iloc[train_sets[nr_fold]].reset_index(drop=True)
        X_valid = train_X.iloc[valid_sets[nr_fold]].reset_index(drop=True)
        X_test = test_X.copy()
        #y
        y_train = train_y.iloc[train_sets[nr_fold]].reset_index(drop=True)
        y_valid = train_y.iloc[valid_sets[nr_fold]].reset_index(drop=True)
        #raw_y
        raw_y_train = raw_train_y.iloc[train_sets[nr_fold]].reset_index(drop=True)
        raw_y_valid = raw_train_y.iloc[valid_sets[nr_fold]].reset_index(drop=True)

        #pred df
        sub_train = pd.DataFrame()
        sub_train[target_id] = train_id.iloc[valid_sets[nr_fold]].tolist()

        #feats
        f_y_enc = f_hcc[:]
        f_cat = list(set(feats).difference(f_hcc))

        #using transformers
        for k, v in tfs.items():        
            X_train = tf_transform(X_train, method=v, stem=k, opt_fit=True, f_sel=f_cat)
            X_valid = tf_transform(X_valid, method=v, stem=k, opt_fit=False, f_sel=f_cat)
            X_test = tf_transform(X_test, method=v, stem=k, opt_fit=False, f_sel=f_cat)
            
        #embedding
        X_train = tf_transform(X_train, method=embed, stem='embed', opt_fit=True, f_sel=f_cat)
        X_valid = tf_transform(X_valid, method=embed, stem='embed', opt_fit=False, f_sel=f_cat)
        X_test = tf_transform(X_test, method=embed, stem='embed', opt_fit=False, f_sel=f_cat)

        #known cluster
        f = 'f_clu_max{:03d}'.format(n_clust)
        f_y_enc.append(f)
        X_train[f] = clust.fit_predict(X_train)
        X_valid[f] = clust.predict(X_valid)
        X_test[f] = clust.predict(X_test) 

        #encode y from factorization
        for f in f_y_enc:
            y_enc_df = pd.DataFrame({target: raw_y_train, f: X_train[f].values})
            rplc = np.median(raw_y_train)
            y_enc_dict = y_enc_df.groupby(f)[target].median().to_dict()
                
            f_m = 'f_y_enc_{}'.format(f)
            X_train[f_m] = X_train[f].apply(lambda x: y_enc_dict.get(x, rplc))
            X_valid[f_m] = X_valid[f].apply(lambda x: y_enc_dict.get(x, rplc))
            X_test[f_m] = X_test[f].apply(lambda x: y_enc_dict.get(x, rplc))

        #clean NA
        X_train = X_train.apply(np.nan_to_num)
        X_valid = X_valid.apply(np.nan_to_num)
        X_test = X_test.apply(np.nan_to_num)
                           
        #learning
        print(X_train.shape[1])
        for k, reg in reg_scikit.items():
            stem = k
            target_this = 'y_{}'.format(stem)
            reg.fit(X_train, y_train)
    
            preds = reg.predict(X_valid)
            score = r2_score(y_valid, preds)
            rmse = sqrt(mean_squared_error(y_valid, preds))
            print('{}: r2={:.6f}, rmse={:.3f}'.format(k, score, rmse), flush=True)      
            collect_predictions(preds, sub_train, d=nr_splits, col=target_this, opt_test=False)
            collect_predictions(reg.predict(X_test), test_preds, d=nr_splits, col=target_this, opt_test=True)
            
        #end of one fold eval in cv
        train_preds = train_preds.append(sub_train)
        del X_train, X_valid, X_test
        print(end='\n')
    
    #merge y into dataset
    train_preds = train_preds.reset_index(drop=True)
    df_train = df_train.merge(train_preds, how='left', on=target_id)
    df_test = df_test.merge(test_preds, how='left', on=target_id)        

    #performance check for each classifier
    print('\nSummary', flush=True)
    f_preds = [f for f in df_train.columns.tolist() if f.startswith('y_')]
    collect_r2, collect_rmse = {}, {}
    for t in f_preds:
        collect_r2[t] = r2_score(raw_train_y, df_train[t].values)
        collect_rmse[t] = sqrt(mean_squared_error(raw_train_y, df_train[t].values))
        
        score, rmse = collect_r2.get(t, 0), collect_rmse.get(t, 0)
        print('{}: r2={:.6f}, rmse={:.3f}'.format(t[2:], score, rmse), flush=True)
        file = '{}_{}_s{:.5f}_e{:.3f}.csv'.format(tmstmp, t[2:], score, rmse)
        write_csv(df_test[target_id].tolist(), df_test[t].tolist(), stem=file)

    #optimizing weights
    print('\nOptimizing weights', flush=True)
    best_w, best_r2, best_rmse = optimize_weights(trials=10000, y=raw_train_y, df=df_train[f_preds + [target, target_id]].copy(), f_sel=f_preds)

    #
    f_preds = [f for f in f_preds if f in df_test.columns.tolist()]
    #print optimal weights
    print('\nOptimized weights', flush=True)
    for i, t in enumerate(f_preds):
        stdout = 'w {}: {:.3f}, r2 = {:.5f}, rmse = {:.3f}'.format(t[2:], best_w.get(t, 0), collect_r2.get(t, 0), collect_rmse.get(t, 0))
        print(stdout, flush=True)
    #save weighted
    file = '{0}_wsum_s{1:.5f}_e{2:.3f}.csv'.format(tmstmp, best_r2, best_rmse)
    preds = (df_test[f_preds] * pd.Series(best_w)).sum(1)
    write_csv(df_test[target_id].tolist(), preds.tolist(), stem=file)
