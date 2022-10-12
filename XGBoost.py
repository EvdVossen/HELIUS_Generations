#! /usr/bin/python
import os
import numpy as np
import time
import pandas as pd
import matplotlib.pylab as plt

from copy import deepcopy
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, cross_val_score, LeaveOneOut
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
from sklearn.feature_selection import SelectPercentile,f_classif
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

dir1 ='/home/user/project/Machine_learning/'
os.chdir(dir1)

try:
    os.mkdir("output_data/")
except OSError:
    print("Directory already exists")

def raise_error(error_msg):
    exit('ERROR!: '+error_msg)

def loo_cross_validation(X, y, uni_feat_percent, feat_names, preds, mean_fpr, rand_seed=0):
    splitter = LeaveOneOut()
    y_train_preds = []
    y_preds = []
    y_preds_class = []

    param_grid_xgboost = {'max_depth': [2, 5, 7],
                          'learning_rate': [0.01, 0.1],
                          'n_estimators': [100, 300, 800, 1000],
                          'min_child_weight': [1, 5],
                          'gamma': [0.5, 2],
                          'subsample': [0.5, 0.6, 0.8],
                          'colsample_bytree': [0.6, 0.8, 1.0]
                          }

    print("Amount of features included in the model: "+str(X.shape[1]))

    for train, test in splitter.split(X, y):
        X_train, X_test = X[train, :], X[test, :]
        y_train, y_test = y[train], y[test]

        model = XGBClassifier(n_jobs=-1, objective='reg:squarederror', tree_method='auto')
        param_grid = param_grid_xgboost

        skf_xgb = StratifiedShuffleSplit(train_size=0.8, test_size=0.2, n_splits=3, random_state=rand_seed)

        random_search = RandomizedSearchCV(model, param_grid,
                                           n_jobs=-1,
                                           cv=skf_xgb, scoring='roc_auc',
                                           verbose=1, n_iter=10, random_state=rand_seed,
                                           refit=True)

        random_search.fit(X_train, y_train.ravel())

        best_model = random_search.best_estimator_

        imps = best_model.feature_importances_

        preds['feat_imps'].append(imps)
        preds['feat_names'].append(feat_names)

        y_train_preds = best_model.predict_proba(X_train)[:, 1]
        y_preds.append(best_model.predict_proba(X_test)[:, 1]) 
        y_preds_class.append(best_model.predict(X_test))

        score_train = roc_auc_score(y_train, y_train_preds)
        cv_score = cross_val_score(best_model, X_train, y_train, cv=skf_xgb, scoring="roc_auc")

        fpr_tr, tpr_tr, _ = roc_curve(y_train, y_train_preds, pos_label=1)
        preds['score_list_train']['list_tprs_train'].append(np.interp(mean_fpr, fpr_tr, tpr_tr))
        preds['score_list_train']['list_tprs_train'][-1][0] = 0.0
        preds['score_list_train']['auc_train'].append(score_train)

        preds['score_list_cv'].append(cv_score)

    score = roc_auc_score(y, y_preds)
    fpr, tpr, _ = roc_curve(y, y_preds, pos_label=1)
    preds['score_list']['list_tprs'].append(np.interp(mean_fpr, fpr, tpr))
    preds['score_list']['list_tprs'][-1][0] = 0.0
    preds['score_list']['auc'].append(score)

    preds['preds'].append(y_preds)
    preds['y_test'].append(y)

    preds['y_train_preds'].append(y_train_preds)

    print('*' * 15)
    print(f'\n Confusion matrix:\n {confusion_matrix(y, y_preds_class)}\n')

    print(f'\n ROC_AUC:\n {score}\n')
    print('*' * 15)

    return preds

def average_loo(X, y, uni_feat_percent, feat_names, preds, results_path, mean_fpr, 
                y_df, n_runs=20, subset_frac=0.5, rand_seed=0):

    rus = RandomUnderSampler(replacement=False)
    preds_loo = deepcopy(preds)
    np.random.seed(rand_seed)

    for run in range(n_runs):

        X_resampled, y_resampled = rus.fit_resample(X, y)
        y_resampled_df = pd.DataFrame(y_resampled, columns = ["CLASS_Good"])
        random_selection = y_resampled_df.groupby("CLASS_Good").sample(frac=subset_frac).index

        print(f'\nPerforming random undersampling (shuffle {run+1}):')
        print('Total number of subjects after undersampling:')
        print(f' {np.vstack(np.unique([tuple(row) for row in X_resampled], axis=0)).shape[0]}')
        print(f' {sorted(Counter(y_resampled).items())}')
        print(f'\nNumber of randomly selected subjects included:\n {random_selection.values.shape[0]}')


        loo_cross_validation(X = X_resampled[random_selection, :], y = y_resampled[random_selection], 
                             uni_feat_percent = uni_feat_percent, feat_names = feat_names, 
                             preds = preds_loo, mean_fpr = mean_fpr)

    np.save(results_path + '_loo', preds_loo)

    return preds_loo

def main():
    startTime = time.time()
    print('Data loading \n')
    np.random.seed(256)
    stability_samples_run_number=20
    n_features=100
    subset_frac = 0.5

    X_train_df=pd.read_excel(io='X_train.xlsx',
                             index_col='SAMPLE_ID')
    y_train_df=pd.read_excel(io='Y_train.xlsx',
                             index_col='SAMPLE_ID')
    X_train_df.fillna(method='ffill',inplace=True)

    uni_feat_percent=(100/X_train_df.shape[1])*n_features

    X_raw=np.asarray(X_train_df.values,dtype=float)
    y_float=np.asarray(y_train_df.iloc[:,0].values,dtype=float)
    y_int=np.asarray(y_float,dtype=int)

    b = SelectPercentile(score_func=f_classif, percentile = uni_feat_percent)
    b.fit(X_raw,y_int)
    X=X_raw[:, b.get_support(indices=True)]
    y_df=y_train_df.reset_index(drop=True)

    feat_names_full=np.asarray(list(X_train_df.columns),dtype=str)
    feat_names=feat_names_full[b.get_support(indices=True)]

    mean_fpr = np.linspace(0, 1, 100)

    running = True

    results_path = os.path.join(os.getcwd(), 'output_data', 'results_dict')

    if running:
        score_list_dict = {'auc': [], 'list_tprs': []}
        score_list_dict_train = {'auc_train': [], 'list_tprs_train': []}

        results_dict = {'score_list': deepcopy(score_list_dict), 'preds': [], 'feat_imps': [], 'y_test': [], 
                        'score_list_train': deepcopy(score_list_dict_train), 'y_train_preds':[],
                        'feat_names':[], 'score_list_cv': []}

        results_dict = average_loo(X, y_int, uni_feat_percent, feat_names, results_dict,
                    results_path, mean_fpr, y_df = y_df,
                    n_runs=stability_samples_run_number, subset_frac=subset_frac)

    else:
        results_dict = np.load(results_path + '_loo.npy', allow_pickle=True).all()

    list_feat_importances = results_dict['feat_imps']
    tprs = results_dict['score_list']['list_tprs']
    aucs_test =  results_dict['score_list']['auc']

    feat_names = results_dict['feat_names']
    fi = pd.DataFrame(np.array(list_feat_importances))
    fi = pd.Series(fi.values.ravel('F'), name="feat_imps")
    feat_names_dict = pd.DataFrame(np.array(feat_names))
    feat_names_dict = pd.Series(feat_names_dict.values.ravel('F'), name="feat_names")

    fi = pd.concat([feat_names_dict, fi], axis = 1)
    fi_avg = fi.groupby(["feat_names"], as_index=False).mean()

    feature_importance = fi_avg['feat_imps']

    ### Make importances relative to max importance
    feature_importance = np.round(100.0 * (feature_importance / np.nanmax(feature_importance)),2)

    ### Test AUC
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random classifier', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs_test)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('output_data/auc_avg.pdf', bbox_inches='tight')
    plt.close()

    header=np.reshape(np.asarray(['FeatName','RelFeatImp']),(1,2))
    feat_imp_vector=pd.concat([fi_avg["feat_names"], feature_importance], axis = 1)
    feat_imp_vector=feat_imp_vector.sort_values(by="feat_imps", ascending=False)
    feat_vector_save=np.vstack((header,feat_imp_vector))

    np.savetxt('output_data/feat_imp_gini.txt',feat_vector_save,fmt='%s',delimiter='\t')

    ### AUC-information    
    auc_information = (["Train AUC mean" , round(np.mean(results_dict['score_list_train']['auc_train']),2)],
                       ["Train AUC sd", round(np.std(results_dict['score_list_train']['auc_train']),2)],
                       ["CV AUC mean" , round(np.mean(results_dict['score_list_cv']),2)],
                       ["CV AUC sd", round(np.std(results_dict['score_list_cv']),2)],
                       ["Test AUC mean" , round(np.mean(aucs_test),2)],
                       ["Test AUC sd", round(np.std(aucs_test),2)])

    np.savetxt('output_data/AUC_information.txt',auc_information,fmt='%s',delimiter='\t')

    endTime = time.time()
    runTime=endTime-startTime
    print(('Runtime: %.2f seconds' %runTime))

if __name__ == "__main__":
    __spec__ = None
    dummy=main()
