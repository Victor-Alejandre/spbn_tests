import pandas as pd
import pybnesian as pb
from sklearn.model_selection import KFold, StratifiedKFold
import time
import numpy as np
import math
import scipy
import openpyxl


def accuracy(bn, test_data, classes):
    # Ensure the 'Class' column in test_data is categorical with the specified classes
    data = test_data.copy()
    data['Class'] = pd.Categorical(data['Class'], categories=classes)
    solution = data['Class']
    data = data.drop(columns=['Class'])
    N = data.shape[0]
    output = []

    for i in range(N):
        p_s = {}
        for outcome in classes:
            x = data.iloc[i].copy()  # Make a copy to avoid modifying the original series
            # Create a DataFrame for the row, assigning 'Class' the categorical outcome
            x = x.to_frame().T  # Convert Series to DataFrame
            x['Class'] = pd.Categorical([outcome], categories=classes)  # Ensure 'Class' is categorical
            cols = ['Class'] + [col for col in x.columns if col != 'Class']
            x = x[cols]
            p_s[outcome] = math.exp(bn.logl(x)[0])

        # Append the predicted class to the output list
        output.append(max(p_s, key=lambda k: p_s[k]))

    return sum(x == y for x, y in zip(output, solution)) / len(solution)






glass_df = pd.read_csv('uci_datasets/glass_identification/glass.data', header=None)
glass_df = glass_df.drop(columns=[0])
glass_df.columns = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
glass_df['Class'] = glass_df['Class'].astype(str)
glass_df['Class'] = glass_df['Class'].astype('category')


breast_df = pd.read_excel('uci_datasets/BreastTissue.xlsx', sheet_name='Data')
breast_df = breast_df.drop(columns = ['Case #'])
breast_df['Class'] = breast_df['Class'].astype('category')

yeast_df = pd.read_csv('uci_datasets/yeast/yeast.data', sep='  ', header=None)
yeast_df = yeast_df.drop(columns=[0])
yeast_df.columns = ['mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'Class']
yeast_df['Class'] = yeast_df['Class'].astype(str)
yeast_df['Class'] = yeast_df['Class'].astype('category')
features = list(yeast_df.columns)
features.remove('Class')

breast_df_tree_augmented_G_0 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                                 nodes = list(breast_df.columns),
                                                 node_types = [('Class', pb.DiscreteFactorType()), ('I0', pb.LinearGaussianCPDType()),
                                                               ('PA500', pb.LinearGaussianCPDType()), ('HFS', pb.LinearGaussianCPDType()),
                                                               ('DA', pb.LinearGaussianCPDType()), ('Area', pb.LinearGaussianCPDType()),
                                                               ('A/DA', pb.LinearGaussianCPDType()), ('Max_IP', pb.LinearGaussianCPDType()),
                                                               ('DR', pb.LinearGaussianCPDType()), ('P', pb.LinearGaussianCPDType())])
breast_df_tree_augmented_G_1 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                                 nodes = list(breast_df.columns),
                                                 node_types = [('Class', pb.DiscreteFactorType()), ('I0', pb.CKDEType()),
                                                               ('PA500', pb.CKDEType()), ('HFS', pb.CKDEType()),
                                                               ('DA', pb.CKDEType()), ('Area', pb.CKDEType()),
                                                               ('A/DA', pb.CKDEType()), ('Max_IP', pb.CKDEType()),
                                                               ('DR', pb.CKDEType()), ('P', pb.CKDEType())])
glass_df_tree_augmented_G_0 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                                 nodes = list(glass_df.columns),
                                                 node_types = [('Class', pb.DiscreteFactorType()), ('RI', pb.CKDEType()),
                                                               ('Na', pb.LinearGaussianCPDType()), ('Mg', pb.CKDEType()),
                                                               ('Al', pb.LinearGaussianCPDType()), ('Si', pb.CKDEType()),
                                                               ('K', pb.LinearGaussianCPDType()), ('Ca', pb.LinearGaussianCPDType()),
                                                               ('Ba', pb.LinearGaussianCPDType()), ('Fe', pb.LinearGaussianCPDType())])
glass_df_tree_augmented_G_1 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                                 nodes = list(glass_df.columns),
                                                 node_types = [('Class', pb.DiscreteFactorType()), ('RI', pb.CKDEType()),
                                                               ('Na', pb.CKDEType()), ('Mg', pb.CKDEType()),
                                                               ('Al', pb.CKDEType()), ('Si', pb.CKDEType()),
                                                               ('K', pb.CKDEType()), ('Ca', pb.CKDEType()),
                                                               ('Ba', pb.CKDEType()), ('Fe', pb.CKDEType())])
node_type = [('Class', pb.DiscreteFactorType())]
node_type.extend([(variable, pb.LinearGaussianCPDType()) for variable in features])
yeast_df_tree_augmented_G_0 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                                 nodes = list(yeast_df.columns),
                                                 node_types = node_type)
node_type = [('Class', pb.DiscreteFactorType())]
node_type.extend([(variable, pb.CKDEType()) for variable in features])
yeast_df_tree_augmented_G_1 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                                 nodes = list(yeast_df.columns),
                                                 node_types = node_type)

'''y = glass_df['Class']
features = list(glass_df.columns)
features.remove('Class')
X = glass_df[features]'''
y = yeast_df['Class']
X = yeast_df[features]
#y = breast_df['Class']
#features = list(breast_df.columns)
#features.remove('Class')
#X = breast_df[features]
#kf = KFold(n_splits=10, shuffle=True, random_state=1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
operators=pb.OperatorPool([pb.ArcOperatorSet(),pb.ChangeNodeTypeSet()])
results =  pd.DataFrame(columns = ['initial network', 'learning method', 'log_likelihood', 'accuracy'])

for train_index, test_index in skf.split(X, y):
    X_train, X_test = yeast_df.iloc[train_index], yeast_df.iloc[test_index]
    result_SP_nr_G_0 = ['G_0', 'hc_nr']
    SPBNetwork_nr_G_0 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                           nodes = list(yeast_df.columns))
    x = pb.GreedyHillClimbing()
    '''
    SPBNetwork_nr_G_0 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=10),
                                   start=glass_df_tree_augmented_G_0, epsilon=0.01,
                                   arc_whitelist = [('Class', 'RI'), ('Class', 'Na'), ('Class', 'Mg'), ('Class', 'Al'), ('Class', 'Si'),
                                                    ('Class', 'K'), ('Class', 'Ca'), ('Class', 'Ba'), ('Class', 'Fe')])
    
    SPBNetwork_nr_G_0 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(X_train, test_ratio = 0.3, seed=4),
                                   start=breast_df_tree_augmented_G_0, epsilon=0.01,
                                   arc_whitelist = [('Class', 'I0'), ('Class', 'PA500'), ('Class', 'HFS'),
                                                    ('Class', 'DA'), ('Class', 'Area'),
                                                    ('Class', 'A/DA'), ('Class', 'Max_IP'), ('Class', 'DR'), ('Class', 'P')])
'''
    SPBNetwork_nr_G_0 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(X_train, test_ratio=0.3, seed=4),
                                   start=yeast_df_tree_augmented_G_0, epsilon=0.01,
                                   arc_whitelist=[('Class', variable) for variable in features])
    SPBNetwork_nr_G_0.fit(X_train)
    slog = SPBNetwork_nr_G_0.slogl(X_test)
    result_SP_nr_G_0.append(slog)
    result_SP_nr_G_0.append(accuracy(SPBNetwork_nr_G_0, X_test, X_train['Class'].cat.categories.tolist()))

    result_SP_nr_G_1 = ['G_1', 'hc_nr']
    SPBNetwork_nr_G_1 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                           nodes = list(yeast_df.columns))
    x = pb.GreedyHillClimbing()
    '''
    SPBNetwork_nr_G_1 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=10),
                                   start=glass_df_tree_augmented_G_1, epsilon=0.01,
                                   arc_whitelist = [('Class', 'RI'), ('Class', 'Na'), ('Class', 'Mg'), ('Class', 'Al'), ('Class', 'Si'),
                                                    ('Class', 'K'), ('Class', 'Ca'), ('Class', 'Ba'), ('Class', 'Fe')])
    
    SPBNetwork_nr_G_1 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=1),
                                   start=breast_df_tree_augmented_G_1, epsilon=0.01,
                                   arc_whitelist=[('Class', 'I0'), ('Class', 'PA500'), ('Class', 'HFS'),
                                                  ('Class', 'DA'), ('Class', 'Area'),
                                                  ('Class', 'A/DA'), ('Class', 'Max_IP'), ('Class', 'DR'),
                                                  ('Class', 'P')])
                                                  '''
    SPBNetwork_nr_G_1 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(X_train, test_ratio=0.3, seed=4),
                                   start=yeast_df_tree_augmented_G_1, epsilon=0.01,
                                   arc_whitelist=[('Class', variable) for variable in features])
    SPBNetwork_nr_G_1.fit(X_train)
    slog = SPBNetwork_nr_G_1.slogl(X_test)
    result_SP_nr_G_1.append(slog)
    result_SP_nr_G_1.append(accuracy(SPBNetwork_nr_G_1, X_test, X_train['Class'].cat.categories.tolist()))

    result_SP_ucv_G_0 = ['G_0', 'hc_ucv']
    SPBNetwork_ucv_G_0 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                           nodes = list(yeast_df.columns))
    x = pb.GreedyHillClimbing()
    '''
    SPBNetwork_ucv_G_0 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=10),
                                        start=glass_df_tree_augmented_G_0, m_bselector=pb.UCV(), epsilon=0.01,
                                        arc_whitelist=[('Class', 'RI'), ('Class', 'Na'), ('Class', 'Mg'),
                                                       ('Class', 'Al'), ('Class', 'Si'),
                                                       ('Class', 'K'), ('Class', 'Ca'), ('Class', 'Ba'),
                                                       ('Class', 'Fe')]
                                        )
    
    SPBNetwork_ucv_G_0 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=1),
                                        start=breast_df_tree_augmented_G_0, m_bselector=pb.UCV(), epsilon=0.01,
                                        arc_whitelist=[('Class', 'I0'), ('Class', 'PA500'), ('Class', 'HFS'),
                                                       ('Class', 'DA'), ('Class', 'Area'),
                                                       ('Class', 'A/DA'), ('Class', 'Max_IP'), ('Class', 'DR'),
                                                       ('Class', 'P')])
                                                       '''
    SPBNetwork_ucv_G_0 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=10),
                                        start=yeast_df_tree_augmented_G_0, m_bselector=pb.UCV(), epsilon=0.01,
                                       arc_whitelist=[('Class', variable) for variable in features]
                                        )
    SPBNetwork_ucv_G_0.fit_wbm(X_train, pb.UCV())
    slog = SPBNetwork_ucv_G_0.slogl(X_test)
    result_SP_ucv_G_0.append(slog)
    result_SP_ucv_G_0.append(accuracy(SPBNetwork_ucv_G_0, X_test, X_train['Class'].cat.categories.tolist()))

    result_SP_ucv_G_1 = ['G_1', 'hc_ucv']
    SPBNetwork_ucv_G_1 = pb.HeterogeneousBN(factor_type = [pb.CKDEType(), pb.LinearGaussianCPDType(), pb.DiscreteFactorType()],
                                           nodes = list(yeast_df.columns))
    x = pb.GreedyHillClimbing()
    '''
    SPBNetwork_ucv_G_1 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=10),
                                        start=glass_df_tree_augmented_G_1, m_bselector=pb.UCV(), epsilon=0.01,
                                        arc_whitelist=[('Class', 'RI'), ('Class', 'Na'), ('Class', 'Mg'),
                                                       ('Class', 'Al'), ('Class', 'Si'),
                                                       ('Class', 'K'), ('Class', 'Ca'), ('Class', 'Ba'),
                                                       ('Class', 'Fe')]
                                        )
    
    SPBNetwork_ucv_G_1 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=1),
                                        start=breast_df_tree_augmented_G_1, m_bselector=pb.UCV(), epsilon=0.01,
                                        arc_whitelist=[('Class', 'I0'), ('Class', 'PA500'), ('Class', 'HFS'),
                                                       ('Class', 'DA'), ('Class', 'Area'),
                                                       ('Class', 'A/DA'), ('Class', 'Max_IP'), ('Class', 'DR'),
                                                       ('Class', 'P')]
                                        )'''
    SPBNetwork_ucv_G_1 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(X_train, seed=10),
                                        start=yeast_df_tree_augmented_G_1, m_bselector=pb.UCV(), epsilon=0.01,
                                        arc_whitelist=[('Class', variable) for variable in features]
                                        )
    SPBNetwork_ucv_G_1.fit_wbm(X_train, pb.UCV())
    slog = SPBNetwork_ucv_G_1.slogl(X_test)
    result_SP_ucv_G_1.append(slog)
    result_SP_ucv_G_1.append(accuracy(SPBNetwork_ucv_G_1, X_test, X_train['Class'].cat.categories.tolist()))

    results = pd.concat([results, pd.DataFrame([result_SP_ucv_G_0], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_ucv_G_1], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_nr_G_0], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_nr_G_1], columns=results.columns)])


print(results.groupby(['initial network', 'learning method'])[['log_likelihood', 'accuracy']].mean())
results.to_csv('results_exp_tan_yeast_df.csv')