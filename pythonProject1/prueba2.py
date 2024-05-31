import pandas as pd
import pybnesian as pb
import numpy as np
import math
import normal_mixture_density_functions
import time
import trial_density_functions

results=pd.DataFrame(columns=['bandwidth','N','function','time','holdout_likelihood'])
operators=pb.OperatorPool([pb.ArcOperatorSet(),pb.ChangeNodeTypeSet()])
x = pb.GreedyHillClimbing()
ucv= pb.UCV()
bcv = pb.BCV()
scv = pb.SCV()
PI = pb.PI()

for i in range(0,100):
    result_ucv = ['kdenetwork_ucv', 100, 'kurtotic']
    sample = normal_mixture_density_functions.sampling_kurtotic(100)
    # sample = trial_density_functions.sampling_F_1(500)
    holdout_likelihood = pb.HoldoutLikelihood(sample)
    training_data = holdout_likelihood.training_data()
    test_data = holdout_likelihood.test_data()
    KDENetwork_ucv = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_ucv.fit_wbm(training_data, ucv)
    KDENetwork_ucv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_ucv,
                                    m_bselector=ucv, epsilon=0.01)
    KDENetwork_ucv.fit_wbm(training_data, ucv)
    t_fin = time.time()
    result_ucv.append(t_fin - t_0)

    result_nr = ['kde_network_normal_rule', 100, 'kurtotic']
    KDENetwork_nr = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_nr.fit(training_data)
    KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                               start=KDENetwork_nr, epsilon=0.01)
    KDENetwork_nr.fit(training_data)
    t_fin = time.time()
    result_nr.append(t_fin - t_0)

    result_SP_ucv = ['SP_network_ucv', 100, 'kurtotic']
    SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_ucv.fit_wbm(training_data, ucv)
    SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                    start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
    SPBNetwork_ucv.fit_wbm(training_data, ucv)
    t_fin = time.time()
    result_SP_ucv.append(t_fin - t_0)

    result_SP_nr = ['SP_network_normal_rule', 100, 'kurtotic']
    SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_nr.fit(training_data)
    SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                               start=SPBNetwork_nr, epsilon=0.01)
    SPBNetwork_nr.fit(training_data)
    t_fin = time.time()
    result_SP_nr.append(t_fin - t_0)

    result_bcv = ['kdenetwork_bcv', 100, 'kurtotic']
    KDENetwork_bcv = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_bcv.fit_wbm(training_data, bcv)
    KDENetwork_bcv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_bcv,
                                    m_bselector=bcv, epsilon=0.01)
    KDENetwork_bcv.fit_wbm(training_data, bcv)
    t_fin = time.time()
    result_bcv.append(t_fin - t_0)

    result_SP_bcv = ['SP_network_bcv', 100, 'kurtotic']
    SPBNetwork_bcv = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_bcv.fit_wbm(training_data, bcv)
    SPBNetwork_bcv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                    start=SPBNetwork_bcv, m_bselector=bcv, epsilon=0.01)
    SPBNetwork_bcv.fit_wbm(training_data, bcv)
    t_fin = time.time()
    result_SP_bcv.append(t_fin - t_0)

    result_scv = ['kdenetwork_scv', 100, 'kurtotic']
    KDENetwork_scv = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_scv.fit_wbm(training_data, scv)
    KDENetwork_scv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_scv,
                                    m_bselector=scv, epsilon=0.01)
    KDENetwork_scv.fit_wbm(training_data, scv)
    t_fin = time.time()
    result_scv.append(t_fin - t_0)

    result_SP_scv = ['SP_network_scv', 100, 'kurtotic']
    SPBNetwork_scv = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_scv.fit_wbm(training_data, scv)
    SPBNetwork_scv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                    start=SPBNetwork_scv, m_bselector=scv, epsilon=0.01)
    SPBNetwork_scv.fit_wbm(training_data, scv)
    t_fin = time.time()
    result_SP_scv.append(t_fin - t_0)

    result_PI = ['kdenetwork_PI', 100, 'kurtotic']
    KDENetwork_PI = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_PI.fit_wbm(training_data, PI)
    KDENetwork_PI = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_PI,
                                   m_bselector=PI, epsilon=0.01)
    KDENetwork_PI.fit_wbm(training_data, PI)
    t_fin = time.time()
    result_PI.append(t_fin - t_0)

    result_SP_PI = ['SP_network_PI', 100, 'kurtotic']
    SPBNetwork_PI = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_PI.fit_wbm(training_data, PI)
    SPBNetwork_PI = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_PI, m_bselector=PI, epsilon=0.01)
    SPBNetwork_PI.fit_wbm(training_data, PI)
    t_fin = time.time()
    result_SP_PI.append(t_fin - t_0)

    result_ucv.append(KDENetwork_ucv.slogl(test_data))
    result_bcv.append(KDENetwork_bcv.slogl(test_data))
    result_scv.append(KDENetwork_bcv.slogl(test_data))
    result_PI.append(KDENetwork_bcv.slogl(test_data))
    result_nr.append(KDENetwork_nr.slogl(test_data))
    result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
    result_SP_bcv.append(SPBNetwork_bcv.slogl(test_data))
    result_SP_scv.append(SPBNetwork_scv.slogl(test_data))
    result_SP_PI.append(SPBNetwork_PI.slogl(test_data))
    result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

    results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_bcv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_scv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_PI], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_bcv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_scv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_PI], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])

    result_ucv = ['kdenetwork_ucv', 100, 'kurtotic']
    sample = normal_mixture_density_functions.sampling_separated_bimodal(100)
    # sample = trial_density_functions.sampling_F_1(500)
    holdout_likelihood = pb.HoldoutLikelihood(sample)
    training_data = holdout_likelihood.training_data()
    test_data = holdout_likelihood.test_data()
    KDENetwork_ucv = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_ucv.fit_wbm(training_data, ucv)
    KDENetwork_ucv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_ucv,
                                    m_bselector=ucv, epsilon=0.01)
    KDENetwork_ucv.fit_wbm(training_data, ucv)
    t_fin = time.time()
    result_ucv.append(t_fin - t_0)

    result_nr = ['kde_network_normal_rule', 100, 'separated_bimodal']
    KDENetwork_nr = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_nr.fit(training_data)
    KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                               start=KDENetwork_nr, epsilon=0.01)
    KDENetwork_nr.fit(training_data)
    t_fin = time.time()
    result_nr.append(t_fin - t_0)

    result_SP_ucv = ['SP_network_ucv', 100, 'separated_bimodal']
    SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_ucv.fit_wbm(training_data, ucv)
    SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                    start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
    SPBNetwork_ucv.fit_wbm(training_data, ucv)
    t_fin = time.time()
    result_SP_ucv.append(t_fin - t_0)

    result_SP_nr = ['SP_network_normal_rule', 100, 'separated_bimodal']
    SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_nr.fit(training_data)
    SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                               start=SPBNetwork_nr, epsilon=0.01)
    SPBNetwork_nr.fit(training_data)
    t_fin = time.time()
    result_SP_nr.append(t_fin - t_0)

    result_bcv = ['kdenetwork_bcv', 100, 'separated_bimodal']
    KDENetwork_bcv = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_bcv.fit_wbm(training_data, bcv)
    KDENetwork_bcv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_bcv,
                                    m_bselector=bcv, epsilon=0.01)
    KDENetwork_bcv.fit_wbm(training_data, bcv)
    t_fin = time.time()
    result_bcv.append(t_fin - t_0)

    result_SP_bcv = ['SP_network_bcv', 100, 'separated_bimodal']
    SPBNetwork_bcv = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_bcv.fit_wbm(training_data, bcv)
    SPBNetwork_bcv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                    start=SPBNetwork_bcv, m_bselector=bcv, epsilon=0.01)
    SPBNetwork_bcv.fit_wbm(training_data, bcv)
    t_fin = time.time()
    result_SP_bcv.append(t_fin - t_0)

    result_scv = ['kdenetwork_scv', 100, 'separated_bimodal']
    KDENetwork_scv = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_scv.fit_wbm(training_data, scv)
    KDENetwork_scv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_scv,
                                    m_bselector=scv, epsilon=0.01)
    KDENetwork_scv.fit_wbm(training_data, scv)
    t_fin = time.time()
    result_scv.append(t_fin - t_0)

    result_SP_scv = ['SP_network_scv', 100, 'separated_bimodal']
    SPBNetwork_scv = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_scv.fit_wbm(training_data, scv)
    SPBNetwork_scv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                    start=SPBNetwork_scv, m_bselector=scv, epsilon=0.01)
    SPBNetwork_scv.fit_wbm(training_data, scv)
    t_fin = time.time()
    result_SP_scv.append(t_fin - t_0)

    result_PI = ['kdenetwork_PI', 100, 'separated_bimodal']
    KDENetwork_PI = pb.KDENetwork(list(sample.columns))
    t_0 = time.time()
    KDENetwork_PI.fit_wbm(training_data, PI)
    KDENetwork_PI = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_PI,
                                   m_bselector=PI, epsilon=0.01)
    KDENetwork_PI.fit_wbm(training_data, PI)
    t_fin = time.time()
    result_PI.append(t_fin - t_0)

    result_SP_PI = ['SP_network_PI', 100, 'separated_bimodal']
    SPBNetwork_PI = pb.SemiparametricBN(list(sample.columns))
    t_0 = time.time()
    SPBNetwork_PI.fit_wbm(training_data, PI)
    SPBNetwork_PI = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_PI, m_bselector=PI, epsilon=0.01)
    SPBNetwork_PI.fit_wbm(training_data, PI)
    t_fin = time.time()
    result_SP_PI.append(t_fin - t_0)

    result_ucv.append(KDENetwork_ucv.slogl(test_data))
    result_bcv.append(KDENetwork_bcv.slogl(test_data))
    result_scv.append(KDENetwork_bcv.slogl(test_data))
    result_PI.append(KDENetwork_bcv.slogl(test_data))
    result_nr.append(KDENetwork_nr.slogl(test_data))
    result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
    result_SP_bcv.append(SPBNetwork_bcv.slogl(test_data))
    result_SP_scv.append(SPBNetwork_scv.slogl(test_data))
    result_SP_PI.append(SPBNetwork_PI.slogl(test_data))
    result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

    results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_bcv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_scv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_PI], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_bcv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_scv], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_PI], columns=results.columns)])
    results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])



results.to_csv('results.csv')

result = results.groupby(['bandwidth', 'function'])['holdout_likelihood'].mean()

print(result)