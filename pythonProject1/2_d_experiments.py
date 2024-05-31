import pandas as pd
import pybnesian as pb
import numpy as np
import math
import normal_mixture_density_functions
import time
import trial_density_functions
from functools import partial
import matplotlib.pyplot as plt
from mise_functions import mise, amise

def kde(x,KDE):
    col=[]
    for i in range(x.shape[0]):
        col.append(f'x{i+1}')
    loglikelihood=KDE.logl(pd.DataFrame([x],columns=col))

    return math.exp(loglikelihood[0])





if __name__=='__main__':
    results=pd.DataFrame(columns=['bandwidth','N','function','time','holdout_likelihood'])
    operators=pb.OperatorPool([pb.ArcOperatorSet(),pb.ChangeNodeTypeSet()])
    x = pb.GreedyHillClimbing()
    ucv= pb.UCV()
    for N in [100, 1000, 10000]:

        result_ucv=['kdenetwork_ucv',N,'kurtotic']
        sample = normal_mixture_density_functions.sampling_kurtotic(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
        training_data=holdout_likelihood.training_data()
        test_data=holdout_likelihood.test_data()
        KDENetwork_ucv = pb.KDENetwork(list(sample.columns))
        t_0=time.time()
        KDENetwork_ucv.fit_wbm(training_data,  ucv)
        KDENetwork_ucv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_ucv, m_bselector= ucv, epsilon=0.01)
        KDENetwork_ucv.fit_wbm(training_data,  ucv)
        t_fin=time.time()
        result_ucv.append(t_fin-t_0)

        result_nr = ['kde_network_normal_rule', N, 'kurtotic']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                        start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv= ['SP_network_ucv', N, 'kurtotic']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'kurtotic']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)


        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)


        result_ucv=['kdenetwork_ucv',N,'bimodal']
        sample = normal_mixture_density_functions.sampling_bimodal(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
        training_data=holdout_likelihood.training_data()
        test_data=holdout_likelihood.test_data()
        KDENetwork_ucv = pb.KDENetwork(list(sample.columns))
        t_0=time.time()
        KDENetwork_ucv.fit_wbm(training_data,  ucv)
        KDENetwork_ucv = x.estimate_wbm(operators=pb.ArcOperatorSet(), score=holdout_likelihood, start=KDENetwork_ucv, m_bselector= ucv, epsilon=0.01)
        KDENetwork_ucv.fit_wbm(training_data,  ucv)
        t_fin=time.time()
        result_ucv.append(t_fin-t_0)

        result_nr = ['kde_network_normal_rule', N, 'bimodal']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                        start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv= ['SP_network_ucv', N, 'bimodal']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'bimodal']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)


        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)

        result_ucv = ['kdenetwork_ucv', N, 'trimodal']
        sample = normal_mixture_density_functions.sampling_trimodal(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
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

        result_nr = ['kde_network_normal_rule', N, 'trimodal']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                   start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv = ['SP_network_ucv', N, 'trimodal']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'trimodal']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)

        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)

        result_ucv = ['kdenetwork_ucv', N, 'separated_bimodal']
        sample = normal_mixture_density_functions.sampling_separated_bimodal(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
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

        result_nr = ['kde_network_normal_rule', N, 'separated_bimodal']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                   start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv = ['SP_network_ucv', N, 'separated_bimodal']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'separated_bimodal']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)

        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)

        result_ucv = ['kdenetwork_ucv', N, 'asymmetric_bimodal']
        sample = normal_mixture_density_functions.sampling_asymmetric_bimodal(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
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

        result_nr = ['kde_network_normal_rule', N, 'asymmetric_bimodal']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                   start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv = ['SP_network_ucv', N, 'asymmetric_bimodal']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'asymmetric_bimodal']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)

        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)

        result_ucv = ['kdenetwork_ucv', N, 'double_fountain']
        sample = normal_mixture_density_functions.sampling_double_fountain(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
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

        result_nr = ['kde_network_normal_rule', N, 'double_fountain']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                   start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv = ['SP_network_ucv', N, 'double_fountain']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'double_fountain']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)

        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)

        result_ucv = ['kdenetwork_ucv', N, 'F_1']
        sample = trial_density_functions.sampling_F_1(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
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

        result_nr = ['kde_network_normal_rule', N, 'F_1']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                   start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv = ['SP_network_ucv', N, 'F_1']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'F_1']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)

        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)

        result_ucv = ['kdenetwork_ucv', N, 'F_2']
        sample = trial_density_functions.sampling_F_2(N)
        holdout_likelihood = pb.HoldoutLikelihood(sample, seed=10)
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

        result_nr = ['kde_network_normal_rule', N, 'F_2']
        KDENetwork_nr = pb.KDENetwork(list(sample.columns))
        t_0 = time.time()
        KDENetwork_nr.fit(training_data)
        KDENetwork_nr = x.estimate(operators=pb.ArcOperatorSet(), score=holdout_likelihood,
                                   start=KDENetwork_nr, epsilon=0.01)
        KDENetwork_nr.fit(training_data)
        t_fin = time.time()
        result_nr.append(t_fin - t_0)

        result_SP_ucv = ['SP_network_ucv', N, 'F_2']
        SPBNetwork_ucv = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        SPBNetwork_ucv = x.estimate_wbm(operators=operators, score=holdout_likelihood,
                                        start=SPBNetwork_ucv, m_bselector=ucv, epsilon=0.01)
        SPBNetwork_ucv.fit_wbm(training_data, ucv)
        t_fin = time.time()
        result_SP_ucv.append(t_fin - t_0)

        result_SP_nr = ['SP_network_normal_rule', N, 'F_2']
        SPBNetwork_nr = pb.SemiparametricBN(list(sample.columns))
        t_0 = time.time()
        SPBNetwork_nr.fit(training_data)
        SPBNetwork_nr = x.estimate(operators=operators, score=holdout_likelihood,
                                   start=SPBNetwork_nr, epsilon=0.01)
        SPBNetwork_nr.fit(training_data)
        t_fin = time.time()
        result_SP_nr.append(t_fin - t_0)

        result_ucv.append(KDENetwork_ucv.slogl(test_data))
        result_nr.append(KDENetwork_nr.slogl(test_data))
        result_SP_ucv.append(SPBNetwork_ucv.slogl(test_data))
        result_SP_nr.append(SPBNetwork_nr.slogl(test_data))

        results = pd.concat([results, pd.DataFrame([result_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_nr], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_ucv], columns=results.columns)])
        results = pd.concat([results, pd.DataFrame([result_SP_nr], columns=results.columns)])
        print(results)




        





    results.to_csv(r"C:\Users\G513\PycharmProjects\pythonProject1\resultados\resultados.csv")