import pandas as pd
import pybnesian as pb
import time
import numpy as np
import math
import scipy
import normal_mixture_density_functions
# X=pd.DataFrame(np.random.multivariate_normal([0,2,5],np.random.rand(3,3),1000),columns=['x1','x2','x3'])

# X.to_csv('output.csv')


def sampling_gtn(N = 100):
    u = np.random.uniform(size=N)
    sample = np.zeros((N, 5))
    for i in range(0, N):
        sample[i, 0] = np.random.normal(0, 1)
        if u[i] <= 0.5:
            sample[i, 1] = np.random.normal(-2, 2)
        else:
            sample[i, 1] = np.random.normal(2, 2)
        sample[i, 2] = sample[i, 0]*sample[i, 1] + np.random.normal(0, 1)
        sample[i, 3] = np.random.normal(10 + 0.8*sample[i, 2], 0.5)
        sample[i, 4] = 1/(1 + math.exp(-1 * sample[i, 3])) + np.random.normal(0, 0.5)
    return pd.DataFrame(sample, columns = ['A', 'B', 'C', 'D', 'E'])

def gtn(x):
    density = scipy.stats.norm().pdf(x[0])
    density *= 0.5 * scipy.stats.norm().pdf(x[1], -2, 2) + 0.5 * scipy.stats.norm().pdf(x[1], 2 , 2)
    density *= x[0] * x[1] + scipy.stats.norm().pdf(x[2])
    density *= scipy.stats.norm().pdf(x[3], 10 + 0.8*x[2], 0.5)
    density *= scipy.stats.norm().pdf(x[4], 0, 0.5) + 1/(1 + math.exp(-1 * x[4]))
    return density



def arcs_to_adjacency_matrix(arcs, node_map):
    """
    Convert a list of arcs to an adjacency matrix.

    Parameters:
    arcs (list of tuples): List of arcs (edges) in the graph, represented as tuples of letters.
    node_map (dict): Mapping from letters to integer node identifiers.

    Returns:
    numpy.ndarray: Adjacency matrix of the graph.
    """
    num_nodes = len(node_map)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for arc in arcs:
        # Map letters to integer node identifiers
        i, j = node_map[arc[0]], node_map[arc[1]]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # For undirected graphs

    return adj_matrix

def hamming_distance(arcs1, arcs2, node_map):
    """
    Compute the Hamming distance between two graphs represented as lists of arcs.

    Parameters:
    arcs1 (list of tuples): List of arcs (edges) in the first graph.
    arcs2 (list of tuples): List of arcs (edges) in the second graph.
    num_nodes (dict): Number of nodes in the graphs.

    Returns:
    int: The Hamming distance between the two graphs.
    """
    # Convert arcs to adjacency matrices
    graph1 = arcs_to_adjacency_matrix(arcs1, node_map)
    graph2 = arcs_to_adjacency_matrix(arcs2, node_map)

    # Compute the Hamming distance between the adjacency matrices
    hamming_dist = np.sum(np.abs(graph1 - graph2))

    return hamming_dist

def structural_hamming_distance(arcs1, arcs2):
    """
    Compute the structural Hamming distance between two directed acyclic graphs (DAGs)
    represented as lists of arcs.

    Parameters:
    arcs1 (list of tuples): List of arcs (edges) in the first graph.
    arcs2 (list of tuples): List of arcs (edges) in the second graph.

    Returns:
    int: The structural Hamming distance between the two graphs.
    """
    # Convert lists of arcs to sets for efficient comparison
    arcs_set1 = set(arcs1)
    arcs_set2 = set(arcs2)

    # Compute the structural Hamming distance
    additions = len(arcs_set2 - arcs_set1)
    removals = len(arcs_set1 - arcs_set2)

    # Compute the number of arcs that need to be reversed
    reverse_count = 0
    for arc in arcs_set1:
        if arc not in arcs_set2 and (arc[1], arc[0]) in arcs_set2:
            reverse_count += 1

    # Total structural Hamming distance
    hamming_dist = additions + removals + reverse_count

    return hamming_dist

def node_type_hamming_distance(node_types1, node_types2):
    distance = 0
    for i in range(0, len(node_types1)):
        if node_types1[i] != node_types2[i]:
            distance += 1

    return distance

ground_truth_network = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'],
                                           arcs = [('A', 'C'), ('B', 'C'), ('C','D'), ('D', 'E')],
                                           node_types = [('A', pb.LinearGaussianCPDType()), ('B', pb.CKDEType()), ('C', pb.CKDEType()), ('D', pb.LinearGaussianCPDType()), ('E', pb.CKDEType())])
G_0 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'],
                            node_types = [('A', pb.LinearGaussianCPDType()), ('B', pb.LinearGaussianCPDType()), ('C', pb.LinearGaussianCPDType()), ('D', pb.LinearGaussianCPDType()), ('E', pb.LinearGaussianCPDType())])
G_1 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'],
                            node_types = [('A', pb.CKDEType()), ('B', pb.CKDEType()), ('C', pb.CKDEType()), ('D', pb.CKDEType()), ('E', pb.CKDEType())])

sample = sampling_gtn(2000)
validation_sample = sampling_gtn(1000)
operators=pb.OperatorPool([pb.ArcOperatorSet(),pb.ChangeNodeTypeSet()])
results =  pd.DataFrame(columns = ['initial network', 'learning method', 'log_likelihood', 'HMD', 'SHD', 'THMD'])
node_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}



result_SP_pc_RCoT_nr= ['RCoT', 'pc_nr' ]
SPBNetwork_pc_RCoT = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_pc_RCoT = x.estimate(pb.RCoT(sample))
SPBNetwork_pc_RCoT = pb.SemiparametricBN(SPBNetwork_pc_RCoT.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_pc_RCoT = x.estimate(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                 start=SPBNetwork_pc_RCoT, epsilon=0.01)
SPBNetwork_pc_RCoT.fit(sample)
slog=SPBNetwork_pc_RCoT.slogl(validation_sample)
result_SP_pc_RCoT_nr.append(slog)
result_SP_pc_RCoT_nr.append(hamming_distance(SPBNetwork_pc_RCoT.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_RCoT_nr.append(structural_hamming_distance(SPBNetwork_pc_RCoT.arcs(), ground_truth_network.arcs()))
result_SP_pc_RCoT_nr.append(node_type_hamming_distance(list(SPBNetwork_pc_RCoT.node_types().values()), list(ground_truth_network.node_types().values()))
)
print(result_SP_pc_RCoT_nr)

result_SP_pc_plc_nr= ['plc', 'pc__nr' ]
SPBNetwork_pc_plc = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_pc_plc = x.estimate(pb.LinearCorrelation(sample))
SPBNetwork_pc_plc = pb.SemiparametricBN(SPBNetwork_pc_plc.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_pc_plc = x.estimate(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                    start=SPBNetwork_pc_plc, epsilon=0.01)
SPBNetwork_pc_plc.fit(sample)
slog=SPBNetwork_pc_plc.slogl(validation_sample)
result_SP_pc_plc_nr.append(slog)
result_SP_pc_plc_nr.append(hamming_distance(SPBNetwork_pc_plc.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_plc_nr.append(structural_hamming_distance(SPBNetwork_pc_plc.arcs(), ground_truth_network.arcs()))
result_SP_pc_plc_nr.append(node_type_hamming_distance(list(SPBNetwork_pc_plc.node_types().values()), list(ground_truth_network.node_types().values())))
print(result_SP_pc_plc_nr)


result_SP_pc_RCoT_ucv= ['RCoT', 'pc_ucv' ]
SPBNetwork_ucv_G_0 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_ucv_G_0 = x.estimate(pb.RCoT(sample))
SPBNetwork_ucv_G_0 = pb.SemiparametricBN(SPBNetwork_ucv_G_0.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_ucv_G_0 = x.estimate_wbm(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                start=SPBNetwork_ucv_G_0, m_bselector=pb.UCV(), epsilon=0.01)
SPBNetwork_ucv_G_0.fit_wbm(sample, pb.UCV())
slog=SPBNetwork_ucv_G_0.slogl(validation_sample)
result_SP_pc_RCoT_ucv.append(slog)
result_SP_pc_RCoT_ucv.append(hamming_distance(SPBNetwork_ucv_G_0.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_RCoT_ucv.append(structural_hamming_distance(SPBNetwork_ucv_G_0.arcs(), ground_truth_network.arcs()))
result_SP_pc_RCoT_ucv.append(node_type_hamming_distance(list(SPBNetwork_ucv_G_0.node_types().values()), list(ground_truth_network.node_types().values())))
print(result_SP_pc_RCoT_ucv)

result_SP_pc_plc_ucv= ['plc', 'pc_ucv' ]
SPBNetwork_ucv_G_1 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_ucv_G_1 = x.estimate(pb.LinearCorrelation(sample))
SPBNetwork_ucv_G_1 = pb.SemiparametricBN(SPBNetwork_ucv_G_1.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_ucv_G_1 = x.estimate_wbm(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                    start=SPBNetwork_ucv_G_1, m_bselector=pb.UCV(), epsilon=0.01)
SPBNetwork_ucv_G_1.fit_wbm(sample, pb.UCV())
slog=SPBNetwork_ucv_G_1.slogl(validation_sample)
result_SP_pc_plc_ucv.append(slog)
result_SP_pc_plc_ucv.append(hamming_distance(SPBNetwork_ucv_G_1.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_plc_ucv.append(structural_hamming_distance(SPBNetwork_ucv_G_1.arcs(), ground_truth_network.arcs()))
result_SP_pc_plc_ucv.append(node_type_hamming_distance(list(SPBNetwork_ucv_G_1.node_types().values()), list(ground_truth_network.node_types().values())))
print(result_SP_pc_plc_ucv)


result_SP_pc_RCoT_PI= ['RCoT', 'pc_PI' ]
SPBNetwork_PI_G_0 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_PI_G_0 = x.estimate(pb.RCoT(sample))
SPBNetwork_PI_G_0 = pb.SemiparametricBN(SPBNetwork_PI_G_0.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_PI_G_0 = x.estimate_wbm(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                start=SPBNetwork_PI_G_0, m_bselector=pb.PI(), epsilon=0.01)
SPBNetwork_PI_G_0.fit_wbm(sample, pb.PI())
slog=SPBNetwork_PI_G_0.slogl(validation_sample)
result_SP_pc_RCoT_PI.append(slog)
result_SP_pc_RCoT_PI.append(hamming_distance(SPBNetwork_PI_G_0.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_RCoT_PI.append(structural_hamming_distance(SPBNetwork_PI_G_0.arcs(), ground_truth_network.arcs()))
result_SP_pc_RCoT_PI.append(node_type_hamming_distance(list(SPBNetwork_PI_G_0.node_types().values()), list(ground_truth_network.node_types().values())))
print(result_SP_pc_RCoT_PI)

result_SP_pc_plc_PI= ['plc', 'pc_PI' ]
SPBNetwork_PI_G_1 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_PI_G_1 = x.estimate(pb.LinearCorrelation(sample))
SPBNetwork_PI_G_1 = pb.SemiparametricBN(SPBNetwork_PI_G_1.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_PI_G_1 = x.estimate_wbm(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                    start=SPBNetwork_PI_G_1, m_bselector=pb.PI(), epsilon=0.01)
SPBNetwork_PI_G_1.fit_wbm(sample, pb.PI())
slog=SPBNetwork_PI_G_1.slogl(validation_sample)
result_SP_pc_plc_PI.append(slog)
result_SP_pc_plc_PI.append(hamming_distance(SPBNetwork_PI_G_1.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_plc_PI.append(structural_hamming_distance(SPBNetwork_PI_G_1.arcs(), ground_truth_network.arcs()))
result_SP_pc_plc_PI.append(node_type_hamming_distance(list(SPBNetwork_PI_G_1.node_types().values()), list(ground_truth_network.node_types().values())))
print(result_SP_pc_plc_PI)

'''
result_SP_pc_RCoT_scv= ['RCoT', 'pc_scv' ]
SPBNetwork_scv_G_0 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_scv_G_0 = x.estimate(pb.RCoT(sample))
SPBNetwork_scv_G_0 = pb.SemiparametricBN(SPBNetwork_scv_G_0.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_scv_G_0 = x.estimate_wbm(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                start=SPBNetwork_scv_G_0, m_bselector=pb.SCV(), epsilon=0.01)
SPBNetwork_scv_G_0.fit_wbm(sample, pb.SCV())
slog=SPBNetwork_scv_G_0.slogl(validation_sample)
result_SP_pc_RCoT_scv.append(slog)
result_SP_pc_RCoT_scv.append(hamming_distance(SPBNetwork_scv_G_0.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_RCoT_scv.append(structural_hamming_distance(SPBNetwork_scv_G_0.arcs(), ground_truth_network.arcs()))
result_SP_pc_RCoT_scv.append(node_type_hamming_distance(list(SPBNetwork_scv_G_0.node_types().values()), list(ground_truth_network.node_types().values())))
print(result_SP_pc_RCoT_scv)

result_SP_pc_plc_scv= ['plc', 'pc_scv' ]
SPBNetwork_scv_G_1 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
x = pb.PC()
SPBNetwork_scv_G_1 = x.estimate(pb.LinearCorrelation(sample))
SPBNetwork_scv_G_1 = pb.SemiparametricBN(SPBNetwork_scv_G_1.to_dag())
x = pb.GreedyHillClimbing()
SPBNetwork_scv_G_1 = x.estimate_wbm(operators=pb.ChangeNodeTypeSet(), score=pb.ValidatedLikelihood(sample, seed=10),
                                    start=SPBNetwork_scv_G_1, m_bselector=pb.SCV(), epsilon=0.01)
SPBNetwork_scv_G_1.fit_wbm(sample, pb.SCV())
slog=SPBNetwork_scv_G_1.slogl(validation_sample)
result_SP_pc_plc_scv.append(slog)
result_SP_pc_plc_scv.append(hamming_distance(SPBNetwork_scv_G_1.arcs(), ground_truth_network.arcs(), node_map))
result_SP_pc_plc_scv.append(structural_hamming_distance(SPBNetwork_scv_G_1.arcs(), ground_truth_network.arcs()))
result_SP_pc_plc_scv.append(node_type_hamming_distance(list(SPBNetwork_scv_G_1.node_types().values()), list(ground_truth_network.node_types().values())))
print(result_SP_pc_plc_scv)
'''
results = pd.concat([results, pd.DataFrame([result_SP_pc_RCoT_nr], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_pc_plc_nr], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_pc_RCoT_ucv], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_pc_plc_ucv], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_pc_RCoT_PI], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_pc_plc_PI], columns=results.columns)])
'''
results = pd.concat([results, pd.DataFrame([result_SP_pc_RCoT_scv], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_pc_plc_scv], columns=results.columns)])
'''


print(results)
results.to_csv('results_exp_1_pc_2000.csv')
