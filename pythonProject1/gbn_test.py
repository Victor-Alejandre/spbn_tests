import pandas as pd
import pybnesian as pb
import time
import numpy as np
import math
import scipy

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
    num_nodes (int): Number of nodes in the graphs.

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

ecoli_arcs = pd.read_csv('gbn_datasets/ecoli_arcs.csv')
ecoli_df = pd.read_csv('gbn_datasets/ecoli70_200_data.csv')
ecoli_df = ecoli_df.drop(columns=['Unnamed: 0'])
validation_ecoli = pd.read_csv('gbn_datasets/ecoli70_validation_data.csv')

def bnlearn_gbn_proccesing(data, arc_data):
    gbn = pb.GaussianNetwork(nodes = list(data.columns), arcs = list(arc_data.itertuples(index=False, name=None)))
    return gbn

ecoli_70_gbn = bnlearn_gbn_proccesing(ecoli_df, ecoli_arcs)
results =  pd.DataFrame(columns = ['initial network', 'learning method', 'log_likelihood', 'HMD', 'SHD', 'THMD'])
operators=pb.OperatorPool([pb.ArcOperatorSet(),pb.ChangeNodeTypeSet()])
node_map = {}
for i in range(len(ecoli_df.columns)):
    node_map[list(ecoli_df.columns)[i]] = i
#Initialization networks
ecoli_gbn = pb.GaussianNetwork(nodes = list(ecoli_df.columns))
G_0 = pb.SemiparametricBN(list(ecoli_df.columns),
                            node_types = [(variable, pb.LinearGaussianCPDType()) for variable in list(ecoli_df.columns)])
G_1 = pb.SemiparametricBN(list(ecoli_df.columns),
                            node_types = [(variable, pb.CKDEType()) for variable in list(ecoli_df.columns)])


result_gbn_G_0= ['ecoli_gbn', 'hc_gbn_bic' ]
GBNetwork_G_0 = pb.GaussianNetwork(nodes = list(ecoli_df.columns))
x = pb.GreedyHillClimbing()
GBNetwork_G_0 = x.estimate(operators=pb.ArcOperatorSet(), score=pb.BIC(ecoli_df),
                                start=ecoli_gbn, epsilon=0.01)
GBNetwork_G_0.fit(ecoli_df)
slog=GBNetwork_G_0.slogl(validation_ecoli)
result_gbn_G_0.append(slog)
result_gbn_G_0.append(hamming_distance(GBNetwork_G_0.arcs(), ecoli_70_gbn.arcs(), node_map))
result_gbn_G_0.append(structural_hamming_distance(GBNetwork_G_0.arcs(), ecoli_70_gbn.arcs()))
result_gbn_G_0.append(node_type_hamming_distance(list(GBNetwork_G_0.node_types().values()), list(ecoli_70_gbn.node_types().values())))

result_gbn_G_1= ['ecoli_gbn', 'hc_gbn_bde' ]
GBNetwork_G_1 = pb.GaussianNetwork(nodes = list(ecoli_df.columns))
x = pb.GreedyHillClimbing()
GBNetwork_G_1 = x.estimate(operators=pb.ArcOperatorSet(), score=pb.BGe(ecoli_df),
                                start=ecoli_gbn, epsilon=0.01)
GBNetwork_G_1.fit(ecoli_df)
slog=GBNetwork_G_1.slogl(validation_ecoli)
result_gbn_G_1.append(slog)
result_gbn_G_1.append(hamming_distance(GBNetwork_G_1.arcs(), ecoli_70_gbn.arcs(), node_map))
result_gbn_G_1.append(structural_hamming_distance(GBNetwork_G_1.arcs(), ecoli_70_gbn.arcs()))
result_gbn_G_1.append(node_type_hamming_distance(list(GBNetwork_G_1.node_types().values()), list(ecoli_70_gbn.node_types().values())))

result_SP_nr_G_0= ['G_0', 'hc_nr' ]
SPBNetwork_nr_G_0 = pb.SemiparametricBN(list(ecoli_df.columns))
x = pb.GreedyHillClimbing()
SPBNetwork_nr_G_0 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(ecoli_df, seed=10),
                                start=G_0, epsilon=0.01)
SPBNetwork_nr_G_0.fit(ecoli_df)
slog=SPBNetwork_nr_G_0.slogl(validation_ecoli)
result_SP_nr_G_0.append(slog)
result_SP_nr_G_0.append(hamming_distance(SPBNetwork_nr_G_0.arcs(), ecoli_70_gbn.arcs(), node_map))
result_SP_nr_G_0.append(structural_hamming_distance(SPBNetwork_nr_G_0.arcs(), ecoli_70_gbn.arcs()))
result_SP_nr_G_0.append(node_type_hamming_distance(list(SPBNetwork_nr_G_0.node_types().values()), list(ecoli_70_gbn.node_types().values())))

result_SP_nr_G_1= ['G_1', 'hc_nr' ]
SPBNetwork_nr_G_1 = pb.SemiparametricBN(list(ecoli_df.columns))
x = pb.GreedyHillClimbing()
SPBNetwork_nr_G_1 = x.estimate(operators=operators, score=pb.HoldoutLikelihood(ecoli_df, seed=10),
                                start=G_1, epsilon=0.01)
SPBNetwork_nr_G_1.fit(ecoli_df)
slog = SPBNetwork_nr_G_1.slogl(validation_ecoli)
result_SP_nr_G_1.append(slog)
result_SP_nr_G_1.append(hamming_distance(SPBNetwork_nr_G_1.arcs(), ecoli_70_gbn.arcs(), node_map))
result_SP_nr_G_1.append(structural_hamming_distance(SPBNetwork_nr_G_1.arcs(), ecoli_70_gbn.arcs()))
result_SP_nr_G_1.append(node_type_hamming_distance(list(SPBNetwork_nr_G_1.node_types().values()), list(ecoli_70_gbn.node_types().values())))


result_SP_ucv_G_0= ['G_0', 'hc_ucv' ]
SPBNetwork_ucv_G_0 = pb.SemiparametricBN(list(ecoli_df.columns))
x = pb.GreedyHillClimbing()
SPBNetwork_ucv_G_0 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(ecoli_df, seed=10),
                                start=G_0, m_bselector=pb.UCV(), epsilon=0.01)
SPBNetwork_ucv_G_0.fit_wbm(ecoli_df, pb.UCV())
slog=SPBNetwork_ucv_G_0.slogl(validation_ecoli)
result_SP_ucv_G_0.append(slog)
result_SP_ucv_G_0.append(hamming_distance(SPBNetwork_ucv_G_0.arcs(), ecoli_70_gbn.arcs(), node_map))
result_SP_ucv_G_0.append(structural_hamming_distance(SPBNetwork_ucv_G_0.arcs(), ecoli_70_gbn.arcs()))
result_SP_ucv_G_0.append(node_type_hamming_distance(list(SPBNetwork_ucv_G_0.node_types().values()), list(ecoli_70_gbn.node_types().values())))

result_SP_ucv_G_1= ['G_1', 'hc_ucv' ]
SPBNetwork_ucv_G_1 = pb.SemiparametricBN(list(ecoli_df.columns))
x = pb.GreedyHillClimbing()
SPBNetwork_ucv_G_1 = x.estimate_wbm(operators=operators, score=pb.HoldoutLikelihood(ecoli_df, seed=10),
                                start=G_1, m_bselector=pb.UCV(), epsilon=0.01)
SPBNetwork_ucv_G_1.fit_wbm(ecoli_df, pb.UCV())
slog=SPBNetwork_ucv_G_1.slogl(validation_ecoli)
result_SP_ucv_G_1.append(slog)
result_SP_ucv_G_1.append(hamming_distance(SPBNetwork_ucv_G_1.arcs(), ecoli_70_gbn.arcs(), node_map))
result_SP_ucv_G_1.append(structural_hamming_distance(SPBNetwork_ucv_G_1.arcs(), ecoli_70_gbn.arcs()))
result_SP_ucv_G_1.append(node_type_hamming_distance(list(SPBNetwork_ucv_G_1.node_types().values()), list(ecoli_70_gbn.node_types().values())))

results = pd.concat([results, pd.DataFrame([result_SP_ucv_G_0], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_ucv_G_1], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_nr_G_0], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_SP_nr_G_1], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_gbn_G_0], columns=results.columns)])
results = pd.concat([results, pd.DataFrame([result_gbn_G_1], columns=results.columns)])


print(results)
results.to_csv('results_gbn_ecoli_test.csv')