import pandas as pd
import pybnesian as pb
import time
import numpy as np
import math
import scipy
import normal_mixture_density_functions
# X=pd.DataFrame(np.random.multivariate_normal([0,2,5],np.random.rand(3,3),1000),columns=['x1','x2','x3'])

# X.to_csv('output.csv')


def sampling_gtn(seed, N = 100):
    np.random.seed(seed)
    u = np.random.uniform(size=N)
    sample = np.zeros((N, 5))
    for i in range(0, N):
        np.random.seed(seed + i + 105)
        sample[i, 0] = np.random.normal(0, 1)
        if u[i] <= 0.5:
            np.random.seed(seed + i + 105)
            sample[i, 1] = np.random.normal(-2, 2)
        else:
            np.random.seed(seed + i + 105)
            sample[i, 1] = np.random.normal(2, 2)
        np.random.seed(seed + i + 105)
        sample[i, 2] = sample[i, 0]*sample[i, 1] + np.random.normal(0, 1)
        sample[i, 3] = np.random.normal(10 + 0.8*sample[i, 2], 0.5)
        sample[i, 4] = 1/(1 + math.exp(-1 * sample[i, 3])) + np.random.normal(0, 0.5)
    return pd.DataFrame(sample, columns = ['A', 'B', 'C', 'D', 'E'])

def gtn(x):
    density = scipy.stats.norm().pdf(x[0])
    density *= 0.5 * scipy.stats.norm(-2, 2).pdf(x[1]) + 0.5 * scipy.stats.norm(2 , 2).pdf(x[1])
    density *=  scipy.stats.norm(x[0] * x[1] , 1).pdf(x[2])
    density *= scipy.stats.norm(10 + 0.8*x[2], 0.5).pdf(x[3])
    density *= scipy.stats.norm(1/(1 + math.exp(-1 * x[4])), 0.5).pdf(x[4]) 
    return math.log(density)


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

ground_truth_network = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'],
                                           arcs = [('A', 'C'), ('B', 'C'), ('C','D'), ('D', 'E')],
                                           node_types = [('A', pb.LinearGaussianCPDType()), ('B', pb.CKDEType()), ('C', pb.CKDEType()), ('D', pb.LinearGaussianCPDType()), ('E', pb.CKDEType())])
G_0 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'],
                            node_types = [('A', pb.LinearGaussianCPDType()), ('B', pb.LinearGaussianCPDType()), ('C', pb.LinearGaussianCPDType()), ('D', pb.LinearGaussianCPDType()), ('E', pb.LinearGaussianCPDType())])
G_1 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'],
                            node_types = [('A', pb.CKDEType()), ('B', pb.CKDEType()), ('C', pb.CKDEType()), ('D', pb.CKDEType()), ('E', pb.CKDEType())])

results =  pd.DataFrame(columns = ['initial network', 'learning method', 'log_likelihood', 'HMD', 'SHD', 'THMD'])
node_map = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
for k in range(0, 1):
    sample = sampling_gtn(k, 10000)
    validation_sample = sampling_gtn(k + 1000, 1000)
    validation_likelihood = ['ground network', 'none', validation_sample.apply(gtn, axis = 1).sum(), 0, 0, 0]
    results = pd.concat([results, pd.DataFrame([validation_likelihood], columns=results.columns)])
    print(results)
    operators=pb.OperatorPool([pb.ArcOperatorSet(),pb.ChangeNodeTypeSet()])
    
    result_SP_scv_G_0= ['G_0', 'hc_scv' ]
    SPBNetwork_scv_G_0 = pb.SemiparametricBN(['A', 'B', 'C', 'D', 'E'])
    x = pb.GreedyHillClimbing()
    SPBNetwork_scv_G_0 = x.estimate_wbm(operators=operators, score=pb.ValidatedLikelihood(sample, seed=10, k = 5),
                                    start=G_0, m_bselector=pb.SCV(), epsilon=0.01)
    SPBNetwork_scv_G_0.fit_wbm(sample, pb.SCV())
    slog=SPBNetwork_scv_G_0.slogl(validation_sample)
    result_SP_scv_G_0.append(slog)
    result_SP_scv_G_0.append(hamming_distance(SPBNetwork_scv_G_0.arcs(), ground_truth_network.arcs(), node_map))
    result_SP_scv_G_0.append(structural_hamming_distance(SPBNetwork_scv_G_0.arcs(), ground_truth_network.arcs()))
    result_SP_scv_G_0.append(node_type_hamming_distance(list(SPBNetwork_scv_G_0.node_types().values()), list(ground_truth_network.node_types().values())))



    results = pd.concat([results, pd.DataFrame([result_SP_scv_G_0], columns=results.columns)])


    print(results)
    results.to_csv('results_exp_1_scv_G0_10000_0.csv')