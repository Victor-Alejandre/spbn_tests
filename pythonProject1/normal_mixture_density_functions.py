import scipy.stats
import math
import numpy as np
import pandas as pd



def kurtotic(x):
    covariance_matrix_1 = np.array([[1,1],[1,4]])
    covariance_matrix_2=np.array([[4,-1/3],[-1/3,4/9]])
    means=np.zeros((2,2))
    covariance_matrices=np.zeros((2,2,2))
    covariance_matrices[0]=covariance_matrix_1
    covariance_matrices[1]=covariance_matrix_2
    weights=np.array([2/3,1/3])
    pdf=weights[0]*scipy.stats.multivariate_normal.pdf(x,means[0], covariance_matrices[0])+weights[1]*scipy.stats.multivariate_normal.pdf(x, means[1], covariance_matrices[1])
    return pdf

def kurtotic_parameters():
    covariance_matrix_1 = np.array([[1, 1], [1, 4]])
    covariance_matrix_2 = np.array([[4, -1 / 3], [-1 / 3, 4 / 9]])
    means = np.zeros((2, 2))
    covariance_matrices = np.zeros((2, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    weights = np.array([2 / 3, 1 / 3])
    return weights, means, covariance_matrices

def bimodal(x):
    covariance_matrix_1 = 4/9*np.eye(2)
    covariance_matrix_2=4/9*np.eye(2)
    means=np.zeros((2,2))
    means[0]=np.array([1,0])
    means[1]=np.array([-1,0])
    covariance_matrices=np.zeros((2,2,2))
    covariance_matrices[0]=covariance_matrix_1
    covariance_matrices[1]=covariance_matrix_2
    weights=np.array([0.5,0.5])

    pdf = 0.5 * scipy.stats.multivariate_normal.pdf(x, means[0],
                                                    covariance_matrices[0]) + 0.5 * scipy.stats.multivariate_normal.pdf(
        x, means[1], covariance_matrices[1])
    return pdf

def bimodal_parameters():
    covariance_matrix_1 = 4 / 9 * np.eye(2)
    covariance_matrix_2 = 4 / 9 * np.eye(2)
    means = np.zeros((2, 2))
    means[0] = np.array([1, 0])
    means[1] = np.array([-1, 0])
    covariance_matrices = np.zeros((2, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    weights = np.array([0.5, 0.5])
    return weights, means, covariance_matrices


def separated_bimodal(x):
    covariance_matrix_1 = np.array([[4/9,4/15],[4/15,4/9]])
    covariance_matrix_2 =np.array([[4/9,4/15],[4/15,4/9]])
    means = np.zeros((2, 2))
    means[0] = np.array([-2, 2])
    means[1] = np.array([2, -2])
    covariance_matrices = np.zeros((2, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    weights = np.array([0.5, 0.5])

    pdf = 0.5 * scipy.stats.multivariate_normal.pdf(x, means[0],
                                                    covariance_matrices[0]) + 0.5 * scipy.stats.multivariate_normal.pdf(
        x, means[1], covariance_matrices[1])
    return pdf


def separated_bimodal_parameters():
    covariance_matrix_1 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_2 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    means = np.zeros((2, 2))
    means[0] = np.array([-2, 2])
    means[1] = np.array([2, -2])
    covariance_matrices = np.zeros((2, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    weights = np.array([0.5, 0.5])
    return weights, means, covariance_matrices


def asymmetric_bimodal(x):
    covariance_matrix_1 = np.array([[4/9,4/15],[4/15,4/9]])
    covariance_matrix_2 = 4/9*np.eye(2)
    means = np.zeros((2, 2))
    means[0] = np.array([-1, 1])
    means[1] = np.array([1, -1])
    covariance_matrices = np.zeros((2, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    weights = np.array([0.5, 0.5])

    pdf = 0.5 * scipy.stats.multivariate_normal.pdf(x, means[0],
                                                    covariance_matrices[0]) + 0.5 * scipy.stats.multivariate_normal.pdf(
        x, means[1], covariance_matrices[1])
    return pdf

def asymmetric_bimodal_parameters():
    covariance_matrix_1 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_2 = 4 / 9 * np.eye(2)
    means = np.zeros((2, 2))
    means[0] = np.array([-1, 1])
    means[1] = np.array([1, -1])
    covariance_matrices = np.zeros((2, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    weights = np.array([0.5, 0.5])
    return weights, means, covariance_matrices

def trimodal(x):
    covariance_matrix_1 = 1 / 25 * np.array([[9,63/10],[63/10,49/4]])
    covariance_matrix_2 = 1 / 25 * np.array([[9,0],[0,49/4]])
    covariance_matrix_3=  1 / 25 * np.array([[9,0],[0,49/4]])
    means = np.zeros((3, 2))
    means[0] = np.array([-1, 0])
    means[1] = np.array([1,2/math.sqrt(3)])
    means[2] = np.array([1,-2/math.sqrt(3)])
    covariance_matrices = np.zeros((3, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    covariance_matrices[2] = covariance_matrix_3
    weights = np.array([3/7, 3/7, 1/7])

    pdf = weights[0] * scipy.stats.multivariate_normal.pdf(x, means[0],
                                                    covariance_matrices[0]) + weights[1] * scipy.stats.multivariate_normal.pdf(
        x, means[1], covariance_matrices[1]) + weights[2] * scipy.stats.multivariate_normal.pdf(x, means[2],
                                                    covariance_matrices[2])
    return pdf

def trimodal_parameters():
    covariance_matrix_1 = 1 / 25 * np.array([[9, 63 / 10], [63 / 10, 49 / 4]])
    covariance_matrix_2 = 1 / 25 * np.array([[9, 0], [0, 49 / 4]])
    covariance_matrix_3 = 1 / 25 * np.array([[9, 0], [0, 49 / 4]])
    means = np.zeros((3, 2))
    means[0] = np.array([-1, 0])
    means[1] = np.array([1, 2 / math.sqrt(3)])
    means[2] = np.array([1, -2 / math.sqrt(3)])
    covariance_matrices = np.zeros((3, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    covariance_matrices[2] = covariance_matrix_3
    weights = np.array([3 / 7, 3 / 7, 1 / 7])
    return weights, means, covariance_matrices


def double_fountain(x):
    covariance_matrix_1 =  np.array([[4/9,4/15],[4/15,4/9]])
    covariance_matrix_2 = np.array([[4/9,4/15],[4/15,4/9]])
    covariance_matrix_3=  1 / 15 * np.array([[1/15,1/25],[1/25,1/15]])
    covariance_matrix_4 =  1 / 15 * np.array([[1/15,1/25],[1/25,1/15]])
    covariance_matrix_5 =  1 / 15 * np.array([[1/15,1/25],[1/25,1/15]])
    covariance_matrix_6 = 1 / 9 * np.array([[1, 3/5], [3/5, 1]])
    covariance_matrix_7 = 1 / 15 * np.array([[1/15,1/25],[1/25,1/15]])
    covariance_matrix_8 = 1 / 15 * np.array([[1/15,1/25],[1/25,1/15]])
    covariance_matrix_9 = 1 / 15 * np.array([[1/15,1/25],[1/25,1/15]])
    means = np.zeros((9, 2))
    means[0] = np.array([-3/2, 0])
    means[1] = np.array([3/2, 0])
    means[2] = np.array([-1-3/2,-1])
    means[3] = np.array([-3 / 2, 0])
    means[4] = np.array([1 - 3 / 2, 1])
    means[5] = np.zeros(2)
    means[6] = np.array([-1 + 3 / 2, -1])
    means[7] = np.array([ 3 / 2, 0])
    means[8] = np.array([1 + 3 / 2, 1])
    covariance_matrices = np.zeros((9, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    covariance_matrices[2] = covariance_matrix_3
    covariance_matrices[3] = covariance_matrix_4
    covariance_matrices[4] = covariance_matrix_5
    covariance_matrices[5] = covariance_matrix_6
    covariance_matrices[6] = covariance_matrix_7
    covariance_matrices[7] = covariance_matrix_8
    covariance_matrices[8] = covariance_matrix_9
    weights = np.array([12/25, 12/25, 1/350,1/350,1/350,8/350,1/350,1/350,1/350])

    pdf = weights[0] * scipy.stats.multivariate_normal.pdf(x, means[0],
                                                    covariance_matrices[0]) + weights[1] * scipy.stats.multivariate_normal.pdf(
        x, means[1], covariance_matrices[1]) + weights[2] * scipy.stats.multivariate_normal.pdf(x, means[2],
                                                    covariance_matrices[2])
    return pdf

def double_fountain_parameters():
    covariance_matrix_1 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_2 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_3 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_4 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_5 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_6 = 1 / 9 * np.array([[1, 3 / 5], [3 / 5, 1]])
    covariance_matrix_7 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_8 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_9 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    means = np.zeros((9, 2))
    means[0] = np.array([-3 / 2, 0])
    means[1] = np.array([3 / 2, 0])
    means[2] = np.array([-1 - 3 / 2, -1])
    means[3] = np.array([-3 / 2, 0])
    means[4] = np.array([1 - 3 / 2, 1])
    means[5] = np.zeros(2)
    means[6] = np.array([-1 + 3 / 2, -1])
    means[7] = np.array([3 / 2, 0])
    means[8] = np.array([1 + 3 / 2, 1])
    covariance_matrices = np.zeros((9, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    covariance_matrices[2] = covariance_matrix_3
    covariance_matrices[3] = covariance_matrix_4
    covariance_matrices[4] = covariance_matrix_5
    covariance_matrices[5] = covariance_matrix_6
    covariance_matrices[6] = covariance_matrix_7
    covariance_matrices[7] = covariance_matrix_8
    covariance_matrices[8] = covariance_matrix_9
    weights = np.array([12 / 25, 12 / 25, 1 / 350, 1 / 350, 1 / 350, 8 / 350, 1 / 350, 1 / 350, 1 / 350])
    return weights, means, covariance_matrices






def sampling_kurtotic(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = np.array([[1, 1], [1, 4]])
    covariance_matrix_2 = np.array([[4 , -1 / 3], [-1 / 3, 4 / 9]])
    for i in range(N):
        if u[i]<=2/3:
            sample[i]=np.random.multivariate_normal(np.array([0,0]),covariance_matrix_1)
        else:
            sample[i] = np.random.multivariate_normal(np.array([0,0]),covariance_matrix_2)

    return pd.DataFrame(sample,columns=['x1','x2'])

def sampling_bimodal(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = 4/9*np.eye(2)
    covariance_matrix_2=4/9*np.eye(2)
    means = np.zeros((2, 2))
    means[0] = np.array([1, 0])
    means[1] = np.array([-1, 0])
    for i in range(N):
        if u[i]<=0.5:
            sample[i]=np.random.multivariate_normal(means[0],covariance_matrix_1)
        else:
            sample[i] = np.random.multivariate_normal(means[1],covariance_matrix_2)

    return pd.DataFrame(sample,columns=['x1','x2'])


def sampling_separated_bimodal(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_2 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    means = np.zeros((2, 2))
    means[0] = np.array([-2, 2])
    means[1] = np.array([2, -2])
    for i in range(N):
        if u[i]<=0.5:
            sample[i]=np.random.multivariate_normal(means[0],covariance_matrix_1)
        else:
            sample[i] = np.random.multivariate_normal(means[1],covariance_matrix_2)

    return pd.DataFrame(sample,columns=['x1','x2'])


def sampling_asymmetric_bimodal(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_2 = 4 / 9 * np.eye(2)
    means = np.zeros((2, 2))
    means[0] = np.array([-1, 1])
    means[1] = np.array([1, -1])
    for i in range(N):
        if u[i]<=0.5:
            sample[i]=np.random.multivariate_normal(means[0],covariance_matrix_1)
        else:
            sample[i] = np.random.multivariate_normal(means[1],covariance_matrix_2)

    return pd.DataFrame(sample,columns=['x1','x2'])


def sampling_trimodal(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = 1 / 25 * np.array([[9, 63 / 10], [63 / 10, 49 / 4]])
    covariance_matrix_2 = 1 / 25 * np.array([[9, 0], [0, 49 / 4]])
    covariance_matrix_3 = 1 / 25 * np.array([[9, 0], [0, 49 / 4]])
    means = np.zeros((3, 2))
    means[0] = np.array([-1, 0])
    means[1] = np.array([1, 2 / math.sqrt(3)])
    means[2] = np.array([1, -2 / math.sqrt(3)])
    for i in range(N):
        if u[i]<=3/7:
            sample[i]=np.random.multivariate_normal(means[0],covariance_matrix_1)
        elif 3/7<u[i]<=6/7:
            sample[i] = np.random.multivariate_normal(means[1], covariance_matrix_2)
        else:
            sample[i] = np.random.multivariate_normal(means[2],covariance_matrix_3)

    return pd.DataFrame(sample,columns=['x1','x2'])


def sampling_double_fountain(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_2 = np.array([[4 / 9, 4 / 15], [4 / 15, 4 / 9]])
    covariance_matrix_3 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_4 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_5 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_6 = 1 / 9 * np.array([[1, 3 / 5], [3 / 5, 1]])
    covariance_matrix_7 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_8 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    covariance_matrix_9 = 1 / 15 * np.array([[1 / 15, 1 / 25], [1 / 25, 1 / 15]])
    means = np.zeros((9, 2))
    means[0] = np.array([-3 / 2, 0])
    means[1] = np.array([3 / 2, 0])
    means[2] = np.array([-1 - 3 / 2, -1])
    means[3] = np.array([-3 / 2, 0])
    means[4] = np.array([1 - 3 / 2, 1])
    means[5] = np.zeros(2)
    means[6] = np.array([-1 + 3 / 2, -1])
    means[7] = np.array([3 / 2, 0])
    means[8] = np.array([1 + 3 / 2, 1])
    covariance_matrices = np.zeros((9, 2, 2))
    covariance_matrices[0] = covariance_matrix_1
    covariance_matrices[1] = covariance_matrix_2
    covariance_matrices[2] = covariance_matrix_3
    covariance_matrices[3] = covariance_matrix_4
    covariance_matrices[4] = covariance_matrix_5
    covariance_matrices[5] = covariance_matrix_6
    covariance_matrices[6] = covariance_matrix_7
    covariance_matrices[7] = covariance_matrix_8
    covariance_matrices[8] = covariance_matrix_9
    weights = np.array([12 / 25, 12 / 25, 1 / 350, 1 / 350, 1 / 350, 8 / 350, 1 / 350, 1 / 350, 1 / 350])
    for i in range(N):
        if u[i]<=168/350:
            sample[i]=np.random.multivariate_normal(means[0],covariance_matrix_1)
        elif 168/350<u[i]<=336/350:
            sample[i] = np.random.multivariate_normal(means[1], covariance_matrix_2)
        elif 336/350<u[i]<=337/350:
            sample[i] = np.random.multivariate_normal(means[2], covariance_matrix_3)
        elif 337/350<u[i]<=338/350:
            sample[i] = np.random.multivariate_normal(means[3], covariance_matrix_4)
        elif 338/350<u[i]<=339/350:
            sample[i] = np.random.multivariate_normal(means[4], covariance_matrix_5)
        elif 339/350<u[i]<=347/350:
            sample[i] = np.random.multivariate_normal(means[5], covariance_matrix_6)
        elif 347/350<u[i]<=348/350:
            sample[i] = np.random.multivariate_normal(means[6], covariance_matrix_7)
        elif 348/350<u[i]<=349/350:
            sample[i] = np.random.multivariate_normal(means[7], covariance_matrix_8)

        else:
            sample[i] = np.random.multivariate_normal(means[8],covariance_matrix_9)

    return pd.DataFrame(sample,columns=['x1','x2'])


