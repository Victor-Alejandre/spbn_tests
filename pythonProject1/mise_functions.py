import numpy as np
import scipy
import math
def amise(data,weigths, means, covariance_matrixs, bandwidth):
    N=data.shape[0]
    d=data.shape[1]

    det_H=np.linalg.det(bandwidth)
    amise = (N ** (-1)) * ((4 * math.pi) ** (-d / 2)) * (det_H ** (-1/2))

    k=len(weigths)
    zero_mean = np.zeros(d)
    I_d=np.eye(d)
    for i in range(k):
        for j in range(k):
            point = means[i]-means[j]
            cov=covariance_matrixs[i]+covariance_matrixs[j]
            A=np.linalg.inv(cov)
            B=np.matmul(A,I_d-2*np.matmul(np.outer(point,point),A))
            C=np.matmul(A,I_d-np.matmul(np.outer(point,point),A))

            amise = amise + weigths[i] * weigths[j] * scipy.stats.multivariate_normal.pdf(point,zero_mean,cov) * \
                    (2 * np.diagonal(np.matmul(np.matmul(bandwidth,A),np.matmul(bandwidth,B))).sum() + (np.diagonal(np.matmul(bandwidth,C)).sum())**2)

    return amise


def mise(data, weigths, means, covariance_matrixs, bandwidth):
    N = data.shape[0]
    d = data.shape[1]

    det_H = np.linalg.det(bandwidth)
    mise = (N ** (-1)) * ((4 * math.pi) ** (-d / 2)) * (det_H ** (-1 / 2))
    k = len(weigths)
    zero_mean = np.zeros(d)
    I_d = np.eye(d)
    omega_0=np.zeros((k,k))
    omega_1=np.zeros((k,k))
    omega_2=np.zeros((k,k))
    for i in range(k):
        for j in range(k):
            point = means[i] - means[j]
            cov = covariance_matrixs[i] + covariance_matrixs[j]
            omega_0[i,j] = scipy.stats.multivariate_normal.pdf(point,zero_mean,cov)
            omega_1[i, j] = scipy.stats.multivariate_normal.pdf(point,zero_mean,bandwidth+cov)
            omega_2[i, j] = scipy.stats.multivariate_normal.pdf(point,zero_mean,2*bandwidth+cov)

    quadratic_matrix=(1-N**(-1))*omega_2 - 2*omega_1 + omega_0

    mise=mise + np.dot(weigths.T,np.dot(quadratic_matrix, weigths))

    return mise


covariance_matrix_1 = np.array([[1,-0.5],[-0.5,1]])
covariance_matrix_2 = np.array([[1, 0.5], [0.5, 1]])
covariance_matrices=np.zeros((2,2,2))
covariance_matrices[0]= covariance_matrix_1
covariance_matrices[1]= covariance_matrix_2
means=np.zeros((2,2))
means[0]= np.array([-1,-1])
means[1]=np.array([1,1])
weigths=np.array([0.5,0.5])



