import scipy.stats
import math
import numpy as np
import pandas as pd


def F_1(x,rho=0.9):
    covariance_matrix=1/(1-rho**2)*np.array([[1, rho, rho**2, rho**3, rho**4],
                                            [ rho, 1, rho, rho**2, rho**3],
                                            [ rho**2, rho, 1, rho, rho**2],
                                            [ rho**3, rho**2, rho, 1,  rho],
                                            [ rho**4, rho**3, rho**2, rho, 1]])

    return scipy.stats.multivariate_normal.pdf(x, 2*np.ones(5), covariance_matrix)

def F_2(x,d=5):
    covariance_matrix=np.eye(d)
    return 0.5*scipy.stats.multivariate_normal.pdf(x, 2*np.ones(5), covariance_matrix)+0.5*scipy.stats.multivariate_normal.pdf(x, -1.5*np.ones(5), covariance_matrix)

def F_3(x,d=5):
    covariance_matrix = np.eye(d)
    return 0.5*scipy.stats.multivariate_t.pdf(x, 2*np.ones(5), covariance_matrix,5)+0.5*scipy.stats.multivariate_t.pdf(x, -1.5*np.ones(5), covariance_matrix,5)




def F_4(x,rho=0.9):
    covariance_matrix=1/(1-rho**2)*np.array([[1, rho, rho**2, rho**3, rho**4],
                                            [ rho, 1, rho, rho**2, rho**3],
                                            [ rho**2, rho, 1, rho, rho**2],
                                            [ rho**3, rho**2, rho, 1,  rho],
                                            [ rho**4, rho**3, rho**2, rho, 1]])
    return 2*scipy.stats.multivariate_normal.pdf(x,2*np.ones(5),covariance_matrix)*scipy.stats.multivariate_normal.cdf(-0.5*np.dot(np.ones(5),x-2*np.ones(5)),0,1)

def F_5(x):
    covariance_matrix=np.eye(5)
    point=x-np.ones(5)*2
    return 2*scipy.stats.multivariate_t.pdf(x,np.zeros(5),covariance_matrix,5)*scipy.stats.multivariate_t.cdf(-0.5*np.dot(np.ones(5),x-2*np.ones(5))*math.sqrt((10)/((np.dot(point.T ,np.dot(np.linalg.inv(covariance_matrix), point)))+5)),0,1,10)


def F_6(x):
    covariance_matrix = np.array([[4/9,0],[0,4/9]])

    return 0.5*scipy.stats.multivariate_normal.pdf(x, np.array([1,0]), covariance_matrix)+0.5*scipy.stats.multivariate_normal.pdf(x, np.array([-1,0]), covariance_matrix)

def F_7(x):
    covariance_matrix_1 = np.array([[1,-0.5],[-0.5,1]])
    covariance_matrix_2 = np.array([[1, 0.5], [0.5, 1]])

    return 0.5*scipy.stats.multivariate_normal.pdf(x, np.array([-1,-1]), covariance_matrix_1)+0.5*scipy.stats.multivariate_normal.pdf(x, np.array([1,1]), covariance_matrix_2)
def F_8(x):
    covariance_matrix_1 = np.array([[9/25,63/250],[63/250,9/25]])
    covariance_matrix_2 = np.array([[9/25,0],[0,9/25]])

    return 3/7*scipy.stats.multivariate_normal.pdf(x, np.array([-1,0]), covariance_matrix_1)+3/7*scipy.stats.multivariate_normal.pdf(x, np.array([1,2/np.sqrt(3)]), covariance_matrix_2)+1/7*scipy.stats.multivariate_normal.pdf(x, np.array([1,-2/np.sqrt(3)]), covariance_matrix_2)

def F_9(x):
    covariance_matrix_1 = np.eye(2)
    covariance_matrix_2 = np.array([[0.8,-0.72],[-0.72,0.8]])

    return 4/11*scipy.stats.multivariate_normal.pdf(x, np.array([-2,2]), covariance_matrix_1)+3/11*scipy.stats.multivariate_normal.pdf(x, np.array([0,0]), covariance_matrix_2)+4/11*scipy.stats.multivariate_normal.pdf(x, np.array([2,-2]), covariance_matrix_1)

def sampling_F_1(N=100,rho=0.9):
    covariance_matrix= 1/(1-rho**2)*np.array([[1, rho, rho**2, rho**3, rho**4],
                                            [ rho, 1, rho, rho**2, rho**3],
                                            [ rho**2, rho, 1, rho, rho**2],
                                            [ rho**3, rho**2, rho, 1,  rho],
                                            [ rho**4, rho**3, rho**2, rho, 1]])
    return pd.DataFrame(np.random.multivariate_normal(2*np.ones(5),covariance_matrix,N),columns=['x1','x2','x3','x4','x5'])


def sampling_F_2(N=100,d=5):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,d))
    for i in range(N):
        if u[i]<=0.5:
            sample[i]=np.random.multivariate_normal(-1.5*np.ones(d),np.eye(d))
        else:
            sample[i] = np.random.multivariate_normal(2 * np.ones(d), np.eye(d))

    return pd.DataFrame(sample,columns=['x1','x2','x3','x4','x5'])


def sampling_F_3(N=100,d=5):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,d))
    for i in range(N):
        if u[i]<=0.5:
            sample[i]=scipy.stats.multivariate_t.rvs(-1.5 * np.ones(d), np.eye(d),5)
        else:
            sample[i] = scipy.stats.multivariate_t.rvs(2 * np.ones(d), np.eye(d),5)

    return pd.DataFrame(sample,columns=['x1','x2','x3','x4','x5'])

def sampling_F_4(N=100, rho=0.9):
    covariance_matrix=1/(1-rho**2)*np.array([[1, rho, rho**2, rho**3, rho**4],
                                            [ rho, 1, rho, rho**2, rho**3],
                                            [ rho**2, rho, 1, rho, rho**2],
                                            [ rho**3, rho**2, rho, 1,  rho],
                                            [ rho**4, rho**3, rho**2, rho, 1]])
    delta=1/(np.sqrt(1+np.dot(-0.5*np.ones(5).T,np.dot(covariance_matrix,-0.5*np.ones(5)))))*np.dot(covariance_matrix,-0.5*np.ones(5))
    covariance_matrix=np.block([[np.ones(1),     delta],
                         [delta[:, None], covariance_matrix]])

    auxiliary_sample=scipy.stats.multivariate_normal.rvs(np.zeros(6), covariance_matrix,N)
    x0, x1 = auxiliary_sample[:, 0], auxiliary_sample[:, 1:]
    inds = x0 <= 0
    x1[inds] = -1 * x1[inds]


    return pd.DataFrame(x1, columns=['x1', 'x2', 'x3', 'x4', 'x5'])



def sampling_F_5(N=100):
    covariance_matrix=np.eye(5)
    sample = np.zeros((N, 5))
    n_sample=0
    while n_sample<(N-1):
        z = scipy.stats.multivariate_t.rvs(np.zeros(5),covariance_matrix,5)
        u = np.random.uniform(0, 2 * scipy.stats.multivariate_t.pdf(z,np.zeros(5),covariance_matrix,5))
        if not u > F_5(z):
            sample[n_sample] = z
            n_sample += 1




    return pd.DataFrame(sample, columns=['x1', 'x2', 'x3', 'x4', 'x5'])


def sampling_F_6(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix = np.array([[4/9,0],[0,4/9]])
    for i in range(N):
        if u[i]<=0.5:
            sample[i]=np.random.multivariate_normal(np.array([1,0]),covariance_matrix)
        else:
            sample[i] = np.random.multivariate_normal(np.array([-1,0]),covariance_matrix)

    return pd.DataFrame(sample,columns=['x1','x2'])

def sampling_F_7(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = np.array([[1, -0.5], [-0.5, 1]])
    covariance_matrix_2 = np.array([[1, 0.5], [0.5, 1]])
    for i in range(N):
        if u[i]<=0.5:
            sample[i]=np.random.multivariate_normal(np.array([1,1]),covariance_matrix_2)
        else:
            sample[i] = np.random.multivariate_normal(np.array([-1,-1]),covariance_matrix_1)

    return pd.DataFrame(sample,columns=['x1','x2'])

def sampling_F_8(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = np.array([[9 / 25, 63 / 250], [63 / 250, 9 / 25]])
    covariance_matrix_2 = np.array([[9 / 25, 0], [0, 9 / 25]])
    for i in range(N):
        if u[i]<=3/7:
            sample[i]=np.random.multivariate_normal(np.array([-1,0]),covariance_matrix_1)
        elif 3/7<u[i]<=6/7:
            sample[i] = np.random.multivariate_normal(np.array([1, 2/np.sqrt(3)]), covariance_matrix_2)
        else:
            sample[i] = np.random.multivariate_normal(np.array([1, -2/np.sqrt(3)]), covariance_matrix_2)

    return pd.DataFrame(sample,columns=['x1','x2'])

def sampling_F_9(N=100):
    u=np.random.uniform(size=N)
    sample=np.zeros((N,2))
    covariance_matrix_1 = np.eye(2)
    covariance_matrix_2 = np.array([[0.8, -0.72], [-0.72, 0.8]])
    for i in range(N):
        if u[i]<=4/11:
            sample[i]=np.random.multivariate_normal(np.array([-2,2]),covariance_matrix_1)
        elif 4/11<u[i]<=7/11:
            sample[i] = np.random.multivariate_normal(np.array([0,0]), covariance_matrix_2)
        else:
            sample[i] = np.random.multivariate_normal(np.array([2,-2]), covariance_matrix_2)

    return pd.DataFrame(sample,columns=['x1','x2'])


