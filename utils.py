import numpy             as np
import matplotlib.pyplot as plt


from fermat               import Fermat
from scipy.spatial        import  distance_matrix
from scipy.sparse         import csr_matrix
from scipy.sparse.csgraph import shortest_path

########################
#Generation of datasets#
########################

def generateCircle(N_signal = 100, sd = 0):
    '''
    Sample N_signal points (signal) points from the uniform distribution on the unit circle in R^2.
    If sd is nonzero, add a Gaussian noise to the points
        
    Input: 
    N_signal: number of sample points on the circle
    sd: standard deviation of the noise    
    
    Output : 
    data : a (N_signal)x2 matrix, the sampled points concatenated 
    '''
    rand_uniform = np.random.rand(N_signal)*2-1    
    X = np.cos(2*np.pi*rand_uniform)
    Y = np.sin(2*np.pi*rand_uniform)

    data = np.stack((X,Y)).transpose()
    
    data = data+np.random.normal(0,sd**2, np.shape(data)) #add noise

    return data

def generateLemniscate(N_signal = 100, sd= 0):
    '''
    Sample N_signal points (signal) points from the uniform distribution on the lemniscate.
        
    Input: 
    N_signal: number of sample points on the circle  
    
    Output : 
    data : a (N_signal)x2 matrix, the sampled points concatenated 
    '''
    
    I = np.linspace(0, 2*np.pi, N_signal+1)
    I = I[0:-1]     

    X=np.sin(I)/(1+np.cos(I)**2)
    Y=np.cos(I)*np.sin(I)/(1+np.cos(I)**2)
    
    L = np.column_stack([2.2*X,2.9*Y])
    
    L = L + np.random.normal(0,sd**2, np.shape(L)) #add noise

    return L


def generateLemniscateLifted(N_signal = 100, l=0.3, sd = 0):
    L = generateLemniscate(N_signal)
    X = L[:, 0]
    Y = L[:, 1]
    alpha = l/2
    step = int(np.ceil(N_signal/4))
    Z1 = -alpha*np.linspace(0,1,step) + alpha
    Z2 = -alpha*np.linspace(0,1,step)
    Z3 =  alpha*np.linspace(0,1,step) - alpha
    Z4 =  alpha*np.linspace(0,1,step)
    Z = np.concatenate([Z1,Z2,Z3,Z4])
    
    LL = np.column_stack((X,Y,Z[0:N_signal]))
    
    LL = LL + np.random.normal(0,sd**2, np.shape(LL)) #add noise
    return LL


def generateEyeglasses(N_signal= 100, l = 0.5, sd = 0):
    
    X = generateLemniscate(N_signal)
    
    X1 = np.concatenate([X[X[:, 0]>l],X[X[:, 0]<-l]])
    m = N_signal-len(X1)
    seg  = np.linspace(-l, l, int(m/2))
    
    x = np.max(X1[(X1[:,1]<=l*1.2) & (X1[:,1]>0)][:,1])
    k = (x-l/2)/(seg[0]**2)
    
    y2 = (k*seg**2)+ l/2
    y3 = -(k*seg**2)-l/2
    X_obs = np.concatenate([X1[:,0],seg, seg])
    Y_obs = np.concatenate([X1[:,1], y2, y3])
    
    E = np.column_stack((X_obs,Y_obs))
    
    E = E + np.random.normal(0,sd**2, np.shape(E)) #add noise
    
    return E
    
    
####################################
#Computation of intrinsic distances#
####################################

def compute_fermat_distance(data, p):
    
    #Compute euclidean distances
    distances = distance_matrix(data,data)
    
    # Initialize the model
    fermat = Fermat(alpha = p, path_method='FW') 

    # Fit
    fermat.fit(distances)
    
    ##Compute Fermat distances
    fermat_dist = fermat.get_distances()
    
    return  fermat_dist


def compute_knn_geodesic_distance(data, k):
    
    distances = distance_matrix(data,data)

    # Initialize the model
    f_aprox_D = Fermat(1, path_method='D', k=k) 

    # Fit
    f_aprox_D.fit(distances)
    knn_dist = f_aprox_D.get_distances() 
    
    return knn_dist

def compute_eps_geodesic_distance(data, eps):
    
    distances = distance_matrix(data,data)
    eps_matrix = np.where(distances>eps, 0, distances)
    graph = csr_matrix(eps_matrix)
    eps_dist = shortest_path(csgraph=graph, directed=False)
    
    return eps_dist


def plotEpsGraph(data, eps, size):
    fig = plt.figure( figsize=size ) 
    ax = fig.add_subplot(1, 1, 1) 
    ax.axis('off')
    plt.scatter(data[:,0],data[:,1], alpha = 0.5, c='darkslategrey');

    d = distance_matrix(data, data)
    
    #plot edges 
    for j in range(np.shape(data)[0]):
        for k in range(j, np.shape(data)[0]):
            if d[j,k]<eps:
                plt.plot([data[j,0], data[k,0]], [data[j,1], data[k,1]], c = 'darkslategrey', alpha = 0.5)


def Fermat_dgm(data, p, rescaled = False, d=None, mu=None):
    distance_matrix = compute_fermat_distance(data, p)
    if rescaled:
        distance_matrix = (distance_matrix*len(data)**((p-1)/d))/mu
    rips = Rips()
    dgms = rips.fit_transform(distance_matrix, distance_matrix=True)
    fig = plt.figure()
    rips.plot(dgms, lifetime = True)
    plt.title('Fermat distance with p = %s'%(p))
    return dgms