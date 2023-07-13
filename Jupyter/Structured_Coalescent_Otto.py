##################################################
##### Module Package 'Structured Coalescent' #####
#####                                        #####
#####                                        #####
#####                                        #####
##### by Moritz Otto                         #####
#####                                        #####
##### 13th July 2023                         #####
#####                                        #####
#####                                        #####
##### moritz.otto@uni-koeln.de               #####
##### Import python packages and self        #####
##### written functions                      #####
##################################################


##### 1) Import standard packages
import numpy as np
import math
import random
import scipy
import scipy.sparse as sps
import scipy.sparse.linalg as sp_linalg
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


##### Import MSPRIME
import msprime  # >= 0.5

from tqdm import tqdm

from IPython.display import Image
from IPython.display import clear_output
from IPython.display import SVG


##### Idea: Use msprime to determine pairwise coalescence times of a Markovian Jump process with
#####       state-space 'states'
#####       Transition / Migration matrix 'M'
#####       population distribution 'pop_size'

#######################################
###### Pairwise coalescence times #####
#######################################

def make_ctime_msprime(states,M,pop_size,iterations=100,rep=1,sample_state=[np.inf,np.inf], file_name="CTime_msprime.csv",save_as_file=True):
    #Sample rep-many starting points
    #If rep=inf, go through ALL pairwise starting points of sample state
    #For each starting point, run iterations-many coalescence events, using ms_prime
    #Safe the outcome in Coalescence_time_data

    Coalescence_time_data = [(['particles'],['distr'])]
    n = len(states)
    
    if (np.inf in sample_state):
        sample_state = range(n)
    
    def run_sim(i,j,Coalescence_time_data):
        #clear_output()
        #print(f'{states[i]},{states[j]}')
        alpha=np.zeros(n)
        alpha[i] += 1
        alpha[j] += 1
        pop_configs = []
        for k in range(n):
            pop_configs.append(msprime.PopulationConfiguration(initial_size=pop_size[k],sample_size=int(alpha[k])))

        T_mrca = np.zeros(iterations)
        k=0
        for ts in msprime.simulate(population_configurations= pop_configs, migration_matrix=M, num_replicates=iterations, record_provenance = False):
            tree = ts.first()
            T_mrca[k] = tree.time(tree.root)
            k += 1
        Coalescence_time_data = Coalescence_time_data + [([states[i],states[j]],T_mrca.copy())]
        return(Coalescence_time_data)
    
    print(file_name)
    if (rep == np.inf):
        for i in tqdm(sample_state):
            for j in sample_state:
                Coalescence_time_data = run_sim(i,j,Coalescence_time_data)
                if(save_as_file):
                    with open(file_name, 'w') as file:
                        file.write(str(Coalescence_time_data))
    else:
        for k in tqdm(range(rep)):
            i = random.randint(0,n-1)
            j = random.randint(0,n-1)
            Coalescence_time_data = run_sim(i,j,Coalescence_time_data)
            if(save_as_file):
                with open(file_name, 'w') as file:
                    file.write(str(Coalescence_time_data))
                
    
    return(Coalescence_time_data)



##### Define the functions for the Transition Matrix P_UE of the 
##### Unequal Recombination jumping process

def discrete_gamma_prob(k,E):
    alpha = 2/E
    Z = np.exp(alpha) /(np.exp(alpha)-1)**2
    p = k * np.exp(-alpha * k) / Z
    return p

def P_E_func(k,a,b,E):
    alpha = 2/E
    Z = (a+b) * (1-np.exp(alpha)) / (np.exp(alpha)-1)
    out = []
    for k_val in k:
        c1 = max(0,k_val-a)
        c2 = max(0,k_val-b)
        out.append(1/Z * (2*np.exp(-alpha*k_val) - np.exp(-alpha* c1) - np.exp(-alpha *c2)))
    return out

def coef(a,b,n_max):
    #enumeration of (1,1), (1,2), (1,3), ... , (1,n_max), (2,1), (2,2), ... , (n_max,1),...,(n_max,n_max)
    return (a-1)*n_max+b - 1

def n_max_UE(E):
    n_max = int(np.around(-E/2 * np.log(0.0001 / (np.exp(2/E)-1) ),decimals=0))
    return(n_max)

def inv_coef(k,n_max):
    #return (1,1), (1,2), (1,3), ... , (1,n_max), (2,1), (2,2), ... , (n_max,1),...,(n_max,n_max)
    #for count k
    a = int(math.floor(k / n_max))
    b = k-a*n 
    return [a,b]  


#######################################
##### Unequal recombination model #####
#######################################

def states_UE(E):
    states = []
    n_max = n_max_UE(E)
    for i in range(1,n_max+1):
        for j in range(1,n_max+1):
            states += [(f'{i}|{j}')]
    return(states)


def pop_distr_UE(E,states):
    pop_distr = []
    for x in states:
        a = int(x.split('|')[0])
        b = int(x.split('|')[1])
        L = a+b-1
        pop_distr += [discrete_gamma_prob(L,E)]
    return(pop_distr)


def make_P_UE(E):
    n_max = n_max_UE(E)
    P = np.zeros((n_max**4)).reshape(n_max**2,n_max**2)


    for a in tqdm(range(1,n_max+1)):
        for b in range(1,n_max+1):
            #(a,b) - row
            #We jump from (a,b) to either (a,k) or (k,b)

            for k in range(1,n_max):
                P[coef(a,b,n_max),coef(k,b,n_max)] += a/(a+b) * P_E_func([k],a,b,E)[0]
                P[coef(a,b,n_max),coef(a,k,n_max)] += b/(a+b) * P_E_func([k],a,b,E)[0]

    for i in range(0,n_max**2):
        P[i,i] = 0
    
    
    for i in range(0,n_max**2):
        P[i,:] *= 1/sum(P[i,:])
    
    
    
    return P

##############################
##### Continental Island #####
##############################

def states_Cont():
    states = []
    for i in range(5):
        for j in range(5):
            states += [[i,j]]
    return(states)

def pop_distr_Cont():
    pop_distr = np.array([
        1.,2,4,2,1,
        2,4,8,4,2,
        4,8,16,8,4,
        2,4,8,4,2,
        1,2,4,2,1
        ])
    return(1/100 * pop_distr)

def jump_prob(a,b,c,d):
    prob_vec = np.array([0.05,0.2,0.5,0.2,0.05])

    out = 0
    if (a == c):
        out = 0.5*prob_vec[d]
    if (b == d):
        out = 0.5*prob_vec[c]
    if (a == c and b == d):
        out = 0
    return(out)

def make_P_Cont(states):
    P = np.empty((25,25))
    for a in range(5):
        for b in range(5):
            for c in range(5):
                for d in range(5):
                    P[states.index([a,b]),states.index([c,d])] = jump_prob(a, b, c, d)
    return(P)

############################
##### Symmetric Island #####
############################

def states_Sym():
    states = [1,2,3,4,5]
    return(states)

def make_P_Sym():
    P = np.array([
        [0,1.,1,1,1],
        [1,0,1,1,1],
        [1,1,0,1,1],
        [1,1,1,0,1],
        [1,1,1,1,0]
    ])
    return(P)

def pop_distr_Sym():
    pop_distr = np.array([1,1,1,1,1.])
    return(1/5 * pop_distr)


##### According to Wilkinson-Herbots, one may solve the linear equation system, to
##### determine the Laplace transformation of the pairwise coalescence time
#####
##### We do so, using sparse matrices

def make_A_Laplace_sps(M,coalescence,s):
    n = len(coalescence)
    A_sps = sps.coo_matrix((n**2, n**2))
    b_vec = np.zeros(n**2)
    tm=0
    r = []
    c = []
    d = []
    for i in tqdm(range(n)):
        for j in range(n):
            if (i != j):
                r += [tm]
                c += [coef(i+1,j+1,n)]
                d += [(sum(M[i,])-M[i,i])/2 + (sum(M[j,])-M[j,j])/2 + s]
                for k in range(n):
                    if (k != i):
                        r += [tm]
                        c += [coef(j+1,k+1,n)]
                        d += [0.5*(-M[i,k])]
                for k in range(n):
                    if (k != j):
                        r += [tm]
                        c += [coef(i+1,k+1,n)]
                        d += [0.5*(-M[j,k])]
            if (i == j):
                b_vec[tm] = coalescence[i]
                r += [tm]
                c += [coef(i+1,i+1,n)]
                d += [ coalescence[i] + (sum(M[i,])-M[i,i]) + s ]
                for k in range(n):
                    if (k != i):
                        r += [tm]
                        c += [coef(i+1,k+1,n)]
                        d += [-M[i,k] ]
            tm+=1
    
    A_sps += sps.coo_matrix((d, (r, c)),shape=(n**2,n**2))
    return(A_sps,b_vec)

def make_ctime_lineq(M,coalescence,srange):
    Laplace_sps_solved = []
    for it in range(len(srange)):
        clear_output()
        s = srange[it]
        print(f'{it+1}/{len(srange)}')
        print('Generate Coefficient matrix A:')
        new_A_mat , new_b = make_A_Laplace_sps(M*2,coalescence,s)
        print('Solve Ax=b')
        Laplace_sps_solved.append([s,sp_linalg.spsolve(new_A_mat,new_b)])
    
    #Formatting output
    T = Laplace_sps_solved
    new_mat = np.zeros((len(srange),len(T[0][1])+1))

    
    for i in range(len(srange)):
        new_mat[i,0] = T[i][0]
        new_mat[i,1:] = T[i][1]
        
    return(new_mat)

def emp_laplace(srange,data):
    out =[]
    for s in srange:
        out += [1/len(data) * sum(np.exp(-s*data))]
    return(out)

def E_Var_lineq(data):
    #Exp-value = d/ds E[exp(-sX)] | s=0
    out = []
    time=0
    
    for k in range(1,data.shape[1]):
        E_val = -(data[time+1,k] - data[time,k] ) / (data[time+1,0] - data[time,0])
        E2_val = (data[time+2,k] + data[time,k] - 2*data[time+1,k] ) / ((data[time+1,0] - data[time,0])**2)
        
        
        Var_val = E2_val - E_val*E_val
        
        out += [[k,np.around(E_val,decimals=4),np.around(np.sqrt(Var_val),decimals=4)]]
    
    return(out)

def E_Var_msprime(data):
    out = []
    for k in range(1,len(data)):
        out += [[k, np.around(np.mean(data[k][1]),decimals=4),np.around(np.sqrt(np.var(data[k][1])),decimals=4)]]
    return(out)


###################################
##### Site frequency spectrum #####
###################################


def make_SFS(N,M,m,alpha,pop_configs,theta,iterations=10000,file_name='SFS',save_as_file=True):
    my_bins = np.array(range(int(sum(alpha))+2))-0.5
    Out_table = np.append([-1,-2],my_bins[1:]-0.5)
    mu = theta / (2*N)
    
    SFS = []
    
    for tm in tqdm(range(iterations)):
        ts = msprime.simulate(population_configurations= pop_configs, migration_matrix=M, mutation_rate = mu)
        tree = ts.first()
        SNP_Mat = ts.genotype_matrix()
        SFS = np.append(SFS,SNP_Mat.sum(axis=1))
    counts, bins = np.histogram(SFS,bins=my_bins,density=True)
    Out_table = np.vstack([Out_table,np.append([theta,m],counts.copy())])
    
    if(save_as_file):
        f = open(file_name, 'w')
        f.write(str(Out_table))
        f.close()
    
    return(Out_table)

###################################    
##### Stationary distribution #####
###################################

def make_stat_distr(E):
    n_max = n_max_UE(E)
    def make_inv_P(E):

        
        A_mat = np.zeros((n_max**4)).reshape(n_max**2,n_max**2)


        for a in tqdm(range(1,n_max+1)):
            for b in range(1,n_max+1):
                #A_(a,b) - row
                #We either jump to (a,b) from (a,k) or (k,b)
                A_mat[coef(a,b,n_max),coef(a,b,n_max)] = -1
                for k in range(1,n_max):
                    A_mat[coef(a,b,n_max),coef(k,b,n_max)] += k/(k+b) * P_E_func([a],k,b,E)[0]
                    A_mat[coef(a,b,n_max),coef(a,k,n_max)] += k/(a+k) * P_E_func([b],a,k,E)[0]


        tmp = np.full((1, n_max**2), 1)
        A_mat = np.concatenate((A_mat, tmp))
        return(A_mat)
    
    A_mat = make_inv_P(E)
    A_solve = np.matmul(A_mat.transpose(),A_mat)
    
    b = np.zeros(n_max**2 + 1)
    b[-1] = 1
    b_solve = np.matmul(A_mat.transpose(),b)

    Stat_distr = np.linalg.solve(A_solve,b_solve)
    return(Stat_distr)


########################
###### Mixing time #####
########################

def my_dif(P,pi):
    n = len(pi)
    dif_vec = [1/2*sum(np.abs(np.subtract(P[i,:],pi))) for i in range(0,n)]
    return(max(dif_vec))

def Mixing_time(E):
    Stat_distr = make_stat_distr(E)
    P = make_P_UE(E)

    TV = [my_dif(P,Stat_distr)]
    P_tmp = P
    eps = 0.05
    while TV[-1] > 0.25:
        clear_output()
        print(f'Calculating P^{len(TV)} for E={E}')
        P_tmp = np.matmul(P_tmp,P)
        TV.append(my_dif(P_tmp,Stat_distr))
    t_mix = len(TV)
    return(t_mix)
