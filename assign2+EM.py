
# coding: utf-8

# # EM

# In[50]:

import numpy as np
global K,N
K= 50
N = 50


# In[51]:

def data_gen(t_dist,z_dist):
    "Generating data for the algorithm"
    T = np.zeros([K,N],dtype=int)
    Z = np.zeros([N,K],dtype=int)
    D = np.zeros([N,K],dtype=int)
    for k in range(K):
        T[k,:] = 1+np.random.choice(6,N,p=t_dist[k,:])
    for n in range(N):
        Z[n,:] = 1+np.random.choice(6,K,p=z_dist[n,:])
    D = Z + np.transpose(T)
    "a row in S has sum for one player across K tables"
    return D


# In[52]:

"a row in r[n.k] has responsibility for 6 values of table's dice for a given value of player's dice"
def resp_gen(t_dist,z_dist,D):
    r=np.zeros([N,K,6,6],dtype=float)
    for n in range(N):
        for k in range(K):
            for i in range(6):
                for j in range(6):
                    if(D[n,k]==i+j+2):
                        r[n,k,i,j]= t_dist[k,j]*z_dist[n,i]
            if(np.sum(r[n,k,:,:])==0):
                print("Its fucking zero!",[n,k,D[n,k]])
            else:
                r[n,k,:,:] = r[n,k,:,:]/np.sum(r[n,k,:,:])                 
    return r


# In[53]:

def update_dist(D,r):
    "Updates the dice distribution using values for responibility"
    t_dist= np.zeros([K,6],dtype=float)
    z_dist= np.zeros([N,6],dtype=float)
    
    "theta"
    for n in range(N):
        "theta^n"
        for i in range(6):
            "theta_i^n"
            resp = 0
            for k in range(K):
                for j in range(6):
                    if(i+j+2==D[n,k]):
                        resp=resp+r[n,k,i,j]
            z_dist[n,i] = resp
        if(np.sum(z_dist[n,:])!=0):
            z_dist[n,:] = z_dist[n,:]/np.sum(z_dist[n,:])
        
    "pi"
    for k in range(K):
        "pi^k"
        for j in range(6):
            "theta_j^k"
            resp = 0
            for n in range(N):
                for i in range(6):
                    if(i+j+2==D[n,k]):
                        resp = resp +r[n,k,i,j]
            t_dist[k,j] = resp
        if(np.sum(t_dist[k,:])!=0):
            t_dist[k,:] = t_dist[k,:]/np.sum(t_dist[k,:])
    return [t_dist,z_dist]


# In[54]:

t_dist_r = np.zeros([K,6],dtype=float)
z_dist_r = np.zeros([N,6],dtype=float)
"Original Dice Distribution"
for k in range(K):
    t_dist_r[k,:] = np.array([1/10,1/10,1/10,1/10,1/10,1/2],dtype=float)
for n in range(N):
    if(n%2==1):
        z_dist_r[n,:] = np.array([1/10,1/10,1/10,1/2,1/10,1/10],dtype=float)
    else:
        z_dist_r[n,:] = np.array([1/10,1/10,1/10,1/2,1/10,1/10],dtype=float)
D = data_gen(t_dist_r,z_dist_r)
print(D)


# In[55]:

t_dist = np.zeros([K,6],dtype=float)
z_dist = np.zeros([N,6],dtype=float)
"Initial guess for dice distribution"
for k in range(K):
    t_dist[k,:] = np.array([1/3,2/15,2/15,2/15,2/15,2/15],dtype=float)
for n in range(N):
    z_dist[n,:] = np.array([1/3,2/15,2/15,2/15,2/15,2/15],dtype=float)
iter=100
for i in range(iter):
    "responsibility gen by prob dist of table and player"
    r=resp_gen(t_dist,z_dist,D)
    [t_dist,z_dist]=update_dist(D,r)
print("Number of iterations: ",iter)
print("Table's Dice Distribution:")
print(t_dist)
print("Player's Dice Distribution:")
print(z_dist)


# In[56]:

print(D)


# In[60]:

"Data generated using distribution deduced by EM"
D_new = data_gen(t_dist,z_dist)


# In[61]:

"generates frequency of occurrence of sum values"
def sum_pr(Sp):
    freq = np.zeros(11,dtype=float)
    for n in range(N):
        for k in range(K):
            freq[Sp[n,k]-2]=freq[Sp[n,k]-2]+1
    freq = freq * 100/(N*K)
    return freq


# In[62]:

"Visualizing difference in frequency of occurrence of sum values between original and deduced dice distribution"
import matplotlib.pyplot as plt
n_groups = 11

sum_new = sum_pr(D_new)

sum_orig =sum_pr(D)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.3

opacity = 0.6
rects1 = plt.bar(index, sum_new, bar_width,
                 alpha=opacity,
                 color='b',
                 label='EM')

rects2 = plt.bar(index + bar_width, sum_orig, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Original')

plt.xlabel('Sum')
plt.ylabel('% Occurance')
"plt.title('Obtained frequency of sum values')"
plt.xticks(index + bar_width, (2,3,4,5,6,7,8,9,10,11,12))
plt.legend()

plt.tight_layout()
plt.show()


# In[ ]:




# In[ ]:




# In[ ]:



