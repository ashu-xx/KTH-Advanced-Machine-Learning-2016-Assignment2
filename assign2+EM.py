
# coding: utf-8

# # EM

# In[76]:

import numpy as np
global K,N
K= 50
N = 50


# In[77]:

def data_gen(t_dist,z_dist):
    T = np.zeros([K,N],dtype=int)
    Z = np.zeros([N,K],dtype=int)
    for k in range(K):
        T[k,:] = 1+np.random.choice(6,N,p=t_dist[k,:])
    for n in range(N):
        Z[n,:] = 1+np.random.choice(6,K,p=z_dist[n,:])
    D = Z + np.transpose(T)
    "a row in S has sum for one player across K tables"
    return D


# In[78]:

"a row in r[n.k] has responsibility for 6 values of table's dice for a given value of player's dice"
def resp_gen(t_dist,z_dist):
    r=np.zeros([N,K,6,6],dtype=float)
    for n in range(N):
        for k in range(K):
            for i in range(6):
                for j in range(6):
                    r[n,k,i,j]= t_dist[k,j]*z_dist[n,i]
    return r


# In[79]:

def update_dist(D,r):
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


# In[80]:

t_dist_r = np.zeros([K,6],dtype=float)
z_dist_r = np.zeros([N,6],dtype=float)
"Real"
for k in range(K):
    t_dist_r[k,:] = np.array([1/12,1/4, 1/12, 1/4,1/12,1/4],dtype=float)
    "t_dist_r[k,:] = np.array([1/6,1/6,1/6,1/6,1/6,1/6],dtype=float)"
for n in range(N):
    z_dist_r[n,:] = np.array([1/12,1/4, 1/12, 1/4,1/12,1/4],dtype=float)
    "z_dist_r[n,:] = np.array([1/6,1/6,1/6,1/6,1/6,1/6],dtype=float)"
D = data_gen(t_dist_r,z_dist_r)
print(D)


# In[81]:

t_dist = np.zeros([K,6],dtype=float)
z_dist = np.zeros([N,6],dtype=float)
"Starting point"
for k in range(K):
    t_dist[k,:] = np.array([1/6,1/6,1/6,1/6,1/6,1/6],dtype=float)
for n in range(N):
    z_dist[n,:] = np.array([1/6,1/6,1/6,1/6,1/6,1/6],dtype=float)
iter=1000
for i in range(iter):
    "responsibility gen by prob dist of table and player"
    r=resp_gen(t_dist,z_dist)
    [t_dist,z_dist]=update_dist(D,r)
print("Number of iterations: ",iter)
print("Table's Dice Distribution:")
print(t_dist)
print("Player's Dice Distribution:")
print(z_dist)


# In[82]:

D_new = data_gen(t_dist,z_dist)


# In[83]:

def sum_pr(Sp):
    freq = np.zeros(11,dtype=float)
    for n in range(N):
        for k in range(K):
            freq[Sp[n,k]-2]=freq[Sp[n,k]-2]+1
    freq = freq * 100/(N*K)
    return freq


# In[84]:


import matplotlib.pyplot as plt
n_groups = 11

means_men = sum_pr(D_new)

means_women =sum_pr(D)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.3

opacity = 0.6
rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 label='EM')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
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



