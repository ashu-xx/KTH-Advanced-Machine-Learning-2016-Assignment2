
# coding: utf-8

# # Casino Model

# In[128]:

import numpy as np
import matplotlib.pyplot as plt


# In[129]:

global K, Num, Tr
Tr = np.array([[1/4,3/4],[3/4, 1/4]], dtype=float)
K=100
Num = 100
def distr(x,l):
    return {
        'u': np.random.choice(6, l, p=[1/6, 1/6, 1/6,1/6,1/6,1/6]),
        'b_ev': np.random.choice(6, l, p=[1/12, 1/4, 1/12,1/4,1/12,1/4]),
        'b_od': np.random.choice(6, l, p=[1/4, 1/12, 1/4,1/12,1/4,1/12]),
        'bo': np.random.choice(6, l, p=[1/3, 0, 1/3,0,1/3,0]),
        'be': np.random.choice(6, l, p=[0, 1/3, 0,1/3,0,1/3]),
        'b6':np.random.choice(6, l, p=[1/10, 1/10, 1/10,1/10,1/10,1/2]),
        'b4':np.random.choice(6, l, p=[1/10, 1/10, 1/10,1/2,1/10,1/10]),
    }[x]
def mkc():
    r = np.zeros((K,), dtype=np.int)
    r[0] = np.random.choice(2, 1, p=[1/2, 1/2])
    for i in range(1,K):
        if(r[i-1]== 0):
            r[i] = np.random.choice(2, 1, p=Tr[0,:])
        else :
            r[i] = np.random.choice(2, 1, p=Tr[1,:])
    return r    


# In[130]:

def play(t1,t2,p):
    T = np.array([distr(t1,K),distr(t2,K)])
    Z = distr(p,K)
    S = Z
    R = mkc()
    for i in range(K):
        if(R[i]==0):
            S[i] = 2 + S[i] + T[0,i]
        else:
            S[i] = 2 + S[i] + T[1,i]
    return S


# In[131]:

Sp = np.zeros((Num,K), dtype=int)
for i in range(Num):
    Sp[i,:] = play('b_ev','b_ev','be')


# In[132]:

def sum_pr(Sp):
    freq = np.zeros(11,dtype=float)
    for n in range(Num):
        for k in range(K):
            freq[Sp[n,k]-2]=freq[Sp[n,k]-2]+1
    freq = freq * 100/(Num*K)
    return freq


# In[133]:

n_groups = 11

means_men = sum_pr(Sp)

means_women =(3,6,8,11,14,17,14,11,8,6,3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.3

opacity = 0.6
rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Biased to even')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Ideal-Unbiased')

plt.xlabel('Sum')
plt.ylabel('% Occurance')
"plt.title('Obtained frequency of sum values')"
plt.xticks(index + bar_width, (2,3,4,5,6,7,8,9,10,11,12))
plt.legend()

plt.tight_layout()
plt.show()


# In[134]:

n_groups = 11

means_men = sum_pr(Sp)

means_women =(3,6,8,11,14,17,14,11,8,6,3)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.3

opacity = 0.6
rects1 = plt.bar(index, means_men, bar_width,
                 alpha=opacity,
                 color='b',
                 label='Biased to 6')

rects2 = plt.bar(index + bar_width, means_women, bar_width,
                 alpha=opacity,
                 color='r',
                 label='Ideal-Unbiased')

plt.xlabel('Sum')
plt.ylabel('% Occurance')
"plt.title('Obtained frequency of sum values')"
plt.xticks(index + bar_width, (2,3,4,5,6,7,8,9,10,11,12))
plt.legend()

plt.tight_layout()
plt.show()


# In[119]:

def pr(S):
    for i in range(Num):
        print("P{0}".format(i+1), end="")
        for j in range(K):
            print ('&',S[i,j], end="")
        print("\\\\ \hline")


# In[160]:

pr(Sp)


# In[ ]:



