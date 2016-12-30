
# coding: utf-8

# # Variational Inference

# In[87]:

import matplotlib.pyplot as plot
import scipy.stats as ss
import numpy as np
import math as math


# In[88]:

"genertes posterior probability q(mu,tau|D) for a set of values of mu_N,lamba_N,a_N,b_N"
def est_posti(a_N,b_N,mew_N,l_N,mew,tau):
    mean_mew = mew_N
    if(l_N==0):
        print("l_0 is zero for esti_posti, printing mew,tau",[mew,tau])
        var_mew=1
    else:
        var_mew = 1/l_N
    p1 = ss.norm(mean_mew, var_mew).pdf(mew)
    
    rv = ss.gamma(a_N, loc=0, scale=1/b_N)
    p2= rv.pdf(tau)
    
    return p1*p2


# In[89]:

"generates true posterior probability p(mu,tau|D) for the given value of mu_0,lamba_0,a_0,b_0"
def true_posti(a_0,b_0,mew_0,l_0,mew,tau,x,N):
    mean_mew = (mew_0*l_0 + np.sum(x))/(l_0+N)
    var_mew = 1/((l_0+N)*tau)
    p1 = ss.norm(mean_mew, var_mew).pdf(mew)
    
    a_tr = a_0 + 0.5*N
    b_tr = b_0 + 0.5*l_0*mew_0*mew_0 + 0.5*np.sum(x**2)
    rv = ss.gamma(a_tr, loc=0, scale=1/b_tr)
    p2= rv.pdf(tau)
    
    return p1*p2


# In[90]:

"Used to invoke the posterior probability functions and generate zz for plotting purposes"
def posti_plot(a_0,b_0,mew_0,l_0,m,t,div,z,flag,x=np.zeros(100) ,N=100):
    if (flag==1):
        for i in range(div):
            for j in range(div):
                z[i,j]=true_posti(a_0,b_0,mew_0,l_0,m[j],t[i],x,N)
    else:
        for i in range(div):
            for j in range(div):
                z[i,j]=est_posti(a_0,b_0,mew_0,l_0,m[j],t[i])
    return z


# In[91]:

"generates the observations"
def data(N):
    D = np.random.rand(N)
    "zero mean"
    D = (D - np.mean(D))
    "unit variance"
    D = D/(np.var(D)**0.5)
    return D


# In[92]:

"updates b_N"
def update_b(b_0,x2,l_0,mew_0,N,mew_N,l_N,x):
    return (b_0 + 0.5*(x2+l_0*mew_0*mew_0) + 0.5*(N+l_0)*(mew_N*mew_N+1/l_N) - (x+l_0*mew_0)*mew_N)


# In[93]:

"for plotting contours"
div = 100
m=np.linspace(-1,1,div)
t=np.linspace(0.01,2,div)
[xx,yy] = np.meshgrid(m,t)


# In[94]:

"Number of observations"
N = 10
iter = 10
div=100

"generating observations"
D= data(N)

"setting mu_0,lamba_0,a_0,b_0"
a_0 = 0
b_0 = 0
l_0 = 0
mew_0 = 0


# In[95]:

l_N=np.zeros(iter)
b_N=np.zeros(iter)

"initial guess for mu_N,lamba_N,a_N,b_N"
a_N=5
b_N[0]=np.array([5])
l_N[0]=np.array([0.5])
mew_N=0.5

for i in range(iter):
    if(i==0):
        continue
    print("Iter: ",i)
    "update q(tau)"
    a_N = a_0 + (N+1)/2
    b_N[i] = update_b(b_0,np.sum(D**2),l_0,mew_0,N,mew_N,l_N[i-1],np.sum(D))
    
    "update q(mu)"
    mew_N = (l_0*mew_0 + N* np.mean(D))/(l_0 + N)
    if(b_N[i]==0):
        print("b is zero",i)
        l_N[i]=1
    else:
        l_N[i] = ((l_0+N)*a_N/b_N[i])
    print("Estimated [a_0,b_0,mew_0,l_0*tau] = ", [a_N,b_N[i],mew_N,l_N[i]])


# In[96]:

"Plotting true posterior"
zz_r = np.zeros((div,div),dtype=float)
zz_r = posti_plot(a_0,b_0,mew_0,l_0,m,t,div,zz_r,1,D,N)


# In[103]:

"Plotting estimated posterior distribution after the initial guess"
zz_1 = np.zeros((div,div),dtype=float)
zz_1 = posti_plot(5,b_N[0],0.5,l_N[0],m,t,div,zz_2,0)

"Plotting estimated posterior distribution after iterating once for q(tau)"
zz_2 = np.zeros((div,div),dtype=float)
zz_2 = posti_plot(a_N,b_N[1],0.5,l_N[0],m,t,div,zz_2,0)

"Plotting estimated posterior distribution after iterating once for q(tau) as well as q(mu)"
zz_3 = np.zeros((div,div),dtype=float)
zz_3 = posti_plot(a_N,b_N[1],mew_N,l_N[1],m,t,div,zz_2,0)


# In[104]:

"Plotting for estimated posterior distribution after convergence"
zz_f = np.zeros((div,div),dtype=float)
zz_f = posti_plot(a_N,b_N[iter-1],mew_N,l_N[iter-1],m,t,div,zz_f,0)


# In[105]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_1,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[106]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_2,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[107]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_3,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[108]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_f,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[ ]:




# In[ ]:




# In[ ]:



