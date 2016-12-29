
# coding: utf-8

# # Variational Inference

# In[ ]:

import matplotlib.pyplot as plot
import scipy.stats as ss
import numpy as np
import math as math


# In[2]:

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


# In[3]:

def true_posti(a_0,b_0,mew_0,l_0,mew,tau,x,N):
    mean_mew = (mew_0*l_0 + np.sum(x))/(l_0+N)
    var_mew = 1/((l_0+N)*tau)
    p1 = ss.norm(mean_mew, var_mew).pdf(mew)
    
    a_tr = a_0 + 0.5*N
    b_tr = b_0 + 0.5*l_0*mew_0*mew_0 + 0.5*np.sum(x**2)
    rv = ss.gamma(a_tr, loc=0, scale=1/b_tr)
    p2= rv.pdf(tau)
    
    return p1*p2


# In[4]:

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


# In[42]:

def data(a_0,b_0,l_0,mew_0,N):
    tau = np.random.gamma(a_0,1/b_0)
    mew = np.random.normal(mew_0,1/(l_0*tau))
    
    D = np.random.rand(N)
    D = (D - np.mean(D))
    return [D,tau,mew]


# In[43]:

def update_b(b_0,x2,l_0,mew_0,N,mew_N,l_N,x):
    return (b_0 + 0.5*(x2+l_0*mew_0*mew_0) + 0.5*(N+l_0)*(mew_N*mew_N+1/l_N) - (x+l_0*mew_0)*mew_N)


# In[44]:

div = 100
m=np.linspace(-1,1,div)
t=np.linspace(0.01,2,div)
[xx,yy] = np.meshgrid(m,t)


# In[45]:

N = 10
iter = 10
div=100

"initial"
a_0 = 0
b_0 = 0
l_0 = 0
mew_0 = 0

"actual"
a_0_r = 5
b_0_r = 6
l_0_r = 2
mew_0_r =0
print("Actual [a_0,b_0,mew_0,l_0] = ",[a_0_r,b_0_r,mew_0_r,l_0_r])

[D,tau_r,mew_r] = data(a_0_r,b_0_r,l_0_r,mew_0_r, N)
print("Data [mew,tau] = ",[mew_r,tau_r])


# In[53]:

D = D/(2*np.var(D)**0.5)
a_0_r = 0
b_0_r = 0
l_0_r = 0
mew_0_r =0


mew_N = (l_0*mew_0 + N* np.mean(D))/(l_0 + N)
a_N = a_0 + (N+1)/2
l_N=np.zeros(iter)
b_N=np.zeros(iter)

print("Starting iter")
for i in range(iter):
    if(i==0):
        b_N[i] = update_b(b_0,np.sum(D**2),l_0,mew_0,N,mew_N,0.5,np.sum(D))
    else:
        b_N[i] = update_b(b_0,np.sum(D**2),l_0,mew_0,N,mew_N,l_N[i-1],np.sum(D))
    if(b_N[i]==0):
        print("b is zero",i)
        l_N[i]=1
    else:
        l_N[i] = ((l_0+N)*a_N/b_N[i])
    print("Estimated [a_0,b_0,mew_0,l_0*tau] = ", [a_N,b_N[i],mew_N,l_N[i]])
    
    '''[D,tau_r,mew_r] = data(a_0_r,b_0_r,l_0_r,mew_0_r, N)
    print("Data generated for ",i+2)
    print("Data [mew,tau] = ",[mew_r,tau_r])'''
    
print("Expected Answer [a_0,b_0,mew_0,tau*l_0] = ",[a_0_r,b_0_r,mew_0_r,tau_r*l_0_r])


# In[54]:

zz_r = np.zeros((div,div),dtype=float)
zz_r = posti_plot(a_0_r,b_0_r,mew_0_r,l_0_r,m,t,div,zz_r,1,D,N)
'''zz_r = zz_r/np.sum(zz_r)
zz = np.zeros([iter,div,div],dtype=float)
for i in range(iter):
    zz[i,:,:] = posti_plot(a_N,b_N[i],mew_N,l_N[i],m,t,div,zz[i,:,:],0)'''
print("done 1")
zz_f = np.zeros((div,div),dtype=float)
zz_f = posti_plot(a_N,b_N[iter-1],mew_N,l_N[iter-1],m,t,div,zz_f,0)
print("done f")


# In[55]:

zz_1 = np.zeros((div,div),dtype=float)
zz_1 = posti_plot(5,5,0.5,0.5,m,t,div,zz_1,0)
print("done 11")
zz_2 = np.zeros((div,div),dtype=float)
zz_2 = posti_plot(a_N,b_N[0],0.5,0.5,m,t,div,zz_2,0)
print("done 22")
zz_3 = np.zeros((div,div),dtype=float)
zz_3 = posti_plot(a_N,b_N[0],mew_N,l_N[0],m,t,div,zz_2,0)
print("done 33")


# In[56]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_1,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[57]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_2,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[58]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_3,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[59]:

cp1 = plot.contour(xx,yy,zz_r,colors='g')
cp2 = plot.contour(xx,yy,zz_f,colors='b')
"plot.clabel(cp1, inline=True, fontsize=5)"
plot.xlabel('$\mu')
plot.ylabel('$\tau')
plot.show()


# In[ ]:




# In[ ]:



