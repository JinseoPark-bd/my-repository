#!/usr/bin/env python
# coding: utf-8

# # 3.Analytic Geometry
# ## Manhattan Norm($l_1$norm)
# ### $||x||_1 = \Sigma_{i=1}^{n} |x_i| $
# ## Euclidean Norm ($l_2$norm)
# ### $||x||_2 = \sqrt{ \Sigma_{i=1}^{n} {x_i}^2 } $

# In[27]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams[ "figure.figsize" ] = (10,10)


# In[26]:


# 1.l2norm 그리기
xRight = np.linspace(0,1,500)
xLeft = np.linspace(-1,0,500)

xarr = [xRight, xLeft, xLeft, xRight]
xarr = np.array(xarr)
xarr = xarr.reshape(-1)

yarr = [np.sqrt(1-xRight**2), np.sqrt(1-xLeft**2), -np.sqrt(1-xLeft**2), -np.sqrt(1-xRight**2)]
yarr = np.array(yarr).reshape(-1) 

plt.scatter(xarr, yarr , s=.5, color='r')
plt.show()


# In[25]:


# 1.l2norm 그리기
xRight = np.linspace(0,1,100)
xLeft = np.linspace(-1,0,100)

xarr = [xRight, xLeft, xLeft, xRight]
xarr = np.array(xarr)
xarr = xarr.reshape(-1)

yarr = [np.sqrt(1-xRight**2), np.sqrt(1-xLeft**2), -np.sqrt(1-xLeft**2), -np.sqrt(1-xRight**2)]
yarr = np.array(yarr).reshape(-1) 

plt.scatter(xarr, yarr , s=.5, color='r')

# 2.l1norm 그리기
xarr = [xRight, xLeft, xLeft, xRight]
xarr = np.array(xarr)
xarr = xarr.reshape(-1)

yarr = [1-xRight, 1+xLeft, -(1+xLeft), -(1-xRight)]
yarr = np.array(yarr).reshape(-1)

plt.scatter(xarr, yarr, s=.5, color='b')
plt.title('Manhattan Norm(L1, blue), Euclidean Norm(L2, red)')
plt.legend(["l2","l1"]) # 점 색깔 표시
plt.show()


# In[ ]:




