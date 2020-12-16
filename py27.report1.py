#!/usr/bin/env python
# coding: utf-8

# #실습 Zero
# ##벡터, 행렬연산, 그래프 그리기

# In[3]:


import numpy as np

def print_val(x):
    print "Type:", type(x)
    print "Shape:", x.shape
    print "값:\n", x
    print " "


# In[7]:


x = np.array([1,2,3])
print_val(x)

x[0] = 5
print_val(x)


# In[8]:


y = np.array([[1,2,3],[4,5,6]])
print_val(y)

a = np.zeros((2,2))
print_val(a)

a = np.ones((3,2))
print_val(a)

a = np.eye(3,3)
print_val(a)


# In[11]:


a = np.random.random((4,4))
print_val(a)

a = np.random.randn(4,4)
print_val(a)


# In[12]:


a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print_val(a)

b = a[:2,1:3]
print_val(b)

row1 = a[1,:]
print_val(row1)


# In[17]:


m1 = np.array([[1,2],[3,4]])
m2 = np.array([[5,6],[7,8]])

print_val(m1+m2)
print_val(np.add(m1,m2))

print_val(m1-m2)
print_val(np.subtract(m1,m2))

print_val(m1*m2)
print_val(np.multiply(m1,m2))

print_val(m1/m2)
print_val(np.divide(m1,m2))

print_val(np.sqrt(m1))


# In[19]:


m1 = np.array([[1,2],[3,4]])
m2 = np.array([[5,6],[7,8]])
v1 = np.array([9,10])
v2 = np.array([11,12])

print_val(m1)
print_val(m2)
print_val(v1)
print_val(v2)

print_val(v1.dot(v2))
print_val(np.dot(v1,v2))

print_val(m1.dot(v1))
print_val(np.dot(m1,v1))

print_val(m1.dot(m2))
print_val(np.dot(m1,m2))

print_val(m1)
print_val(m1.T)

print_val(np.sum(m1))
print_val(np.sum(m1,axis=0))
print_val(np.sum(m1,axis=1))


# In[20]:


n1 = np.array([[1,2,3],[4,5,6]])
print n1
print n1.shape


# In[21]:


print np.sum(n1)
print np.sum(n1,axis=0)
print np.sum(n1,axis=1)


# In[22]:


m1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
m2 = np.zeros_like(m1)
print_val(m1)
print_val(m2)


# In[23]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


x = np.arange(0,10,0.1)
y = np.sin(x)

plt.plot(x,y)


# In[27]:


y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x,y_sin)
plt.plot(x,y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('sin and cos')
plt.legend(['sin','cos'])
plt.show()


# In[29]:


plt.subplot(2,1,1)
plt.plot(x,y_sin)
plt.title('sin')

plt.subplot(2,1,2)
plt.plot(x,y_cos)
plt.title('cos')
plt.show()

