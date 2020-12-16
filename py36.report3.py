#!/usr/bin/env python
# coding: utf-8

# ## 4. Matrix Decompositions

# In[3]:


import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import PIL
from PIL import Image


# In[2]:


plt.rcParams["figure.figsize"] = (10,10)
origin2D = np.array([0,0])
origin3D = np.array([0,0,0])
scale = 10


# ### (행)벡터, 열벡터

# In[4]:


print(np.array([1,0])) #(행)벡터
print(np.hstack([1,0])) #horizontal 로 stack

print(np.vstack([1,0])) #열벡터


# ### 4.1.1 Determinant : 면적(2D)/부피(3D 이상) 측정 도구
# ##### 1) The area of the parallelogram spanned by the vectors b** and **g is |det(b,g)|.

# In[5]:


g = np.vstack([1,0])
b = np.vstack([0,1])

# determinant 계산
# 1) g,b를 column vector로 갖는 matrix A 선언
A = np.hstack([g,b])
# A = np.matrix(A)
print("A:")
print(A)
print('\n')

# 2) matrix A의 determinant 계산
print("det(A):")
print(npl.det(A))


# In[6]:


#화살표 그리기: 원점(x,y), 방향(u,v)
plt.axis([-scale/2,scale/2,-scale/2,scale/2])

plt.quiver(origin2D[0], origin2D[1], g[0], g[1], scale = scale, color = "g")
plt.quiver(origin2D[0], origin2D[1], b[0], b[1], scale = scale, color = "b")
plt.quiver(g[0], g[1], b[0], b[1], scale = scale, width = 0.005, color = "b")
plt.quiver(b[0], b[1], g[0], g[1], scale = scale, width = 0.005, color = "g")

plt.show()


# ##### 2) The volume of the paralellepiped spanned by the vectors r,b,g is |det([r,b,g])|.

# In[7]:


r = np.vstack([2,0,-8])
g = np.vstack([6,1,0])
b = np.vstack([1,4,-1])
A = np.hstack([r,g,b])
print("A:")
print(A)
print('\n')
print("det(A):")
print(npl.det(A))


# In[8]:


from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')

# Figure setup.
fig = plt.figure()
ax = plt.axes(projection="3d")
scale3D = 15
ax.set_xlim3d(-scale3D,scale3D)
ax.set_ylim3d(scale3D,-scale3D)
ax.set_zlim3d(-scale3D,scale3D)
ax.grid(b=None)

# determinant 그리기
ax.quiver(origin3D[0],origin3D[1],origin3D[2],
         A[0,0],A[1,0],A[2,0],
         color = "r", linewidths = .5, arrow_length_ratio = .05)
ax.quiver(origin3D[0],origin3D[1],origin3D[2],
         A[0,1],A[1,1],A[2,1],
         color = "g", linewidths = .5, arrow_length_ratio = .05)
ax.quiver(origin3D[0],origin3D[1],origin3D[2],
         A[0,2],A[1,2],A[2,2],
         color = "b", linewidths = .5, arrow_length_ratio = .05)

import itertools as it
quiverkey = dict(linewidths = .5, arrow_length_ratio = .05, label="_nolegend_")
c = ["r","g","b"]
# 에지 그리기 반복
for i in [i for i in list(it.product([0,1,2], repeat=2)) if i[0]!=i[1]]:
    ax.quiver(A[0,i[0]], A[1,i[0]], A[2,i[0]],
             A[0,i[1]], A[1,i[1]], A[2,i[1]],
             color = c[i[1]], **quiverkey)

ax.quiver(A[0,1]+A[0,2], A[1,1]+A[1,2], A[2,1]+A[2,2],
         A[0,0], A[1,0], A[2,0],
         color = "r", **quiverkey)
ax.quiver(A[0,2]+A[0,0], A[1,2]+A[1,0], A[2,2]+A[2,0],
         A[0,1], A[1,1], A[2,1],
         color = "g", **quiverkey)
ax.quiver(A[0,0]+A[0,1], A[1,0]+A[1,1], A[2,0]+A[2,1],
         A[0,2], A[1,2], A[2,2],
         color = "b", **quiverkey)

plt.show()


# ### 4.1.3 Trace

# In[9]:


A = np.array([[3,4,-8],
             [1,3,7],
             [6,-11,2]])
print("A:")
print(A)
print('\n')
print("Trace(A):")
print(np.trace(A))


# In[10]:


x = np.vstack([3,-1])
y = np.vstack([8,5])

print("tr(xy^T): ")
yt = np.transpose(y)
print(np.trace(x.dot(yt))) # x.dot(yt) : cross product (x,y^T)

print("x^Ty: ")
xt = np.transpose(x)
print(xt.dot(y))


# ### 4.2 Cholesky decomposition
# ##### $A = LL^T $

# In[11]:


A = np.vstack([[3,2,2],[2,3,2],[2,2,3]])
print("A:")
print(A)

print("Cholesky(A): L")
print(npl.cholesky(A))

print("L^T")
print(np.transpose(npl.cholesky(A)))


# ### 4.3 Eigendecomposition
# ##### 1) eigen values & eigen vectors of A

# In[12]:


A = np.vstack([[4,2],[1,3]])
print("A:")
print(A)
print('\n')

e_values, e_vectors = npl.eig(A)
print(e_values)
print(e_vectors)

#eigen vector u1, u2
u1 = np.vstack(e_vectors[:,0])
u2 = np.vstack(e_vectors[:,1])
print("u1: ", u1)
print("u2: ", u2)

#eigen value lambda1, lambda2
l1, l2 = e_values[0], e_values[1]
print("eigen values: ", l1, l2)
print('\n')

#Check
print("Au1: ", np.dot(A, u1))
print("l1*u1: ", l1*u1)

print("Au2: ", np.dot(A, u2))
print("l2*u2: ", l2*u2)


# ### 4.4 Singular value decomposition (SVD)
# ##### 1)Stonhenge 이미지 파일 파악

# In[13]:


stonehenge = Image.open('stonehenge.png')
print(stonehenge)
print(stonehenge.format)
print(stonehenge.size)
print(stonehenge.mode)

plt.imshow(stonehenge)
plt.show()


# ##### 2) 픽셀값 0~1 사이로 만들기

# In[14]:


# RGB -> greyscale 로 바꾸기
# numpy arra 로 바꾸기
# 0~1 사이값으로 만들기
imMatrix = np.array(stonehenge.convert("L"))/255.0
print(imMatrix.shape)
print(imMatrix)


# ##### 3) SVD 수행

# In[15]:


scalar = 1/2 #Testing reconstruction of image.
shape = np.shape(imMatrix) #(h,w)
U,S,V = npl.svd(imMatrix)

# h x h identity matrix 곱하기 singular value (635, 635)
# w-h 만큼 뒤에 0 붙이기 (635, 960)
Sd = np.hstack([np.eye(shape[0])*S.copy(),np.zeros((shape[0],shape[1]-shape[0]))])
print(np.shape(U),np.shape(Sd),np.shape(V))

# --- image reconstruction with the SVD ---
# Check 1 . 원본 영상 vs U x Sd x V
# U x Sd x V
usv = U @ Sd @ V
print(np.allclose(imMatrix, usv)) # 같으면 True

# Check 2 . 그려보기
plt.imshow(usv, cmap = 'gray')
plt.show()


# ##### 4) $A_i$ 시각화

# In[26]:


k = 1
print(np.shape(U[:,:k]))
print(np.shape(np.diag(S[:k])))
print(np.shape(V.T[:,:k].T))

m,n = np.shape(imMatrix)
partial,total = k*(m+n)+k, m*n
print(np.ndim(imMatrix),[np.shape(i) for i in [imMatrix,U,Sd,V]])
print(partial,total, partial/total)

size = (200,200)
imtemp = lambda k: (np.vstack(U[:,k-1])@np.vstack([S[k-1]])@np.vstack(V[k-1]).T)*255
for i in list(range(1,6)):
    im = Image.fromarray(imtemp(i).astype('uint8'))
    im.thumbnail(size, Image.ANTIALIAS)
    plt.imshow(im, cmap='gray')
    plt.show()


# ##### 5) $\hat A(i)$ 시각화

# In[29]:


quality = 5
np.shape(np.diag(S[:quality]))
np.shape(U[:,:quality])
np.shape(V[:quality,:])
k = quality
m,n = np.shape(imMatrix)
partial, total = k*(m+n)+k, m*n
np.ndim(imMatrix),[np.shape(i) for i in [imMatrix,U,Sd,V]]
partial,total,partial/total

imtemp = lambda k: (U[:,:k]@np.diag(S[:k])@V.T[:,:k].T)*255
for i in list(range(1,k+1)):
    im = Image.fromarray(imtemp(i).astype('uint8'))
    im.thumbnail(size, Image.ANTIALIAS)
    plt.imshow(im,cmap='gray')
    plt.show()


# ##### 6) Rank-k $\hat A(i)$ 시각화

# In[31]:


k = 30
im = imtemp(k)
m,n = np.shape(imMatrix)
partial,total = (k*(m+n)+k, m*n)
partial,total, partial/total

plt.imshow(im,cmap='gray')
plt.show()

