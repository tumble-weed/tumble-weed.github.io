---
layout: post
title: Calculating high dimensional Hessians in low rank.
---



**Use Case**<br>

I recently had to mull over a toy sub-problem for a bigger problem that has refused to leave me be. Hessians, which are second derivatives in higher dimensions, are a cumbersome object when considered for anything neural network related. I had a scenario where I had 2 points in the input space, which I had forward passed through a neural network. 

For some loss,  whose exact form is unnecessary here, I also have the gradients at these points. A minor note, while normally you might be thinking in terms of gradients of weights of neural networks, these gradients are w.r.t. the input of the neural network. Not that it'll change the discussion, but just for the mental image. 

Now, I have the points, I have their gradients, which means that second derivatives, which are changes in the gradients, should be approximate-able. 

__Symbols__:

$H$: the Hessian of shape $D \times D$. It is a symmetric matrix , implying it has $\frac{D^2 - D}{2} + D = \frac{D(D+1)}{2}$ unique elements. We refer to the $i^{th}$ row of $H$ as $H_{i,:}$ and $j^{th}$ column as $H_{:,j}$. For a vector $v$ the $k^{th}$ element is given by $v_{j}$


$U$, $S$ , $V$ : the elements of the svd of $H$, s.t. $H = U S V^{T}$

Since the Hessian is symmetric, $U=V$.  Note the directions in $U$ are arranged in columns.


<!-- Derivative of a scalar wr.t. scalar $\frac{d (.)}{d (.)}$ -->

we have observed 2 points $x^{1}$ and $x^{2}$ as vectors with $D$ dimensions and also have their gradients $y^{1}$ and $y^{2}$.

Our system is underdetermined as the number of equations needed to find all the unknowns in $H$ arent available. in this case we will appeal to the least norm solution for $H$, i.e. an $H$ that satisfies all the requirements (as well as it can) and has the smallest frobenius norm of all such $H$

if we pack all the $y$ into $Y$ and $x$ into $X$, we have the requirement:
$Y = USU^T X$

since $U^T$ will be the inverse of $U$ due to orthogonality:

$U^T Y = SU^T X$

I consider vectors in $u$ as $u^1$, $`u^2`$ etc. Due to their orthogonality, apart from the overall equality that the equation above imposes, there is also vector wise equality (because $u^i$ cannot point in a direction pointed by $u^j$):

${u^1}^T Y = s_1{u^1}^T X$
${u^2}^T Y = s_2{u^2}^T X$

etc., where the equality is a desirable but not guaranteed to hold. for the least norm solution, we want each of the $u^i$ to pull as much weight as possible, i.e. match the left and right hand sides as well as possible. 

Also consider we have the Gram Schmidt decomposition of the concatenated $[X Y]$, denoted by $G$. We question how $u^i$ relates to this $G$. We can break apart $u^i$ into an element in the space spanned by $G$: $u_{\shortmid}$ and an element perpendicular to it $u_{\perp}$. $u_{\perp}$ has $0$ dot product with all $x$ and $y$, and is wasted mass. So, the most efficient $u^{i}$ has to lie within $G$.We denote this by:
$u^i = G \alpha_i$
where $\alpha$ is the projection vector (which is also unit by the way). if we had $n$ $(x,y)$ pairs, $\alpha$ has $2n$ elements.

We introduce this into our setup:

${u^1}^T Y = s_1{u^1}^T X$

${(G \alpha_1)}^T Y = s_1{(G \alpha_1)}^T X$

${\alpha_1}^T G^T Y = s_1{\alpha_1}^T G^T X$

${\alpha_1}^T \hat{Y} = s_1{\alpha_1}^T \hat{X}$

Here $\hat{Y}$ and $\hat{X}$ are $2n \times n$ matrices.

${\alpha_1}^T \hat{Y} = s_1{\alpha_1}^T \hat{X}$

${\alpha_1}^T \hat{Y} \hat{Y}^T = s_1{\alpha_1}^T \hat{X} \hat{Y}^T$

$` \hat{Y} \hat{Y}^T  {\alpha_1}= s_1 \hat{Y} \hat{X}^T {\alpha_1}`$

$J = \hat{Y} \hat{Y}^T$ is a symmetric matrix, of shape $2n \times 2n$. We can investigate its action by considering its SVD:  $U_J S_J U_J^T$. Because we know that $J$ is made of 2 matrices who's smaller dimension is $n$, it can have a rank of at max $n$. Implying the last $n$ singular values in $S_J$ are $0$.

We can use the SVD to analyse the possiblity of  a pseudo inverse.


$U_J S_J U_J^T  {\alpha_1}= s_1 \hat{Y} \hat{X}^T {\alpha_1}$

$S_J U_J^T  {\alpha_1}= s_1 U_J^T \hat{Y} \hat{X}^T {\alpha_1}$

$S_{J (n \mid 2n)} U_J^T  {\alpha_1}= s_1 U_J^T \hat{Y} \hat{X}^T {\alpha_1}$


Where $S_{J (n \mid 2n)}$ is a matrix with the first $n$ diagonal values possibly non-zero, and the rest as $0$. 

Since $S_{J (n \mid 2n)}$ is diagonal with final $n$ values as 0, we can construct its pseudoinverse $S_{J (n \mid 2n)}^{\dagger}$, which just contains $`1/s_{i}`$ in the first $n$ diagonal elements. 

$S_{J (n \mid 2n)}^{\dagger} S_{J (n \mid 2n)} U_J^T  {\alpha_1}= s_1 S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T {\alpha_1}$

$I_{(n \mid 2n)} U_J^T  {\alpha_1}= s_1 S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T {\alpha_1}$


$U_{J (n \mid 2n)}^T  {\alpha_1}= s_1 S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T {\alpha_1}$


$U_{J (n \mid 2n)}$ is of shape $2n \times 2n$, consisting of orthonormal vectors in the first $n$ columns, and then all $`0`$s. 

For the sake of analysis, I left multiply by $U_{J (n \mid 2n)}$:

$U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  {\alpha_1}= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T {\alpha_1}$

On the left $U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T$ is the SVD for a symmetric matrix with $U_{J (n \mid 2n)}$ as the directions and $I_{n \mid 2n}$ as the singular values. 

This implies only $\alpha_1$ that are in the space of this matrix can be represented correctly. We assume $\alpha_1 = U_\{J (n \mid 2n)\}\hat\{\alpha\}_1$, where $\hat{\alpha}_1$ is stuffed with trailing $`0`$s to be of the same shape as $\alpha_1$

---
$U_{J (n \mid 2n)}$

$U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T$

$U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1$

$U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1=$

$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1 = s_1 U_{J (n \mid 2n)}`$

$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger}`$

$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T`$

$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y}`$

$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T`$

$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}`$

$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1`$


$U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1$


$s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1$

---


$`U_{J (n \mid 2n)} U_{J (n \mid 2n)}^T  U_{J (n \mid 2n)}\hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1`$

$`U_{J (n \mid 2n)} I_{n \mid 2n} \hat{\alpha}_1= s_1 U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1`$

$`U_{J (n \mid 2n)}^T U_{J (n \mid 2n)} I_{n \mid 2n} \hat{\alpha}_1= s_1 U_{J (n \mid 2n)}^T U_{J (n \mid 2n)} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1`$

$` I_{n \mid 2n} \hat{\alpha}_1= s_1 I_{n \mid 2n} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1`$


$` \hat{\alpha}_1= s_1 I_{n \mid 2n} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1`$

$` \frac{\hat{\alpha}_1}{s_1}= I_{n \mid 2n} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}\hat{\alpha}_1`$

Which is a standard eigenvalue equation. then 

$`\hat{\alpha}_1`$ is the eigenvector of $`I_{n \mid 2n} S_{J (n \mid 2n)}^{\dagger} U_J^T \hat{Y} \hat{X}^T U_{J (n \mid 2n)}`$,  

$\frac{1}{s_1}$ is the associated eigenvalue

Once $`\hat{\alpha}_1`$ has been computed, we use it to compute $`\alpha_i`$ as $`\alpha_i = U_{J (n \mid 2n)}\hat{\alpha}_1`$ .

Using $\alpha_1$, we get $u^1$ as $u^1 = G \alpha_1$. We arrange these $u^i$ in a matrix $U_r$, and the $s_i$ in $S_r$

We reconstruct $H$ using these quantities:

$H_r = U_r S_r U_r^T$

<!-- ---

__Example__

A Hessian with rank $2$:
$H = \begin{bmatrix}
 0.68753582 & 0.77990032 & 0.67761106 \\
 0.77990032 & 0.90211256 & 0.71772732 \\
 0.67761106 & 0.71772732 & 0.81647766 \\
 \end{bmatrix}$

 $H = U S U^T$

 $H = \begin{bmatrix}
-0.54958743 & -0.19544726 & -0.81225244\\
 -0.61543158 & -0.56278256 & 0.55183309\\
 -0.56497577 & 0.80316633 & 0.1890138 \\
 \end{bmatrix}    
                \begin{bmatrix}
                2.25745730 & 0 & 0 \\
                0 & 0.148668740 & 0 \\
                0 & 0 & 0 \\
                \end{bmatrix}   \begin{bmatrix}
                                    -0.54958743 & -0.61543158 & -0.56497577 \\
                                    -0.19544726 & -0.56278256  & 0.80316633 \\
                                    -0.81225244  & 0.55183309  & 0.1890138 \\
                                \end{bmatrix}$


Measurements
$X=
\begin{bmatrix}
-2.20796738 & -0.30411266 \\
-1.40391911 & -0.54950598 \\
-0.42806159 & 0.83258512 \\
 \end{bmatrix}$
$Y=
\begin{bmatrix}
-2.90303289 & -0.07347936 \\
-3.29571902 & -0.13532473 \\
-2.85327694 & 0.07932159 \\
 \end{bmatrix}
$ -->

