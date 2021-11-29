**Off-policy Q Learning**

Algorithm parameter: small $\epsilon$ >0

Initialize:#初始化

​	$\epsilon,lr\leftarrow$ an arbitrary

​	Q(s,a) $\leftarrow Zero List$, for all s $\in$ $\varsigma$,a$\in$ $\Alpha$

Repeat forever (for each episode):#训练过程
$$
\epsilon \leftarrow decay(\epsilon) \\
a=\left\{ \begin{flalign} argmax(Q(S_t))\space if \space random > \epsilon\\
random (A)\space if \space random <= \epsilon \end{flalign}\right.
$$

​	get $S_{next}$, $reward$ from $env(s,a)$

​	$Q_{predict} \leftarrow Q(s,a)$

​	$Q_{target} \leftarrow$ $reward$ +$\gamma$*$max(Q(S_{next}))$

​	$Q(s,a)\leftarrow$ $Q(s,a)$+$lr*(Q_{target} - Q_{predict})$
​					



