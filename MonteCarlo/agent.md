**On-policy first-visit MC control**

Algorithm parameter: small $\epsilon$ >0

Initialize:#初始化

​	$\pi\leftarrow$ an arbitrary $\epsilon$-soft policy

​	Q(s,a) $\in$ $\Rho$ (arbitrary), for all s $\in$ $\varsigma$,a$\in$ $\Alpha$

​	Returns(s,a) $\leftarrow$ Empty list, for all s $\in$ $\varsigma$ , a $\in$ $\Alpha$

Repeat forever (for each episode):#训练过程

​	Generate an episode following $\pi$ : $S_0,A_0,R1,...,S_{T-1},A_{T-1},R_T$ #先根据现有参数生成一个episode

​	G$\leftarrow$ 0

​	set(s,a) in this episode

​	Loop s,a in set(s,a) and first in episode:

​		t $\leftarrow$ first (s,a) index in  episode

​		G$\leftarrow$ $R_t+\gamma R_{t+1}+...+\gamma^{T-t} R_T$

​		Append G to Returns($S_t,S_t$)

​		Q($s,a$) $\leftarrow$ average(Returns($S_t,S_t$)) 

​		$A^* \leftarrow$ Argmax$_a$Q($s,a$)

​		For all a $\in$  $\Alpha$($S_t$):
$$
\pi(a|S_t)=\left\{ 
\begin{flalign} \hfill
1-\epsilon + \epsilon/|\Alpha(S_t)|\space if\space a = A^*  \\
\epsilon/|\Alpha(S_t)|\space if a \not= A^* 
\end{flalign}
\right.
$$
​			

​		

 

 



