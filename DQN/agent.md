**Off-policy DQN**

Initialize:#初始化
Initialize replay memory D with capacity N
Initialize action value function Q with random weights $\theta$

Initialize target action value function $\hat Q$ with weights $\hat \theta$

Repeat forever (for each episode):#训练过程

​	Initialize sequence $s_1$={$x_1$} and preprocessed sequence $\Phi_1$ =$\Phi(s_1)$

​	For t=1,T do:				
$$
a_t=\left\{ \begin{flalign} argmax_a(Q(\Phi(s_t),\theta)\space if \space random > \epsilon\\
random (A)\space if \space random <= \epsilon \end{flalign}\right.
$$

​		get $x_{t+1}$ ,$r_t$ from $env(s_t,a_t)$ 

​		preprocess $\Phi_{t+1}$ =$\Phi(x_{t+1})$

​		Store transition ($\Phi_{t}$,$a_t$,$r_t$, $\Phi_{t+1}$)

​		Sample random minibatch of transitions ($\Phi_{j}$,$a_j$,$r_j$, $\Phi_{j+1}$) from $D$
$$
set y_j=\left\{ \begin{flalign} r_j+\gamma max_{a^{'}}\hat Q(\Phi_{j+1},a^{'};\hat \theta) \\
r_j\space if\space episode \space terminates \space at \space step \space j+1  \end{flalign}\right.
$$
​		Perform a gradient descent step on $(y_j-Q(\Phi_j,a_j;\theta))^{2}$ With respect to the network parameters $\theta$

​		Every C step reset $\hat Q = Q$

​	End for

End for					



