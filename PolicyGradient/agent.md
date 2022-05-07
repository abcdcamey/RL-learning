**Policy Gradient**

Initialize the policy parameter $\theta$ at random

Use the policy $\pi$ to collect a trajectory ${\tau = (s_0,a_0,r_1,s_1,a_1,r_2,s_2,...a_H,r_{T+1},s_{T+1}) }$

Estimate the return for trajectory $ \tau: R(\tau) = (G_0,G_1,...,G_T)$

Where $G_k$ is the expected return for transition t:

​	$G_t \leftarrow \sum_{k=t+1}^{T}\gamma^{t-k-1}R_k=r_{t+1}+\gamma G_{t+1}$



Use the trajectory $\tau$ to estimate the  gradient $\nabla R_{\theta}$

   $\nabla R_{\theta} \leftarrow \sum_{t=0}^{T}R(\tau)\nabla logp_{\theta}(a_t|s_t)$

备注：REINFORCE中，$\nabla logp_{\theta}(a_t|s_t)$貌似变成了$ylog(\hat{y})$

Update the weights $\theta$  of the policy 

​	$\theta \leftarrow \theta + \alpha \nabla R_{\theta} $

梯度的更新公式，手推版：

求$\bar R_{\theta}$的最大值，它是关于$\theta$的函数，因此可求$\bar R_{\theta}$对$\theta$的倒数，使用梯度上升求$\bar R_{\theta}$的最大值

$\bar R_{\theta} = \sum_{\tau}R(\tau)p_{\theta}(\tau) = E_{\tau \sim p_{\theta}(\tau)}[R(\tau)]$

$ \nabla \bar R_{\theta} = \sum_{\tau}R(\tau) \nabla p_{\theta}(\tau)=\sum_{\tau}R(\tau) p_{\theta}(\tau) \frac{\nabla p_{\theta}(\tau)}{p_{\theta}(\tau)}$

​	$= \sum_{\tau}R(\tau) p_{\theta}(\tau) \nabla logp_{\theta}(\tau) = E_{\tau \sim p_{\theta}(\tau)}[R(\tau)\nabla logp_{\theta}(\tau)]$

​	$\approx \frac{1}{N}\sum_{n=1}^{N}R(\tau^n) \nabla logp_{\theta}(\tau^{n}) $

​	$=\frac{1}{N}\sum_{n=1}^{N}\sum_{t=1}^{T_n} R(\tau^n) \nabla logp_{\theta}(a_t^n|s_t^n)$

$\nabla logp_{\theta}(\tau)$的计算过程:

$p_{\theta}(\tau) = p(s1)p_{\theta}(a1|s1)p(s2|s1,a1) p_{\theta}(a_2|s_2)p(s_3|s_2,a_2)...$

​		  $= p(s_1)\prod_{t=1}^{T}p_{\theta}(a_t|s_t)p(s_{t+1}|s_t,a_t)$

$\nabla logp_{\theta}(\tau) = \nabla logp(s_1)+\nabla\sum_{t=1}^{T}logp_{\theta}(a_t|s_t)+\nabla \sum_{t=1}^{T}p(s_{t+1}|s_t,a_t)$

​					$=\nabla\sum_{t=1}^{T}logp_{\theta}(a_t|s_t)=\sum_{t=1}^{T}\nabla logp_{\theta}(a_t|s_t)$















 









 





