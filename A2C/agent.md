**A2C**

Initialize the actor parameter at random

Initialize the critic parameter at random

Use the actor to collect n step rollout trajectory:

Use the trajectory $\tau$ to estimate the  gradient $\nabla R_{\theta}$

   $\nabla R_{\theta} \leftarrow \sum_{t=0}^{n}(r_t+V(s_{t+1})-V(s_t))\nabla logp_{\theta}(a_t|s_t)$

Update the weights $\theta$  of the policy 

​	$\theta \leftarrow \theta + \alpha \nabla R_{\theta} $



















 









 





