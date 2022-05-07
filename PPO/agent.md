**PPO-Clip**

Input: initial policy parameters $\theta_{0}$,initial value function parameters $\theta'_{0}$

for i in n_episode:

​	use the actor parameters $\theta_{i}$ and critic parameters $\theta'_{i}$ to collect trajectory $\tau$

​	$\tau=(s_{0},a_{0},dist_{0},value_{0},logprob_{0},r_{0},...,s_{T},a_{T},dist_{T},value_{T},logprob_{T},r_{T},s_{T+1})$

​	update the actor parameters $\theta_{i}$ and critic parameters $\theta'_{i}$ in n steps:
​		get $advantage_{t}$:

​			$\sum_{k=t}^{T}discount_{k-t}(r_{k}+\gamma v_{k+1}-v_{k})$

​		get $newlogprob$ From lastest actor

​		get $new critic value$ from lastest critic

​		compute $probratio = newprob/oldprob$

​		$weightedprobs = advantage*probratio$

​		$weightedclipprobs = advantage*clip(probratio)$

​		$actorloss = -min(weightedprobs,weightedclipprobs)$

​		$ returns = advantage+value$

​		$criticloss = (returns-new critic value)^2$

​		$totalloss=actorloss+0.5*(criticloss)$





​	

 

​	



