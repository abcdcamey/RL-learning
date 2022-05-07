**TD3**

Randomly initialize critic network $Q(s,a|\theta^Q)$ And actor $u(s|\theta^u)$ with weights $\theta^Q$ and $\theta^u$，$Q$ contains $Q_1$、$Q_2$

Initialize target network $Q'$and $u'$ With weights $\theta^{Q'} \leftarrow \theta^Q, \theta^{u'} \leftarrow \theta^u$ 

Initialize replay buffer $R$

For episode=1,M do

​	Select action $a_t=u(s_t|\theta^u)+N_t$ According to the current policy and exploration noise

​	Execute action $a_t$ and observe reward $r_t$ and observe new state $s_{t+1}$

​	Store transition $(s_t,a_t,r_t,s_{t+1})$ in $R$

​	sample a random minibatch transitions $(s_i,a_i,r_i,s_{i+1})$ from $R$

​	set $y_i = r_i+\gamma min(Q_1'(s_{t+1},u'(s_{t+1}|\theta^{u'})|\theta^{Q_1'}),Q_2'(s_{t+1},u'(s_{t+1}|\theta^{u'})|\theta^{Q_2'}))$

​	update critic by minimizing the loss:$L=\frac{1}{N}\sum_i(0.5*(y_i-Q_1(s_i,a_i|\theta^Q_1))^2+0.5*(y_i-Q_2(s_i,a_i|\theta^Q_2))^2)$

​	when policy_freq  update time:#delay update

​	update the actor policy using the sampleed policy gradient:

​		$\nabla_{\theta^u}J\approx\frac{1}{N}\sum_{i}\nabla_{a}Q_1(s,a|\theta^{Q_1})\nabla_{\theta^u}u(s|\theta^{u})$

​	update the target networks:

​		$\theta^{Q'}\leftarrow \tau\theta^Q+(1-\tau)\theta^{Q'}$

​		$\theta^{u'}\leftarrow \tau\theta^u+(1-\tau)\theta^{u'}$

​	end for

End for

 

