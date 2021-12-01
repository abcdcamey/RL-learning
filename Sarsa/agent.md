**On-policy Sarsa**

Algorithm parameter: small $\epsilon$ >0

Initialize:#初始化

​	$\epsilon,lr\leftarrow$ an arbitrary

​	Q(s,a) $\leftarrow Zero List$, for all s $\in$ $\varsigma$,a$\in$ $\Alpha$

Repeat forever (for each episode):#训练过程
$$
\epsilon-greedy选择action过程 \\
$$

​	get $S_{next}$, $reward$ from $env(s,a)$

​	$Q_{predict} \leftarrow Q(s,a)$

​    $a_{next} \leftarrow \epsilon-greedy选择action过程$

​	$Q_{target} \leftarrow$ $reward$ +$\gamma$*$(Q(S_{next},a_{next}))$

​	$Q(s,a)\leftarrow$ $Q(s,a)$+$lr*(Q_{target} - Q_{predict})$

On-policy的Sarsa和Off-policy的Q-learning主要区别在于行为策略和评估策略是否一致,Off-policy的两个策略不是一致的					



