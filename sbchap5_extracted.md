## **Chapter 5** 

## **Monte Carlo Methods** 

In this chapter we consider our first learning methods for estimating value functions and discovering optimal policies. Unlike the previous chapter, here we do not assume complete knowledge of the environment. Monte Carlo methods require only _experience_ —sample sequences of states, actions, and rewards from actual or simulated interaction with an environment. Learning from _actual_ experience is striking because it requires no prior knowledge of the environment’s dynamics, yet can still attain optimal behavior. Learning from _simulated_ experience is also powerful. Although a model is required, the model need only generate sample transitions, not the complete probability distributions of all possible transitions that is required for dynamic programming (DP). In surprisingly many cases it is easy to generate experience sampled according to the desired probability distributions, but infeasible to obtain the distributions in explicit form. 

Monte Carlo methods are ways of solving the reinforcement learning problem based on averaging sample returns. To ensure that well-defined returns are available, here we define Monte Carlo methods only for episodic tasks. That is, we assume experience is divided into episodes, and that all episodes eventually terminate no matter what actions are selected. Only on the completion of an episode are value estimates and policies changed. Monte Carlo methods can thus be incremental in an episode-by-episode sense, but not in a step-by-step (online) sense. The term “Monte Carlo” is often used more broadly for any estimation method whose operation involves a significant random component. Here we use it specifically for methods based on averaging complete returns (as opposed to methods that learn from partial returns, considered in the next chapter). 

Monte Carlo methods sample and average _returns_ for each state–action pair much like the bandit methods we explored in Chapter 2 sample and average 

_rewards_ for each action. The main di↵erence is that now there are multiple states, each acting like a di↵erent bandit problem (like an associative-search or contextual bandit) and that the di↵erent bandit problems are interrelated. That is, the return after taking an action in one state depends on the actions taken in later states in the same episode. Because all the action selections are undergoing learning, the problem becomes nonstationary from the point of view of the earlier state. 

To handle the nonstationarity, we adapt the idea of general policy iteration (GPI) developed in Chapter 4 for DP. Whereas there we _computed_ value functions from knowledge of the MDP, here we _learn_ value functions from sample returns with the MDP. The value functions and corresponding policies still interact to attain optimality in essentially the same way (GPI). As in the DP chapter, first we consider the prediction problem (the computation of _v⇡_ and _q⇡_ for a fixed arbitrary policy _⇡_ ) then policy improvement, and, finally, the control problem and its solution by GPI. Each of these ideas taken from DP is extended to the Monte Carlo case in which only sample experience is available. 

## **5.1 Monte Carlo Prediction** 

We begin by considering Monte Carlo methods for learning the state-value function for a given policy. Recall that the value of a state is the expected return—expected cumulative future discounted reward—starting from that state. An obvious way to estimate it from experience, then, is simply to average the returns observed after visits to that state. As more returns are observed, the average should converge to the expected value. This idea underlies all Monte Carlo methods. 

In particular, suppose we wish to estimate _v⇡_ ( _s_ ), the value of a state _s_ under policy _⇡_ , given a set of episodes obtained by following _⇡_ and passing through _s_ . Each occurrence of state _s_ in an episode is called a _visit_ to _s_ . Of course, _s_ may be visited multiple times in the same episode; let us call the first time it is visited in an episode the _first visit_ to _s_ . The _first-visit MC method_ estimates _v⇡_ ( _s_ ) as the average of the returns following first visits to _s_ , whereas the _every-visit MC method_ averages the returns following all visits to _s_ . These two Monte Carlo (MC) methods are very similar but have slightly di↵erent theoretical properties. First-visit MC has been most widely studied, dating back to the 1940s, and is the one we focus on in this chapter. Every-visit MC extends more naturally to function approximation and eligibility traces, as discussed in Chapters 9 and 7. First-visit MC is shown in procedural form in Figure 5.1. 

Initialize: _⇡_ policy to be evaluated _V_ an arbitrary state-value function _Returns_ ( _s_ ) an empty list, for all _s 2_ S Repeat forever: Generate an episode using _⇡_ For each state _s_ appearing in the episode: _G_ return following the first occurrence of _s_ Append _G_ to _Returns_ ( _s_ ) _V_ ( _s_ ) average( _Returns_ ( _s_ )) 

Figure 5.1: The first-visit MC method for estimating _v⇡_ . Note that we use a capital letter _V_ for the approximate value function because, after initialization, it soon becomes a random variable. 

Both first-visit MC and every-visit MC converge to _v⇡_ ( _s_ ) as the number of visits (or first visits) to _s_ goes to infinity. This is easy to see for the case of first-visit MC. In this case each return is an independent, identically distributed estimate of _v⇡_ ( _s_ ) with finite variance. By the law of large numbers the sequence of averages of these estimates converges to their expected value. Each average is itself an unbiased estimate, and the standard deviation of its error falls as 1 _/[p]_ ~~_n_ ,~~ where _n_ is the number of returns averaged. Every-visit MC is less straightforward, but its estimates also converge asymptotically to _v⇡_ ( _s_ ) (Singh and Sutton, 1996). 

The use of Monte Carlo methods is best illustrated through an example. 

**Example 5.1: Blackjack** The object of the popular casino card game of _blackjack_ is to obtain cards the sum of whose numerical values is as great as possible without exceeding 21. All face cards count as 10, and an ace can count either as 1 or as 11. We consider the version in which each player competes independently against the dealer. The game begins with two cards dealt to both dealer and player. One of the dealer’s cards is face up and the other is face down. If the player has 21 immediately (an ace and a 10-card), it is called a _natural_ . He then wins unless the dealer also has a natural, in which case the game is a draw. If the player does not have a natural, then he can request additional cards, one by one ( _hits_ ), until he either stops ( _sticks_ ) or exceeds 21 ( _goes bust_ ). If he goes bust, he loses; if he sticks, then it becomes the dealer’s turn. The dealer hits or sticks according to a fixed strategy without choice: he sticks on any sum of 17 or greater, and hits otherwise. If the dealer goes bust, then the player wins; otherwise, the outcome—win, lose, or draw—is determined by whose final sum is closer to 21. 


![](data/sbchap5.pdf-0004-02.png)


**----- Start of picture text -----**<br>
After 10,000 episodes After 500,000 episodes<br>Usable +1<br>ace<br>!1<br>No<br>usable<br>ace<br>Dealer showing<br>A<br>10<br>12<br>Player sum<br>21<br>**----- End of picture text -----**<br>


Figure 5.2: Approximate state-value functions for the blackjack policy that sticks only on 20 or 21, computed by Monte Carlo policy evaluation. 

Playing blackjack is naturally formulated as an episodic finite MDP. Each game of blackjack is an episode. Rewards of +1, _−_ 1, and 0 are given for winning, losing, and drawing, respectively. All rewards within a game are zero, and we do not discount ( _γ_ = 1); therefore these terminal rewards are also the returns. The player’s actions are to hit or to stick. The states depend on the player’s cards and the dealer’s showing card. We assume that cards are dealt from an infinite deck (i.e., with replacement) so that there is no advantage to keeping track of the cards already dealt. If the player holds an ace that he could count as 11 without going bust, then the ace is said to be _usable_ . In this case it is always counted as 11 because counting it as 1 would make the sum 11 or less, in which case there is no decision to be made because, obviously, the player should always hit. Thus, the player makes decisions on the basis of three variables: his current sum (12–21), the dealer’s one showing card (ace–10), and whether or not he holds a usable ace. This makes for a total of 200 states. 

Consider the policy that sticks if the player’s sum is 20 or 21, and otherwise hits. To find the state-value function for this policy by a Monte Carlo approach, one simulates many blackjack games using the policy and averages the returns following each state. Note that in this task the same state never recurs within one episode, so there is no di↵erence between first-visit and every-visit MC methods. In this way, we obtained the estimates of the statevalue function shown in Figure 5.2. The estimates for states with a usable ace are less certain and less regular because these states are less common. In any event, after 500,000 games the value function is very well approximated. 

## _5.1. MONTE CARLO PREDICTION_ 

Although we have complete knowledge of the environment in this task, it would not be easy to apply DP methods to compute the value function. DP methods require the distribution of next events—in particular, they require the quantities _p_ ( _s[0] , r|s, a_ )—and it is not easy to determine these for blackjack. For example, suppose the player’s sum is 14 and he chooses to stick. What is his expected reward as a function of the dealer’s showing card? All of these expected rewards and transition probabilities must be computed _before_ DP can be applied, and such computations are often complex and error-prone. In contrast, generating the sample games required by Monte Carlo methods is easy. This is the case surprisingly often; the ability of Monte Carlo methods to work with sample episodes alone can be a significant advantage even when one has complete knowledge of the environment’s dynamics. 

Can we generalize the idea of backup diagrams to Monte Carlo algorithms? The general idea of a backup diagram is to show at the top the root node to be updated and to show below all the transitions and leaf nodes whose rewards and estimated values contribute to the update. For Monte Carlo estimation of _v⇡_ , the root is a state node, and below it is the entire trajectory of transitions along a particular single episode, ending at the terminal state, as in Figure 5.3. Whereas the DP diagram (Figure 3.4a) shows all possible transitions, the Monte Carlo diagram shows only those sampled on the one episode. Whereas the DP diagram includes only one-step transitions, the Monte Carlo diagram goes all the way to the end of the episode. These di↵erences in the diagrams accurately reflect the fundamental di↵erences between the algorithms. 

An important fact about Monte Carlo methods is that the estimates for each state are independent. The estimate for one state does not build upon the estimate of any other state, as is the case in DP. In other words, Monte Carlo methods do not _bootstrap_ as we defined it in the previous chapter. 

In particular, note that the computational expense of estimating the value of a single state is independent of the number of states. This can make Monte Carlo methods particularly attractive when one requires the value of only one or a subset of states. One can generate many sample episodes starting from the states of interest, averaging returns from only these states ignoring all others. This is a third advantage Monte Carlo methods can have over DP methods (after the ability to learn from actual experience and from simulated experience). 

terminal state 

Figure 5.3: The backup diagram for Monte Carlo estimation of _v⇡_ . 

## **Example 5.2: Soap Bubble** 

Suppose a wire frame forming a closed loop is dunked in soapy water to form a soap surface or bubble conforming at its edges to the wire frame. If the geometry of the wire frame is irregular but known, how can you compute the shape of the surface? The shape has the property that the total force on each point exerted by neighborA bubble on a wire loop ing points is zero (or else the shape would change). This means that the surface’s height at any point is the average of its heights at points in a small circle around that point. In addition, the surface must meet at its boundaries with the wire frame. The usual approach to problems of this kind is to put a grid over the area covered by the surface and solve for its height at the grid points by an iterative computation. Grid points at the boundary are forced to the wire frame, and all others are adjusted toward the average of the heights of their four nearest neighbors. This process then iterates, much like DP’s iterative policy evaluation, and ultimately converges to a close approximation to the desired surface. 

This is similar to the kind of problem for which Monte Carlo methods were originally designed. Instead of the iterative computation described above, imagine standing on the surface and taking a random walk, stepping randomly from grid point to neighboring grid point, with equal probability, until you 

reach the boundary. It turns out that the expected value of the height at the boundary is a close approximation to the height of the desired surface at the starting point (in fact, it is exactly the value computed by the iterative method described above). Thus, one can closely approximate the height of the surface at a point by simply averaging the boundary heights of many walks started at the point. If one is interested in only the value at one point, or any fixed small set of points, then this Monte Carlo method can be far more efficient than the iterative method based on local consistency. 

## **5.2 Monte Carlo Estimation of Action Values** 

If a model is not available, then it is particularly useful to estimate _action_ values (the values of state–action pairs) rather than _state_ values. With a model, state values alone are sufficient to determine a policy; one simply looks ahead one step and chooses whichever action leads to the best combination of reward and next state, as we did in the chapter on DP. Without a model, however, state values alone are not sufficient. One must explicitly estimate the value of each action in order for the values to be useful in suggesting a policy. Thus, one of our primary goals for Monte Carlo methods is to estimate _q⇤_ . To achieve this, we first consider the policy evaluation problem for action values. 

The policy evaluation problem for action values is to estimate _q⇡_ ( _s, a_ ), the expected return when starting in state _s_ , taking action _a_ , and thereafter following policy _⇡_ . The Monte Carlo methods for this are essentially the same as just presented for state values, except now we talk about visits to a state– action pair rather than to a state. A state–action pair _s, a_ is said to be visited in an episode if ever the state _s_ is visited and action _a_ is taken in it. The everyvisit MC method estimates the value of a state–action pair as the average of the returns that have followed visits all the visits to it. The first-visit MC method averages the returns following the first time in each episode that the state was visited and the action was selected. These methods converge quadratically, as before, to the true expected values as the number of visits to each state–action pair approaches infinity. 

The only complication is that many state–action pairs may never be visited. If _⇡_ is a deterministic policy, then in following _⇡_ one will observe returns only for one of the actions from each state. With no returns to average, the Monte Carlo estimates of the other actions will not improve with experience. This is a serious problem because the purpose of learning action values is to help in choosing among the actions available in each state. To compare alternatives we need to estimate the value of _all_ the actions from each state, not just the one we currently favor. 

This is the general problem of _maintaining exploration_ , as discussed in the context of the _n_ -armed bandit problem in Chapter 2. For policy evaluation to work for action values, we must assure continual exploration. One way to do this is by specifying that the episodes _start in a state–action pair_ , and that every pair has a nonzero probability of being selected as the start. This guarantees that all state–action pairs will be visited an infinite number of times in the limit of an infinite number of episodes. We call this the assumption of _exploring starts_ . 

The assumption of exploring starts is sometimes useful, but of course it cannot be relied upon in general, particularly when learning directly from actual interaction with an environment. In that case the starting conditions are unlikely to be so helpful. The most common alternative approach to assuring that all state–action pairs are encountered is to consider only policies that are stochastic with a nonzero probability of selecting all actions in each state. We discuss two important variants of this approach in later sections. For now, we retain the assumption of exploring starts and complete the presentation of a full Monte Carlo control method. 

## **5.3 Monte Carlo Control** 

We are now ready to consider how Monte Carlo estimation can be used in control, that is, to approximate optimal policies. The overall idea is to proceed according to the same pattern as in the DP chapter, that is, according to the idea of generalized policy iteration (GPI). In GPI one maintains both an approximate policy and an approximate value function. The value function is repeatedly altered to more closely approximate the value function for the current policy, and the policy is repeatedly improved with respect to the current value function: 


![](data/sbchap5.pdf-0008-06.png)


**----- Start of picture text -----**<br>
evaluation<br>q →  q π<br>π q<br>π →greedy( q )<br>improvement<br>**----- End of picture text -----**<br>


These two kinds of changes work against each other to some extent, as each creates a moving target for the other, but together they cause both policy and 

value function to approach optimality. 

To begin, let us consider a Monte Carlo version of classical policy iteration. In this method, we perform alternating complete steps of policy evaluation and policy improvement, beginning with an arbitrary policy _⇡_ 0 and ending with the optimal policy and optimal action-value function: 


![](data/sbchap5.pdf-0009-04.png)


where _−!_ E denotes a complete policy evaluation and _−!_ I denotes a complete policy improvement. Policy evaluation is done exactly as described in the preceding section. Many episodes are experienced, with the approximate actionvalue function approaching the true function asymptotically. For the moment, let us assume that we do indeed observe an infinite number of episodes and that, in addition, the episodes are generated with exploring starts. Under these assumptions, the Monte Carlo methods will compute each _q⇡k_ exactly, for arbitrary _⇡k_ . 

Policy improvement is done by making the policy greedy with respect to the current value function. In this case we have an _action_ -value function, and therefore no model is needed to construct the greedy policy. For any actionvalue function _q_ , the corresponding greedy policy is the one that, for each _s 2_ S, deterministically chooses an action with maximal action-value: 


![](data/sbchap5.pdf-0009-07.png)


Policy improvement then can be done by constructing each _⇡k_ +1 as the greedy policy with respect to _q⇡k_ . The policy improvement theorem (Section 4.2) then applies to _⇡k_ and _⇡k_ +1 because, for all _s 2_ S, 


![](data/sbchap5.pdf-0009-09.png)


As we discussed in the previous chapter, the theorem assures us that each _⇡k_ +1 is uniformly better than _⇡k_ , or just as good as _⇡k_ , in which case they are both optimal policies. This in turn assures us that the overall process converges to the optimal policy and optimal value function. In this way Monte Carlo methods can be used to find optimal policies given only sample episodes and no other knowledge of the environment’s dynamics. 

We made two unlikely assumptions above in order to easily obtain this guarantee of convergence for the Monte Carlo method. One was that the 

episodes have exploring starts, and the other was that policy evaluation could be done with an infinite number of episodes. To obtain a practical algorithm we will have to remove both assumptions. We postpone consideration of the first assumption until later in this chapter. 

For now we focus on the assumption that policy evaluation operates on an infinite number of episodes. This assumption is relatively easy to remove. In fact, the same issue arises even in classical DP methods such as iterative policy evaluation, which also converge only asymptotically to the true value function. In both DP and Monte Carlo cases there are two ways to solve the problem. One is to hold firm to the idea of approximating _q⇡k_ in each policy evaluation. Measurements and assumptions are made to obtain bounds on the magnitude and probability of error in the estimates, and then sufficient steps are taken during each policy evaluation to assure that these bounds are sufficiently small. This approach can probably be made completely satisfactory in the sense of guaranteeing correct convergence up to some level of approximation. However, it is also likely to require far too many episodes to be useful in practice on any but the smallest problems. 

The second approach to avoiding the infinite number of episodes nominally required for policy evaluation is to forgo trying to complete policy evaluation before returning to policy improvement. On each evaluation step we move the value function _toward q⇡k_ , but we do not expect to actually get close except over many steps. We used this idea when we first introduced the idea of GPI in Section 4.6. One extreme form of the idea is value iteration, in which only one iteration of iterative policy evaluation is performed between each step of policy improvement. The in-place version of value iteration is even more extreme; there we alternate between improvement and evaluation steps for single states. 

For Monte Carlo policy evaluation it is natural to alternate between evaluation and improvement on an episode-by-episode basis. After each episode, the observed returns are used for policy evaluation, and then the policy is improved at all the states visited in the episode. A complete simple algorithm along these lines is given in Figure 5.4. We call this algorithm _Monte Carlo ES_ , for Monte Carlo with Exploring Starts. 

In Monte Carlo ES, all the returns for each state–action pair are accumulated and averaged, irrespective of what policy was in force when they were observed. It is easy to see that Monte Carlo ES cannot converge to any suboptimal policy. If it did, then the value function would eventually converge to the value function for that policy, and that in turn would cause the policy to change. Stability is achieved only when both the policy and the value function are optimal. Convergence to this optimal fixed point seems inevitable as the changes to the action-value function decrease over time, but has not yet 


![](data/sbchap5.pdf-0011-02.png)


**----- Start of picture text -----**<br>
Initialize, for all  s 2  S,  a 2  A( s ):<br>Q ( s, a ) arbitrary<br>⇡ ( s ) arbitrary<br>Returns ( s, a ) empty list<br>Repeat forever:<br>Choose  S 0  2  S and  A 0  2  A( S 0) s.t. all pairs have probability  >  0<br>Generate an episode starting from  S 0 , A 0, following  ⇡<br>For each pair  s, a  appearing in the episode:<br>G return following the first occurrence of  s, a<br>Append  G  to  Returns ( s, a )<br>Q ( s, a ) average( Returns ( s, a ))<br>For each  s  in the episode:<br>⇡ ( s ) argmax a Q ( s, a )<br>**----- End of picture text -----**<br>


Figure 5.4: Monte Carlo ES: A Monte Carlo control algorithm assuming exploring starts and that episodes always terminate for all policies. 

been formally proved. In our opinion, this is one of the most fundamental open theoretical questions in reinforcement learning (for a partial solution, see Tsitsiklis, 2002). 

**Example 5.3: Solving Blackjack** It is straightforward to apply Monte Carlo ES to blackjack. Since the episodes are all simulated games, it is easy to arrange for exploring starts that include all possibilities. In this case one simply picks the dealer’s cards, the player’s sum, and whether or not the player has a usable ace, all at random with equal probability. As the initial policy we use the policy evaluated in the previous blackjack example, that which sticks only on 20 or 21. The initial action-value function can be zero for all state–action pairs. Figure 5.5 shows the optimal policy for blackjack found by Monte Carlo ES. This policy is the same as the “basic” strategy of Thorp (1966) with the sole exception of the leftmost notch in the policy for a usable ace, which is not present in Thorp’s strategy. We are uncertain of the reason for this discrepancy, but confident that what is shown here is indeed the optimal policy for the version of blackjack we have described. 

## _CHAPTER 5. MONTE CARLO METHODS_ 

124 


![](data/sbchap5.pdf-0012-19.png)


**----- Start of picture text -----**<br>
! [*] * Vv * [*] [*]<br>21<br>STICK 20<br>19<br>Usable 18 +1<br>17<br>ace 16<br>15 "1<br>HIT 14<br>13<br>12<br>11<br>A 2 3 4 5 6 7 8 9 10<br>21<br>20<br>STICK 19<br>No 1817 +1<br>usable 16<br>15 "1<br>ace HIT 14<br>13<br>12<br>11<br>A 2 3 4 5 6 7 8 9 10<br>Dealer showing Dealer showing<br>Dealer showing<br>Dealer showing<br>AA<br>AA<br>0<br>1010<br>1010<br>1212<br>1212<br>2121<br>2121<br>Player sum<br>Player sum<br>Player sum<br>Player sum<br>**----- End of picture text -----**<br>


Figure 5.5: The optimal policy and state-value function for blackjack, found by Monte Carlo ES (Figure 5.4). The state-value function shown was computed from the action-value function found by Monte Carlo ES. 

## 

## **5.4 Monte Carlo Control without Exploring Starts** 

How can we avoid the unlikely assumption of exploring starts? The only general way to ensure that all actions are selected infinitely often is for the agent to continue to select them. There are two approaches to ensuring this, resulting in what we call _on-policy_ methods and _o↵-policy_ methods. Onpolicy methods attempt to evaluate or improve the policy that is used to make decisions, whereas o↵-policy methods evaluate or improve a policy di↵erent from that used to generate the data. The Monte Carlo ES method developed above is an example of an on-policy method. In this section we show how an on-policy Monte Carlo control method can be designed that does not use the unrealistic assumption of exploring starts. O↵-policy methods are considered in the next section. 

In on-policy control methods the policy is generally _soft_ , meaning that _⇡_ ( _a|s_ ) _>_ 0 for all _s 2_ S and all _a 2_ A( _s_ ), but gradually shifted closer and closer to a deterministic optimal policy. Many of the methods discussed in Chapter 2 provide mechanisms for this. The on-policy method we present in this section uses _"-greedy_ policies, meaning that most of the time they choose an action that has maximal estimated action value, but with probability _"_ 

## _5.4. MONTE CARLO CONTROL WITHOUT EXPLORING STARTS_ 125 

they instead select an action at random. That is, all nongreedy actions are given the minimal probability of selection, _|_ A( _✏s_ ) _|_[, and the remaining bulk of the] _✏_ probability, 1 _− "_ + _|_ A( _s_ ) _|_[, is given to the greedy action. The] _[ "]_[-greedy policies] are examples of _"-soft_ policies, defined as policies for which _⇡_ ( _a|s_ ) _≥ |_ A( _✏s_ ) _|_[for] all states and actions, for some _" >_ 0. Among _"_ -soft policies, _"_ -greedy policies are in some sense those that are closest to greedy. 

The overall idea of on-policy Monte Carlo control is still that of GPI. As in Monte Carlo ES, we use first-visit MC methods to estimate the action-value function for the current policy. Without the assumption of exploring starts, however, we cannot simply improve the policy by making it greedy with respect to the current value function, because that would prevent further exploration of nongreedy actions. Fortunately, GPI does not require that the policy be taken all the way to a greedy policy, only that it be moved _toward_ a greedy policy. In our on-policy method we will move it only to an _"_ -greedy policy. For any _"_ -soft policy, _⇡_ , any _"_ -greedy policy with respect to _q⇡_ is guaranteed to be better than or equal to _⇡_ . 

That any _"_ -greedy policy with respect to _q⇡_ is an improvement over any _"_ -soft policy _⇡_ is assured by the policy improvement theorem. Let _⇡[0]_ be the _"_ -greedy policy. The conditions of the policy improvement theorem apply because for any _s 2_ S: 


![](data/sbchap5.pdf-0013-04.png)



![](data/sbchap5.pdf-0013-05.png)


Thus, by the policy improvement theorem, _⇡[0] ≥ ⇡_ (i.e., _v⇡0_ ( _s_ ) _≥ v⇡_ ( _s_ ), for all _s 2_ S). We now prove that equality can hold only when both _⇡[0]_ and _⇡_ are optimal among the _"_ -soft policies, that is, when they are better than or equal to all other _"_ -soft policies. 

Consider a new environment that is just like the original environment, except with the requirement that policies be _"_ -soft “moved inside” the environment. The new environment has the same action and state set as the original 

and behaves as follows. If in state _s_ and taking action _a_ , then with probability 1 _− "_ the new environment behaves exactly like the old environment. With probability _"_ it repicks the action at random, with equal probabilities, and then behaves like the old environment with the new, random action. The best one can do in this new environment with general policies is the same as the best one could do in the original environment with _"_ -soft policies. Let e _v⇤_ and e _q⇤_ denote the optimal value functions for the new environment. Then a policy _⇡_ is optimal among _"_ -soft policies if and only if _v⇡_ = e _v⇤_ . From the definition of e _v⇤_ we know that it is the unique solution to 


![](data/sbchap5.pdf-0014-03.png)


When equality holds and the _"_ -soft policy _⇡_ is no longer improved, then we also know, from (5.2), that 


![](data/sbchap5.pdf-0014-05.png)


However, this equation is the same as the previous one, except for the substitution of _v⇡_ for e _v⇤_ . Since e _v⇤_ is the unique solution, it must be that _v⇡_ = e _v⇤_ . 

In essence, we have shown in the last few pages that policy iteration works for _"_ -soft policies. Using the natural notion of greedy policy for _"_ -soft policies, one is assured of improvement on every step, except when the best policy has been found among the _"_ -soft policies. This analysis is independent of how the action-value functions are determined at each stage, but it does assume that they are computed exactly. This brings us to roughly the same point as in the previous section. Now we only achieve the best policy among the _"_ -soft policies, but on the other hand, we have eliminated the assumption of exploring starts. The complete algorithm is given in Figure 5.6. 


![](data/sbchap5.pdf-0015-02.png)


**----- Start of picture text -----**<br>
Initialize, for all  s 2  S,  a 2  A( s ):<br>Q ( s, a ) arbitrary<br>Returns ( s, a ) empty list<br>⇡ ( a|s ) an arbitrary  " -soft policy<br>Repeat forever:<br>(a) Generate an episode using  ⇡<br>(b) For each pair  s, a  appearing in the episode:<br>G return following the first occurrence of  s, a<br>Append  G  to  Returns ( s, a )<br>Q ( s, a ) average( Returns ( s, a ))<br>(c) For each  s  in the episode:<br>a [⇤] arg max a Q ( s, a )<br>For all  a 2  A( s ):<br>1  − "  +  "/| A( s ) | if  a  =  a [⇤]<br>⇡ ( a|s ) ⇢ "/| A( s ) | if  a 6 =  a [⇤]<br>**----- End of picture text -----**<br>


Figure 5.6: An on-policy first-visit MC control algorithm for _"_ -soft policies. 

## **5.5 O↵-policy Prediction via Importance Sampling** 

So far we have considered methods for estimating the value functions for a policy given an infinite supply of episodes generated using that policy. Suppose now that all we have are episodes generated from a _di↵erent_ policy. That is, suppose we wish to estimate _v⇡_ or _q⇡_ , but all we have are episodes following another policy _µ_ , where _µ 6_ = _⇡_ . We call _⇡_ the _target policy_ because learning its value function is the target of the learning process, and we call _µ_ the _behavior policy_ because it is the policy controlling the agent and generating behavior. The overall problem is called _o↵-policy learning_ because it is learning about a policy given only experience “o↵” (not following) that policy. 

In order to use episodes from _µ_ to estimate values for _⇡_ , we must require that every action taken under _⇡_ is also taken, at least occasionally, under _µ_ . That is, we require that _⇡_ ( _a|s_ ) _>_ 0 implies _µ_ ( _a|s_ ) _>_ 0. This is called the assumption of _coverage_ . It follows from coverage that _µ_ must be stochastic in states where it is not identical to _⇡_ . The target policy _⇡_ , on the other hand, may be deterministic, and, in fact, this is a case of particular interest. Typically the target policy is the deterministic greedy policy with respect to the current action-value function estimate. This policy we hope becomes a deterministic optimal policy while the behavior policy remains stochastic and more exploratory, for example, an _"_ -greedy policy. 

Importance sampling is a general technique for estimating expected values under one distribution given samples from another. We apply this technique to o↵-policy learning by weighting returns according to the relative probability of their trajectories occurring under the target and behavior policies, called the _importance-sampling ratio_ . Given a starting state _St_ , the probability of the subsequent state–action trajectory, _At, St_ +1 _, At_ +1 _, . . . , ST_ , occurring under any policy _⇡_ is 


![](data/sbchap5.pdf-0016-03.png)


where _p_ is the state-transition probability function defined by (3.8). Thus, the relative probability of the trajectory under the target and behavior policies (the importance-sampling ratio) is 

Q _T −_ 1 


![](data/sbchap5.pdf-0016-06.png)


Note that although the trajectory probabilities depend on the MDP’s transition probabilities, which are generally unknown, all the transition probabilities cancel and drop out. The importance sampling ratio ends up depending only on the two policies and not at all on the MDP. 

Now we are ready to give a Monte Carto algorithm that uses a batch of observed episodes following policy _µ_ to estimate _v⇡_ ( _s_ ). It is convenient here to number time steps in a way that increases across episode boundaries. That is, if the first episode of the batch ends in a terminal state at time 100, then the next episode begins at time _t_ = 101. This enables us to use time-step numbers to refer to particular steps in particular episodes. In particular, we can define the set of all time steps in which state _s_ is visited, denoted T( _s_ ). This is for an every-visit method; for a first-visit method, T( _s_ ) would only include time steps that were first visits to _s_ within their episode. Also, let _T_ ( _t_ ) denote the first time of termination following time _t_ , and _Gt_ denote the return after _t_ up through _T_ ( _t_ ). Then _{Gt}t2_ T( _s_ ) are the returns that pertain to state _s_ , and _{⇢[T] t_[(] _[t]_[)] _}t2_ T( _s_ ) are the corresponding importance-sampling ratios. To estimate _v⇡_ ( _s_ ), we simply scale the returns by the ratios and average the results: 


![](data/sbchap5.pdf-0016-09.png)



![](data/sbchap5.pdf-0016-10.png)


When importance sampling is done as a simple average in this way it is called _ordinary importance sampling_ . 

An important alternative is _weighted importance sampling_ , which uses a _weighted_ average, defined as 


![](data/sbchap5.pdf-0017-02.png)


or zero if the denominator is zero. To understand these two varieties of importance sampling, consider their estimates after observing a single return. In the weighted-average estimate, the ratio _⇢[T] t_[(] _[t]_[)] for the single return cancels in the numerator and denominator, so that the estimate is equal to the observed return independent of the ratio (assuming the ratio is nonzero). Given that this return was the only one observed, this is a reasonable estimate, but of course its expectation is _vµ_ ( _s_ ) rather than _v⇡_ ( _s_ ), and in this statistical sense it is biased. In contrast, the simple average (5.4) is always _v⇡_ ( _s_ ) in expectation (it is unbiased), but it can be extreme. Suppose the ratio were ten, indicating that the trajectory observed is ten times as likely under the target policy as under the behavior policy. In this case the ordinary importance-sampling estimate would be _ten times_ the observed return. That is, it would be quite far from the observed return even though the episode’s trajectory is considered very representative of the target policy. 

Formally, the di↵erence between the two kinds of importance sampling is expressed in their variances. The variance of the ordinary importancesampling estimator is in general unbounded because the variance of the ratios is unbounded, whereas in the weighted estimator the largest weight on any single return is one. In fact, assuming bounded returns, the variance of the weighted importance-sampling estimator converges to zero even if the variance of the ratios themselves is infinite (Precup, Sutton, and Dasgupta 2001). In practice, the weighted estimator usually has dramatically lower variance and is strongly preferred. A complete every-visit MC algorithm for o↵-policy policy evaluation using weighted importance sampling is given at the end of the next section in Figure 5.9. 

**Example 5.4: O↵-policy Estimation of a Blackjack State Value** We applied both ordinary and weighted importance-sampling methods to estimate the value of a single blackjack state from o↵-policy data. Recall that one of the advantages of Monte Carlo methods is that they can be used to evaluate a single state without forming estimates for any other states. In this example, we evaluated the state in which the dealer is showing a deuce, the sum of the player’s cards is 13, and the player has a usable ace (that is, the player holds an ace and a deuce, or equivalently three aces). The data was generated by starting in this state then choosing to hit or stick at random with equal probability (the behavior policy). The target policy was to stick only on a sum of 20 or 21, as in Example 5.1. The value of this state under the target policy is approximately _−_ 0 _._ 27726 (this was determined by separately generating one-hundred million episodes using the target policy and averaging their returns). Both o↵-policy methods closely approximated this value after 1000 o↵-policy episodes using the random policy. Figure 5.7 shows the mean squared error (estimated from 100 independent runs) for each method as a function of number of episodes. The weighted importance-sampling method has much lower overall error in this example, as is typical in practice. 


![](data/sbchap5.pdf-0018-03.png)


**----- Start of picture text -----**<br>
4<br>Ordinary<br>Mean importance<br>sampling<br>square<br>2<br>error<br>(average over<br>100 runs)<br>Weighted importance sampling<br>0<br>0 10 100 1000 10,000<br>Episodes (log scale)<br>**----- End of picture text -----**<br>


Figure 5.7: Weighted importance sampling produces lower error estimates of the value of a single blackjack state from o↵-policy episodes (see Example 5.4). 


![](data/sbchap5.pdf-0019-01.png)


**----- Start of picture text -----**<br>
R  = +1<br>⇡ (back |s ) = 1<br>0.1<br>back s<br>0.9 end µ (back |s ) = [1]<br>2<br>2<br>Monte-Carlo<br>estimate of<br>          with  vπ ( s )<br>ordinary<br>importance  1<br>sampling<br>(ten runs)<br>0<br>1 10 100 1000 10,000 100,000 1,000,000 10,000,000 100,000,000<br>Episodes (log scale)<br>**----- End of picture text -----**<br>


Figure 5.8: Ordinary importance sampling produces surprisingly unstable estimates on the one-state MDP shown inset (Example 5.5). The correct estimate here is 1, and, even though this is the expected value of a sample return (after importance sampling), the variance of the samples is infinite, and the estimates do not convergence to this value. These results are for o↵-policy first-visit MC. 

## **Example 5.5: Infinite Variance** 

The estimates of ordinary importance sampling will typically have infinite variance, and thus unsatisfactory convergence properties, whenever the scaled returns have infinite variance—and this can easily happen in o↵-policy learning when trajectories contain loops. A simple example is shown inset in Figure 5.8. There is only one nonterminal state _s_ and two actions, end and back. The end action causes a deterministic transition to termination, whereas the back action transitions, with probability 0.9, back to _s_ or, with probability 0.1, on to termination. The rewards are +1 on the latter transition and otherwise zero. Consider the target policy that always selects back. All episodes under this policy consist of some number (possibly zero) of transitions back to _s_ followed by termination with a reward and return of +1. Thus the value of _s_ under the target policy is thus 1. Suppose we are estimating this value from o↵-policy data using the behavior policy that selects end and back with equal probability. The lower part of Figure 5.8 shows ten independent runs of the first-visit MC algorithm using ordinary importance sampling. Even after millions of episodes, the estimates fail to converge to the correct value of 1. In contrast, the weighted importance-sampling algorithm would give an estimate of exactly 1 everafter the first episode that was consistent with the target policy (i.e., that ended with the back action). This is clear because 

that algorithm produces a weighted average of the returns consistent with the target policy, all of which would be exactly 1. 

We can verify that the variance of the importance-sampling-scaled returns is infinite in this example by a simple calculation. The variance of any random variable _X_ is the expected value of the deviation from its mean _X_[¯] , which can be written 


![](data/sbchap5.pdf-0020-04.png)


Thus, if the mean is finite, as it is in our case, the variance is infinite if and only if the expectation of the square of the random variable is infinite. Thus, we need only show that the expected square of the importance-sampling-scaled return is infinite: 


![](data/sbchap5.pdf-0020-07.png)



![](data/sbchap5.pdf-0020-08.png)


To compute this expectation, we break it down into cases based on episode length and termination. First note that, for any episode ending with the end action, the importance sampling ratio is zero, because the target policy would never take this action; these episodes thus contribute nothing to the expectation (the quantity in parenthesis will be zero) and can be ignored. We need only consider episodes that involve some number (possibly zero) of back actions that transition back to the nonterminal state, followed by a back action transitioning to termination. All of these episodes have a return of 1, so the _G_ 0 factor can be ignored. To get the expected square we need only consider each length of episode, multiplying the probability of the episode’s occurrence by the square of its importance-sampling ratio, and add these up: 


![](data/sbchap5.pdf-0020-10.png)



![](data/sbchap5.pdf-0020-11.png)



![](data/sbchap5.pdf-0020-12.png)


## **5.6 Incremental Implementation** 

Monte Carlo prediction methods can be implemented incrementally, on an episode-by-episode basis, using extensions of the techniques described in Chapter 2. Whereas in Chapter 2 we averaged _rewards_ , in Monte Carlo methods we average _returns_ . In all other respects exactly the same methods as used in Chapter 2 can be used for _on-policy_ Monte Carlo methods. For _o↵-policy_ Monte Carlo methods, we need to separately consider those that use _ordinary_ importance sampling and those that use _weighted_ importance sampling. 

In ordinary importance sampling, the returns are scaled by the importance sampling ratio _⇢[T] t_[(] _[t]_[)] (5.3), then simply averaged. For these methods we can again use the incremental methods of Chapter 2, but using the scaled returns in place of the rewards of that chapter. This leaves the case of o↵-policy methods using _weighted_ importance sampling. Here we have to form a weighted average of the returns, and a slightly di↵erent incremental algorithm is required. 

Suppose we have a sequence of returns _G_ 1 _, G_ 2 _, . . . , Gn−_ 1, all starting in the same state and each with a corresponding random weight _Wi_ (e.g., _Wi_ = _⇢[T] t_[(] _[t]_[)] ). We wish to form the estimate 

P _n−_ 1 


![](data/sbchap5.pdf-0021-07.png)


and keep it up-to-date as we obtain a single additional return _Gn_ . In addition to keeping track of _Vn_ , we must maintain for each state the cumulative sum _Cn_ of the weights given to the first _n_ returns. The update rule for _Vn_ is 


![](data/sbchap5.pdf-0021-09.png)



![](data/sbchap5.pdf-0021-11.png)


and 


![](data/sbchap5.pdf-0021-13.png)


where _C_ 0 = 0 (and _V_ 1 is arbitrary and thus need not be specified). Figure 5.9 gives a complete episode-by-episode incremental algorithm for Monte Carlo policy evaluation. The algorithm is nominally for the o↵-policy case, using weighted importance sampling, but applies as well to the on-policy case just by choosing the target and behavior policies as the same. 

Initialize, for all _s 2_ S, _a 2_ A( _s_ ): _Q_ ( _s, a_ ) arbitrary _C_ ( _s, a_ ) 0 _µ_ ( _a|s_ ) an arbitrary soft behavior policy _⇡_ ( _a|s_ ) an arbitrary target policy Repeat forever: Generate an episode using _µ_ : _S_ 0 _, A_ 0 _, R_ 1 _, . . . , ST −_ 1 _, AT −_ 1 _, RT , ST G_ 0 _W_ 1 For _t_ = _T −_ 1 _, T −_ 2 _, . . ._ downto 0: _G γG_ + _Rt_ +1 _C_ ( _St, At_ ) _C_ ( _St, At_ ) + _W Q_ ( _St, At_ ) _Q_ ( _St, At_ ) + _C_ ( _SWt,At_ )[[] _[G][ −][Q]_[(] _[S][t][, A][t]_[)]] _W W[⇡]_[(] _[A][t][|][S][t]_[)] _µ_ ( _At|St_ ) If _W_ = 0 then ExitForLoop 

Figure 5.9: An incremental every-visit MC policy-evaluation algorithm, using weighted importance sampling. The approximation _Q_ converges to _q⇡_ (for all encountered state–action pairs) even though all actions are selected according to a potentially di↵erent policy, _µ_ . In the on-policy case ( _⇡_ = _µ_ ), _W_ is always 1. 

## **5.7 O↵-Policy Monte Carlo Control** 

We are now ready to present an example of the second class of learning control methods we consider in this book: o↵-policy methods. Recall that the distinguishing feature of on-policy methods is that they estimate the value of a policy while using it for control. In o↵-policy methods these two functions are separated. The policy used to generate behavior, called the _behavior_ policy, may in fact be unrelated to the policy that is evaluated and improved, called the _target_ policy. An advantage of this separation is that the target policy may be deterministic (e.g., greedy), while the behavior policy can continue to sample all possible actions. 

O↵-policy Monte Carlo control methods use one of the techniques presented in the preceding two sections. They follow the behavior policy while learning about and improving the target policy. These techniques requires that the behavior policy has a nonzero probability of selecting all actions that might be selected by the target policy (coverage). To explore all possibilities, we require that the behavior policy be soft (i.e., that it select all actions in all states with nonzero probability). 

Figure 5.10 shows an o↵-policy Monte Carlo method, based on GPI and weighted importance sampling, for estimating _q⇤_ . The target policy _⇡_ is the greedy policy with respect to _Q_ , which is an estimate of _q⇡_ . The behavior policy _µ_ can be anything, but in order to assure convergence of _⇡_ to the optimal policy, an infinite number of returns must be obtained for each pair of state and action. This can be assured by choosing _µ_ to be _"_ -soft. 

A potential problem is that this method learns only from the _tails_ of episodes, after the last nongreedy action. If nongreedy actions are frequent, then learning will be slow, particularly for states appearing in the early portions of long episodes. Potentially, this could greatly slow learning. There has been insufficient experience with o↵-policy Monte Carlo methods to assess how serious this problem is. If it is serious, the most important way to address it is probably by incorporating temporal-di↵erence learning, the algorithmic idea developed in the next chapter. Alternatively, if _γ_ is less than 1, then the idea developed in the next section may also help significantly. 

136 

Initialize, for all _s 2_ S, _a 2_ A( _s_ ): _Q_ ( _s, a_ ) arbitrary _C_ ( _s, a_ ) 0 _⇡_ ( _s_ ) a deterministic policy that is greedy with respect to _Q_ Repeat forever: Generate an episode using any soft policy _µ_ : _S_ 0 _, A_ 0 _, R_ 1 _, . . . , ST −_ 1 _, AT −_ 1 _, RT , ST G_ 0 _W_ 1 For _t_ = _T −_ 1 _, T −_ 2 _, . . ._ downto 0: _G γG_ + _Rt_ +1 _C_ ( _St, At_ ) _C_ ( _St, At_ ) + _W Q_ ( _St, At_ ) _Q_ ( _St, At_ ) + _C_ ( _SWt,At_ )[[] _[G][ −][Q]_[(] _[S][t][, A][t]_[)]] _⇡_ ( _St_ ) argmax _a Q_ ( _St, a_ ) (with ties broken arbitrarily) _W W_ 1 _µ_ ( _At|St_ ) If _W_ = 0 then ExitForLoop 

Figure 5.10: An o↵-policy every-visit MC control algorithm, using weighted importance sampling. The policy _⇡_ converges to optimal at all encountered states even though actions are selected according to a di↵erent soft policy _µ_ , which may change between or even within episodes. 

## _⇤_ **5.8 Importance Sampling on Truncated Re-** 

## **turns** 

So far our o↵-policy methods have formed importance-sampling ratios for returns considered as unitary wholes. This is clearly the right thing for a Monte Carlo method to do in the absence of discounting (i.e., if _γ_ = 1), but if _γ <_ 1 then there may be something better. Consider the case where episodes are long and _γ_ is significantly less than 1. For concreteness, say that episodes last 100 steps and that _γ_ = 0. The return from time 0 will then be _G_ 0 = _R_ 1, and its importance sampling ratio will be a product of 100 factors, _⇡_ ( _A_ 0 _|S_ 0) _⇡_ ( _A_ 1 _|S_ 1) _µ_ ( _A_ 0 _|S_ 0) _µ_ ( _A_ 1 _|S_ 1) _[· · ·][ ⇡] µ_ ([(] _A[A]_ 99[99] _|[|] S[S]_ 99[99] )[)][. In ordinary importance sampling, the return will] be scaled by the entire product, but it is really only necessary to scale by the first factor, by _[⇡] µ_ ([(] _A[A]_ 0[0] _|[|] S[S]_ 0[0] )[)][. The other 99 factors] _[ ⇡] µ_ ([(] _A[A]_ 1[1] _|[|] S[S]_ 1[1] )[)] _[· · ·][ ⇡] µ_ ([(] _A[A]_ 99[99] _|[|] S[S]_ 99[99] )[)][are irrelevant] because after the first reward the return has already been determined. These later factors are all independent of the return and of expected value 1; they do not change the expected update, but they add enormously to its variance. In some cases they could even make the variance infinite. Let us now consider an idea for avoiding this large extraneous variance. 

The essence of the idea is to think of discounting as determining a probability of termination or, equivalently, a _degree_ of partial termination. For any _γ 2_ [0 _,_ 1), we can think of the return _G_ 0 as partly terminating in one step, to the degree 1 _− γ_ , producing a return of just the first reward, _R_ 1, and as partly terminating after two steps, to the degree (1 _− γ_ ) _γ_ , producing a return of _R_ 1 + _R_ 2, and so on. The latter degree corresponds to terminating on the second step, 1 _− γ_ , and not having already terminated on the first step, _γ_ . The _−_ degree of termination on the third step is thus (1 _γ_ ) _γ_[2] , with the _γ_[2] reflecting that termination did not occur on either of the first two steps. The partial returns here are called _flat partial returns_ : 


![](data/sbchap5.pdf-0025-02.png)


where “flat” denotes the absence of discounting, and “partial” denotes that these returns do not extend all the way to termination but instead stop at _h_ , called the _horizon_ (and _T_ is the time of termination of the episode). The conventional full return _Gt_ can be viewed as a sum of flat partial returns as suggested above as follows: 


![](data/sbchap5.pdf-0025-04.png)


Now we need to scale the flat partial returns by an importance sampling ratio that is similarly truncated. As _G[h] t_[only involves rewards up to a horizon] _h_ , we only need the ratio of the probabilities up to _h_ . We define an ordinary importance-sampling estimator, analogous to (5.4), as 


![](data/sbchap5.pdf-0025-08.png)


and a weighted importance-sampling estimator, analogous to (5.5), as 


![](data/sbchap5.pdf-0025-12.png)



![](data/sbchap5.pdf-0025-13.png)


## **5.9 Summary** 

The Monte Carlo methods presented in this chapter learn value functions and optimal policies from experience in the form of _sample episodes_ . This gives them at least three kinds of advantages over DP methods. First, they can be used to learn optimal behavior directly from interaction with the environment, with no model of the environment’s dynamics. Second, they can be used with simulation or _sample models_ . For surprisingly many applications it is easy to simulate sample episodes even though it is difficult to construct the kind of explicit model of transition probabilities required by DP methods. Third, it is easy and efficient to _focus_ Monte Carlo methods on a small subset of the states. A region of special interest can be accurately evaluated without going to the expense of accurately evaluating the rest of the state set (we explore this further in Chapter 8). 

A fourth advantage of Monte Carlo methods, which we discuss later in the book, is that they may be less harmed by violations of the Markov property. This is because they do not update their value estimates on the basis of the value estimates of successor states. In other words, it is because they do not bootstrap. 

In designing Monte Carlo control methods we have followed the overall schema of _generalized policy iteration_ (GPI) introduced in Chapter 4. GPI involves interacting processes of policy evaluation and policy improvement. Monte Carlo methods provide an alternative policy evaluation process. Rather than use a model to compute the value of each state, they simply average many returns that start in the state. Because a state’s value is the expected return, this average can become a good approximation to the value. In control methods we are particularly interested in approximating action-value functions, because these can be used to improve the policy without requiring a model of the environment’s transition dynamics. Monte Carlo methods intermix policy evaluation and policy improvement steps on an episode-by-episode basis, and can be incrementally implemented on an episode-by-episode basis. 

Maintaining _sufficient exploration_ is an issue in Monte Carlo control methods. It is not enough just to select the actions currently estimated to be best, because then no returns will be obtained for alternative actions, and it may never be learned that they are actually better. One approach is to ignore this problem by assuming that episodes begin with state–action pairs randomly selected to cover all possibilities. Such _exploring starts_ can sometimes be arranged in applications with simulated episodes, but are unlikely in learning from real experience. In _on-policy_ methods, the agent commits to always exploring and tries to find the best policy that still explores. In _o↵-policy_ methods, the agent also explores, but learns a deterministic optimal policy 

that may be unrelated to the policy followed. 

_O↵-policy Monte Carlo prediction_ refers to learning the value function of a _target policy_ from data generated by a di↵erent _behavior policy_ . Such learning methods are all based on some form of _importance sampling_ , that is, on weighting returns by the ratio of the probabilities of taking the observed actions under the two policies. _Ordinary importance sampling_ uses a simple average of the weighted returns, whereas _weighted importance sampling_ uses a weighted average. Ordinary importance sampling produces unbiased estimates, but has larger, possibly infinite, variance, whereas weighted importance sampling always has finite variance and are preferred in practice. Despite their conceptual simplicity, o↵-policy Monte Carlo methods for both prediction and control remain unsettled and a subject of ongoing research. 

The Monte Carlo methods treated in this chapter di↵er from the DP methods treated in the previous chapter in two major ways. First, they operate on sample experience, and thus can be used for direct learning without a model. Second, they do not bootstrap. That is, they do not update their value estimates on the basis of other value estimates. These two di↵erences are not tightly linked, and can be separated. In the next chapter we consider methods that learn from experience, like Monte Carlo methods, but also bootstrap, like DP methods. 

## **Bibliographical and Historical Remarks** 

The term “Monte Carlo” dates from the 1940s, when physicists at Los Alamos devised games of chance that they could study to help understand complex physical phenomena relating to the atom bomb. Coverage of Monte Carlo methods in this sense can be found in several textbooks (e.g., Kalos and Whitlock, 1986; Rubinstein, 1981). 

An early use of Monte Carlo methods to estimate action values in a reinforcement learning context was by Michie and Chambers (1968). In pole balancing (Example 3.4), they used averages of episode durations to assess the worth (expected balancing “life”) of each possible action in each state, and then used these assessments to control action selections. Their method is similar in spirit to Monte Carlo ES with every-visit MC estimates. Narendra and Wheeler (1986) studied a Monte Carlo method for ergodic finite Markov chains that used the return accumulated from one visit to a state to the next as a reward for adjusting a learning automaton’s action probabilities. 

Barto and Du↵(1994) discussed policy evaluation in the context of classical Monte Carlo algorithms for solving systems of linear equations. They used 

the analysis of Curtiss (1954) to point out the computational advantages of Monte Carlo policy evaluation for large problems. Singh and Sutton (1996) distinguished between every-visit and first-visit MC methods and proved results relating these methods to reinforcement learning algorithms. 

The blackjack example is based on an example used by Widrow, Gupta, and Maitra (1973). The soap bubble example is a classical Dirichlet problem whose Monte Carlo solution was first proposed by Kakutani (1945; see Hersh and Griego, 1969; Doyle and Snell, 1984). The racetrack exercise is adapted from Barto, Bradtke, and Singh (1995), and from Gardner (1973). 

Monte Carlo ES was introduced in the 1998 edition of this book. That may have been the first explicit connection between Monte Carlo estimation and control methods based on policy iteration. 

Efficient o↵-policy learning has become recognized as an important challenge that arises in several fields. For example, it is closely related to the idea of “interventions” and “counterfactuals” in probabalistic graphical (Bayesian) models (e.g., Pearl, 1995; Balke and Pearl, 1994). O↵-policy methods using importance sampling have a long history and yet still are not well understood. Weighted importance sampling, which is also sometimes called normalized importance sampling (e.g., Koller and Friedman, 2009), is discussed by, for example, Rubinstein (1981), Hesterberg (1988), Shelton (2001), and Liu (2001). Combining o↵-policy learning with temporal-di↵erence learning and approximation methods introduces subtle issues that we consider in later chapters. 

The target policy in o↵-policy learning is sometimes referred to in the literature as the “estimation” policy, as it was in the first edition of this book. 

Our treatment of the idea of importance sampling based on truncated returns is based on the analysis and “forward view” of Sutton, Mahmood, Precup, and van Hasselt (2014). A related idea is that of per-decision importance sampling (Precup, Sutton and Singh, 2000). 

## **Exercises** 

**Exercise 5.1** Consider the diagrams on the right in Figure 5.2. Why does the estimated value function jump up for the last two rows in the rear? Why does it drop o↵for the whole last row on the left? Why are the frontmost values higher in the upper diagrams than in the lower? 

**Exercise 5.2** What is the backup diagram for Monte Carlo estimation of _q⇡_ ? 

**Exercise 5.3** What is the Monte Carlo estimate analogous to (5.5) for _action_ 


![](data/sbchap5.pdf-0029-02.png)


**----- Start of picture text -----**<br>
Finish<br>line<br>Finish<br>line<br>Starting line Starting line<br>**----- End of picture text -----**<br>


Figure 5.11: A couple of right turns for the racetrack task. 

values, given returns generated using _µ_ ? 

**Exercise 5.4** What is the equation analogous to (5.5) for _action_ values _Q_ ( _s, a_ ) instead of state values _V_ ( _s_ )? 

**Exercise 5.5** In learning curves such as those shown in Figure 5.7 error generally decreases with training, as indeed happened for the ordinary importancesampling method. But for the weighted importance-sampling method error first increased and then decreased. Why do you think this happened? 

**Exercise 5.6** The results with Example 5.5 and shown in Figure 5.8 used a first-visit MC method. Suppose that instead an every-visit MC method was used on the same problem. Would the variance of the estimator still be infinite? Why or why not? 

**Exercise 5.7** Modify the algorithm for first-visit MC policy evaluation (Figure 5.1) to use the incremental implementation for sample averages described in Section 2.4. 

**Exercise 5.8** Derive the weighted-average update rule (5.7) from (5.6). Follow the pattern of the derivation of the unweighted rule (2.3). 

**Exercise 5.9: Racetrack (programming)** Consider driving a race car around a turn like those shown in Figure 5.11. You want to go as fast as possible, but not so fast as to run o↵the track. In our simplified racetrack, the car is at one of a discrete set of grid positions, the cells in the diagram. The velocity is also discrete, a number of grid cells moved horizontally and vertically per time step. The actions are increments to the velocity components. Each may be changed by +1, _−_ 1, or 0 in one step, for a total of nine actions. 

Both velocity components are restricted to be nonnegative and less than 5, and they cannot both be zero. Each episode begins in one of the randomly selected start states and ends when the car crosses the finish line. The rewards are _−_ 1 for each step that stays on the track, and _−_ 5 if the agent tries to drive o↵the track. Actually leaving the track is not allowed, but the position is always advanced by at least one cell along either the horizontal or vertical axes. With these restrictions and considering only right turns, such as shown in the figure, all episodes are guaranteed to terminate, yet the optimal policy is unlikely to be excluded. To make the task more challenging, we assume that on half of the time steps the position is displaced forward or to the right by one additional cell beyond that specified by the velocity. Apply a Monte Carlo control method to this task to compute the optimal policy from each starting state. Exhibit several trajectories following the optimal policy. 

> _⇤_ **Exercise 5.10** Modify the algorithm for o↵-policy Monte Carlo control (Figure 5.10) to use the idea of the truncated weighted-average estimator (5.9). Note that you will first need to convert this equation to action values. 

