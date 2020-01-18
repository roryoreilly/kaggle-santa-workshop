# Kaggle: Santa's Workshop Tour 2019

This repo is for the Kaggle competition https://www.kaggle.com/c/santa-workshop-tour-2019
A number of ideas were tried for this competition. With the best results from using MIP (Mixed-Integer Programming) Solvers. Two solvers that I attemped to use were IBM's CPLEX and Google's OR-Tools.

## Results
My final results were AAA place out of BBB. This resulted in a MMM medal as it was in the top NNNN. The final solution value was XXXX.

This result was achieved using IBM's CPLEX, running on a Google Compute server.

## Brief Description of competition
- Santa is opening his workshop for 100 days for Christmas.
- 5000 families of varying sizes would like to visit santa's workshop
- Each family has a preference on the day they would like to visit the workshop
- There is a cost to not giving a family their top pick so the goal is to try give a familiy one of the best picks
- There is also a cost to having a large difference in people one day compared to the next
- The final constrait is that there must be between 125 and 300 people for every day that the workshop is open

## Repo packages
cplex
	- Code to run the model with IBM's CPLEX. A full version of CPLEX is required for this.
docloud 
	- Without a full version of CPLEX, it is possible to get a trial of IBM's docloud. This is a server that can run CPLEX models on 10 core 60 GB machines. However, the trial version will only allow a model to last 1 hour of compute time. It did not successfully work due to this limitation.
eda 
	- Any exploritory work for the project
input 
	- data provided by kaggle
ortools 
	- Implementations using Google's OR-Tools. One model using their CBC solver and another using their CP solver
reinforement-learning 
	- An attempted implementation was to turn this problem into a reinforement learning problem. Deep Q and Double Deep Q were used to create an implemention
submissions 
	- Saved submissions during the work. 

## Hardware used
Due to the complexity of the problem, a higher powered computer than what I own was required. Therefore Google Compute was used. This project was run over a few days on a computer with 8vCPUs and 52 GB of memory
