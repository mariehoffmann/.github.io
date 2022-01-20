 # Machine Learning Summary

This blog is intended for those who want to get a comprehensive overview of the basic machine learning (ML) methods. Currently, the ML community subdivides the field into
  1. Unsupervised Learning
  2. Supervised Learning
  3. Reinforcement Learning

This blog will be structured likewise as the complexity of the methods increases. E.g., reinforcement learning is deployed in practice with techniques from supervised learning, i.e., relevant features of an agent's environment are learned via a neural network.

Note: LaTeX formulas are rendered to svg images with black font. Therefore, therefore you cannot use your browser's darkmode if you want to see the formulas :)

---

## Unsupervised Learning

## Supervised Learning

### Linear Regression

In regression we seek to estimate an unknown function _f_        

<img src="https://latex.codecogs.com/svg.image?\begin{array}{lcl}\hat{y}(x)&=&&space;f(x)\\\hat{y}:\mathbb{R}^d&\mapsto&\mathbb{R}\end{array}" title="\begin{array}{lcl}\hat{y}(x)&=& f(x)\\\hat{y}:\mathbb{R}^d&\mapsto&\mathbb{R}\end{array}" />

                  
### Linear Classifyer

<img src="https://latex.codecogs.com/svg.image?\begin{array}{lcl}\hat{y}(x)&=&&space;\underset{y}{\text{argmax}}~f(y,&space;x)\\\hat{y}:\mathbb{R}^d&\mapsto&[1:k]\end{array}" title="\begin{array}{lcl}\hat{y}(x)&=& \underset{y}{\text{argmax}}~f(y, x)\\\hat{y}:\mathbb{R}^d&\mapsto&[1:k]\end{array}" />

## Reinforcement Learning

