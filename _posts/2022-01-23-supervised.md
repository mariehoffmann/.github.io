## Supervised Learning

### Linear Regression

In regression we seek to estimate an unknown function _f_ that maps a _d_-dimensional sample to a continuous value. 

<img src="https://latex.codecogs.com/svg.image?\begin{array}{lcl}\hat{y}(x)&\stackrel{!}{=}&&space;f(x)\\\hat{y}:\mathbb{R}^d&\mapsto&\mathbb{R}\end{array}" title="\begin{array}{lcl}\hat{y}(x)&=& f(x)\\\hat{y}:\mathbb{R}^d&\mapsto&\mathbb{R}\end{array}" />

A simple way of trying to estimate the unknown function, is to learn the importance of each vector component, such that an error function (or loss) is minimized. 

<img src="https://latex.codecogs.com/svg.image?\hat{y}(x)&space;=&space;x&space;\beta" title="\hat{y}(x) = x\beta" />

Note, that we silently assume that the original sample _x^o_ has already been augmented by an extra dimension set to 1. This accounts for the offset or bias. Think of it as the y-axis intercept in the one-dimensional case.

<img src="https://latex.codecogs.com/svg.image?x&space;=&space;[&space;1,&space;x^{o}_1,&space;x^{o}_2,&space;\cdots&space;,&space;x^{o}_{d-1}],&space;~X&space;=&space;\left&space;[&space;\begin{array}{cccc}x_{1,1}&x_{1,2}&\cdots&space;&x_{1,d}\\x_{2,1}&x_{2,2}&\cdots&space;&x_{2,d}\\\vdots&space;&\vdots&\vdots&\vdots\\x_{n,1}&space;&&space;x_{n,2}&\cdots&space;&x_{n,d}\end{array}&space;\right&space;]" title="x = [ 1, x^{o}_1, x^{o}_2, \cdots , x^{o}_{d-1}], ~X = \left [ \begin{array}{cccc}x_{1,1}&x_{1,2}&\cdots &x_{1,d}\\x_{2,1}&x_{2,2}&\cdots &x_{2,d}\\\vdots &\vdots&\vdots&\vdots\\x_{n,1} & x_{n,2}&\cdots &x_{n,d}\end{array} \right ]" />

#### Least Squared Error
A simple way is to express the error is to take the squared difference between estimator and true value. 

<img src="https://latex.codecogs.com/svg.image?L^{ls}(\beta)&space;=&space;||\beta&space;X&space;-&space;y&space;||_2^2" title="L^{ls}(\beta) = ||\beta X - y ||_2^2" />

The corresponding loss function to be minimized is the mean of squared errors (MSE). Some clarification about the semantic differences between _loss_, _cost_ and _objective functions_ can be found [here](https://stats.stackexchange.com/a/179027).

<img src="https://latex.codecogs.com/svg.image?MSE(\beta)&space;=&space;\frac{1}{n}||\beta&space;X&space;-&space;y&space;||_2^2" title="MSE(\beta) = \frac{1}{n}||\beta X - y ||_2^2" />

We are now searching for the parameter combination that minimizes our cost function: 

<img src="https://latex.codecogs.com/svg.image?\beta&space;=&space;\underset{\beta'}{\text{argmin}}~L^{ls}(\beta')" title="\beta = \underset{\beta'}{\text{argmin}}~L^{ls}(\beta')" />

The solution can be computed analytically with linear algebra. 

<img src="https://latex.codecogs.com/svg.image?\begin{array}{rcl}\frac{\partial}{\partial{\beta}}\big&space;[||X\beta&space;-&space;y||^2_2\big&space;]&=&0\\\frac{\partial}{\partial{\beta}}\big&space;[(X\beta&space;-&space;y)^T(X\beta&space;-y)\big&space;]&=&0\\\frac{\partial}{\partial{\beta}}\big&space;[(X\beta)^TX\beta&space;-&space;2(X\beta)^Ty&space;&plus;y^Ty\big&space;]&=&0\\&space;&space;2X^TX\beta&space;-&space;2X^Ty&=&0\\&space;X^TX\beta&=&X^Ty\\&space;(X^TX)^{-1}X^TX\beta&=&(X^TX)^{-1}y^TX\\&space;\beta&=&(X^TX)^{-1}y^TX\end{array}&space;" title="\begin{array}{rcl}\frac{\partial}{\partial{\beta}}\big [||X\beta - y||^2_2\big ]&=&0\\\frac{\partial}{\partial{\beta}}\big [(X\beta - y)^T(X\beta -y)\big ]&=&0\\\frac{\partial}{\partial{\beta}}\big [(X\beta)^TX\beta - 2(X\beta)^Ty +y^Ty\big ]&=&0\\ 2X^TX\beta - 2X^Ty&=&0\\ X^TX\beta&=&X^Ty\\ (X^TX)^{-1}X^TX\beta&=&(X^TX)^{-1}y^TX\\ \beta&=&(X^TX)^{-1}y^TX\end{array} " />


### Summary Linear Regression

The power of linear regression cannot be underestimated. We are not obliged to find a linear combination of the vector components, but can generate arbitrary, non-linear features. 

                  
### From Linear Regression to Classification
In contrast to LR, we want to learn a discriminator _f(x, y)_ that is high if _y_ is the correct class of sample _x_.

<img src="https://latex.codecogs.com/svg.image?\begin{array}{lcl}\hat{y}(x)&=&&space;\underset{y}{\text{argmax}}~f(y,&space;x)\\\hat{y}:\mathbb{R}^d&\mapsto&[1:k]\end{array}" title="\begin{array}{lcl}\hat{y}(x)&=& \underset{y}{\text{argmax}}~f(y, x)\\\hat{y}:\mathbb{R}^d&\mapsto&[1:k]\end{array}" />

### Support Vector Machines (SVM)

Support vector machines (SVMs) allow the modeling of non-linear decision boundaries through the "kernel trick". The additional features are generated on the fly by a kernel function. The kernel returns only the inner product between a sample and a _landmark_ or _support vector_. Similar to the K-nearest neighbor method, the support vectors are taken from the training set.

<img src="https://latex.codecogs.com/svg.image?f(x,&space;x_i)&space;=&space;\theta_i^T\kappa(x,&space;x_i)" title="f(x, x_i) = \theta_i^T\kappa(x, x_i)" />

<img src="https://latex.codecogs.com/svg.image?\begin{array}{ccl}\kappa&space;:&space;\mathbb{R}^d&space;\times&space;\mathbb{R}^d&space;&\mapsto&&space;\mathbb{R}&space;\\\kappa&space;(x,&space;x_i)&space;&=&&space;\langle&space;\phi(x),&space;\phi(x_i)\rangle&space;\\&space;\phi&space;:\mathbb{R}^d&space;&\mapsto&&space;\mathbb{R}^m&space;\end{array}" title="\begin{array}{ccl}\kappa : \mathbb{R}^d \times \mathbb{R}^d &\mapsto& \mathbb{R} \\\kappa (x, x_i) &=& \langle \phi(x), \phi(x_i)\rangle \\ \phi :\mathbb{R}^d &\mapsto& \mathbb{R}^m \end{array}" />

The feature transformation φ projects into an _m_-dimensional space. The number of dimensions _m_ can be potentially much larger than $d$. By taking the inner product the kernel function is returning a scalar.

<img src="https://latex.codecogs.com/svg.image?\langle&space;\cdot,&space;\cdot&space;\rangle:\mathbb{R}^m\times\mathbb{R}^m&space;\mapsto&space;\mathbb{R}" title="\langle \cdot, \cdot \rangle:\mathbb{R}^m\times\mathbb{R}^m \mapsto \mathbb{R}" />
