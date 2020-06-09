# 台大李宏毅助教讲解GCN

##### 引子：RB-tree

Node有性质、Edge有性质

##### 怎么把一个Graph塞入一个NN?

###### Why do we need GNN?

有很多label好的sample,训练，做Classification、Generation
有些东西看起来不是Graph(找出凶手)
	
$$
f(人)=不是凶手
$$
忽略了什么东西？underlying structure and relationship 额外的资讯
这就是用GNN的原因，要考虑所有节点之间的关系

##### How do we utilize the structures and relationship to help our model?

##### What if the graph is larger?

很简单的想法：近朱者赤近墨者黑 Semi-Supervised
用邻居来学习出一个好的representation
Convolution?: kernel/filter->generlized到Graph上
How to embed node into a feature space using convolution?
Solution1:Generlize the concept of convolution(corelation)
	to graph(Spatial-based convolution)  CNN的方法
Solution2:Back to the definition of convolution in signal
	processing (Spectral-based convolution)  玄妙、signal processing


GNN Roadmap;
Convolution

Spatial-based             Spectral-based 谱的
Aggregation  Method		  ChebNet->GCN->HyperGCN
Sum          NN4G
Mean		 ...
Weighted Sum GAT
LSTM
Max Pooling

客观的衡量mode的performance
training set和volidation set

Spatial-based GNN
terminology:术语
	Aggregate:用neighbor feature update下一层的hidden state
	Readout: 把所有nodes的feature集合起来代表整个graph

NN4G(Neural Network for Graph)
Input Layer ->Hidden Layer 0 ->Hidden Layer1
sum邻居+自己的feature

DCNN(Diffusion-Convolution Neural Network)
不同距离的feature 1、2、3都有一个矩阵 叠很多矩阵

GAT(Graph Attention Networks)
f(h3,h0)=energy 3,0
f(h3,h2)=energy 3,2
f(h3,h4)=energy 3,4

h3 = e3,0*h0 +e3,2*h2 +e3,4*h4

GIN结论：Sum instead of mean or max
MLP:MultilayerPerception 多层感知机 而不是一层
MLP(Sum(Neighbour)+k*self)


没办法定义九宫格
把一个graph上的signal透过
fourier transform
再把filter做fourier transform
两个一乘就可以得到Inverse Fourier Transform

一个信号可以看做是一个N维空间的一个向量
一个信号是一组basis的合成，线性组合成Fourier Series
不同component的大小由ak决定，求每个小信号的ak(系数)
在time-domain
选的basis是e jkw0t,

把同样的信号写成另外一种X(jw)ej w dw
X(jw)=积分x(t)e-jwt dt:spectrum,frequency domain Fourier Transform

Spectral Graph Theory
Graph:G=(V,E),N=|V|
用一个邻接矩阵,权是weight,记作A,无向图是一个对称矩阵
一个度矩阵，diagnal对角线上的值是这个节点的度，记作D
一个信号矩阵f:V->RN,signal on graph(vertices)
f(i) denotes this signal on vertex i

Graph Laplacian L = D - A (半正定的对称矩阵，所有vector都是大于等于零的数)
L is symmetric
L=UΛUT  (spectral decomposition)
Λ是diagnal matrix，里面是(λ1,λ2,λ3...)
λ is the frequency, u is the basis corresponding to λ
其中U和UT是正交的(矩阵分解)

在Discrete time Fourier basis 中
频率越大，相邻两点之间的信号变化越大(画个cosx的图像就可以看出来)

Interpreting vertex frequency
 L as an operator on graph
 Given a graph signal f,what does Lf mean?
 Lf = (D-A)f = Df - Af

Lf apply在一个信号上就会是v0和邻接点信号差异的和
a=[2,0,0,0].[4,2,4,-3]-[0,1,1,0][4,2,4,-3]
=2*4 - 2 -4
2*4是signal on v0
-2,-4是signal on v0's neighbors

求出来的结果是Sum of difference between v0 and its neighbors
如果要求能量，是要做一个平方
Power of signal variation between nodes = f transpose L f
直观表现节点间的Smoothness
eigenvector(特征向量)
λ(eigenvalue)特征值的大小代表点之间信号频率的差异大小

Filtering
学到一组参数θ


ChebNet 快、而且可以localize
Use polynominal to parametrize g(L)
    多项式的
Parameters to be learnt:O(K)
Solution to this problem:递归的多项式
Use Chebyshev polynomial 替换拉普拉斯的basis
用Λk非常难算，数学方法简便，降低时间复杂度complexity

GCN
A tilde替偶打(波浪线)

Benchmark
1、Graph Classification : M-NIST and CIFAR10
						  40-50
2、Regression            ZINC Chemistry  9-37
						 PATTERN  		 50-180
						 CLUSTER 分簇 Stochastic Block Model  40-190
3、Edge Classification:   TSP:Traveling Salesman Problem 50-500

DropEdge 可以避免over-smoothing的问题





##### auto-encoder

中间叫embedding
encoder-embedding-decoder
最后需要的其实是embedding

What is good embedding?
NN Encoder吃进去一个图片，就会输出一个embedding,那么怎么判断embedding的好坏呢？有一个Discrimnator(binary classifier)，loss of the classification task is Ld,使这个Ld最小，就代表embedding具有代表性；
如果输出的都不对，(灰色)，就不具有代表性。