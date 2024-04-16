# Paper title Flexible Visual Recognition by Evidential Modeling of Confusion and Ignorance

This readme file is an outcome of the [CENG502 (Spring 2024)](https://ceng.metu.edu.tr/~skalkan/ADL/) project for reproducing a paper without an implementation. See [CENG502 (Spring 2024) Project List](https://github.com/CENG502-Projects/CENG502-Spring2024) for a complete list of all paper reproduction projects.

# 1. Introduction

This project is a reproduction of the paper titled “Flexible Visual Recognition by Evidential Modeling of Confusion and Ignorance”, which was published in the International Conference on Computer Vision (ICCV) in 2023. The paper tackles the inherent difficulties in visual recognition systems, specifically the frequent errors in classifying known categories and the system’s tendency to incorrectly handle images that don’t belong to any of the classes it has been trained on. The goal of this project is to reproduce the results presented in the paper, specifically focusing on the experiments conducted on the CIFAR-10 and CIFAR-100 datasets, and verify the effectiveness of the proposed method

## 1.1. Paper summary

The paper introduces a unique method that allows a visual recognition system to express uncertainty and source trust explicitly. This is particularly crucial in real-world scenarios where the input can be unpredictable and varied. The authors focus on two types of uncertainties: confusion and ignorance. By predicting Dirichlet concentration parameters for singletons, the system forms what the authors call “subjective opinions”. These subjective opinions enable the system to make more flexible decisions, enhancing its adaptability and performance in visual recognition tasks. The effectiveness of the proposed method is demonstrated through a series of experiments on synthetic data analysis, visual recognition, and open-set detection.

# 2. The method and my interpretation

## 2.1. The original method

The original method proposed in the paper is a novel approach to handle uncertainties in visual recognition systems using the theory of Subjective Logic. This theory allows the system to express uncertainty and source trust explicitly, which is crucial in real-world scenarios where the input can be unpredictable and varied.

The method focuses on two types of uncertainties:

1. **Confusion**: This type of uncertainty arises when the system finds it challenging to make a clear distinction between known classes. For instance, an image might contain features that match multiple known classes, making it difficult for the system to confidently assign it to one specific class. The overall confusion $C$ is the total mass of the non-singleton subsets. In other words, the confusion $C$ is the sum of masses shared between two or more classes. This is represented as:

$$C = \sum_{A, A \in 2^{\Theta}, 2 \leq |A| \leq K} b_A$$

2. **Ignorance**: This type of uncertainty comes into play when the system encounters an input that is entirely outside its training distribution. In such cases, the system lacks any relevant evidence to base a decision on, leading to high ignorance. The ignorance $I$ could be calculated similarly as:

$$I = \prod_{j=1}^{K} (1 - pl_j) = \prod_{j=1}^{K} f_{j}^{2}(x)$$

and the total confusion between all different class combinations is $C = U - I$.

The uncertainty $U$ for each sample $x$ comes from two distinct sources, i.e., confusion $C$ and ignorance $I$. This is represented as:

$$U_x = C_x + I_x$$

To model these uncertainties, the method predicts Dirichlet concentration parameters for singletons. In the context of Subjective Logic, a singleton is a set with only one element. These predictions allow the system to form what the authors call “subjective opinions”.

A subjective opinion in this context is a type of probabilistic estimate that provides a probability for each possible outcome, along with an estimate of the uncertainty of these probabilities. This dual nature of subjective opinions allows the system to express nuanced views like “I think this image is probably of a cat, but I’m not very sure”.

By forming these subjective opinions, the system can make more flexible decisions. For example, if the system is very uncertain (high confusion), it can decide to predict multiple classes instead of just one. If the system doesn’t have any relevant evidence (high ignorance), it can decide to reject making a prediction altogether.

This approach allows the system to handle the complexities and uncertainties of real-world visual recognition tasks more effectively, enhancing its adaptability and performance. It’s a significant step forward in the field of visual recognition, paving the way for more robust and flexible systems.

In terms of methodology, the paper uses the theory of Subjective Logic to model the uncertainties. The recognition process is regarded as an evidence-collecting process where confusion is defined as conflicting evidence, while ignorance is the absence of evidence. By predicting Dirichlet concentration parameters for singletons, comprehensive subjective opinions, including confusion and ignorance, could be achieved via further evidence combinations.

The paper proposes to decompose the problem into $K$ plausibility functions $f_i(\cdot)$ for $i = 1, . . . , K$ on the same frame. Each plausibility function $f_i(\cdot)$ is designed to give two predictions only considering class $i$, which is written as $f_i(x) = (pl_i, 1 - pl_i)$.

For $K$ plausibility functions, the belief assignment of any proposition $A$ is combined by computing as:

$$b_A = \sum_{B, \cap B = A} \prod_{j=1}^{K} b_{B,j}(x) = \sum_{B, \cap B = A} \prod_{j=1}^{K} f_{j}^{B}(x)$$

The singleton belief for class $i$ is computed as:

$$b_i = pl_i \prod_{j=1, j \neq i}^{K} (1 - pl_j) = f_{i}^{1}(x) \prod_{j=1, j \neq i}^{K} f_{j}^{2}(x)$$

The total uncertainty $U$ is calculated as: 

$$U=1-\sum_{i=1}^{K}b_{i}$$

Each plausibility function $f_{i}(·)$ can be constructed as a normalized dual-output linear layer or a single multi-output layer after being activated by a sigmoid function $\sigma(·)$. The output is regarded as the value of class plausibility. The plausibility function is then formulated as: 

$$(pl_{i},1-pl_{i})=f_{i}(x)=\sigma(w_{i}^{T}\Phi(x))$$

To encourage the plausibility function to match our expected behavior, i.e., predicting the plausibility instead of the belief of singleton, a regularization term is added as: 

$$L_{reg}=\sum_{i=1}^{K}y_{i}[pl_{i}-(1-\hat{I})]^{2}$$

where $\hat{I}$ is the current estimation of ignorance.

Following EDL, a Kullback-Leibler loss is used to minimize evidence on unrelated classes as: 

$$L_{KL}=KL(Dir(·|α~)||Dir(·|⟨1,…,1⟩))$$

where $α~=y+(1−y)⊙α$, $⊙$ for element-wise multiplication. Combining all terms together yields the final loss as: 

$$L=L_{EDL}+λ_{reg}L_{reg}+λ_{KL}L_{KL}$$

Each loss term is accompanied by a balance weight, and we gradually increase the effect of $L_{KL}$ through an additional annealing coefficient.

After developing opinions with the proposed method, a straightforward solution would be setting the belief threshold for outputs to achieve flexible recognition. The sample will be rejected if the ignorance is too large that no combination would exceed the threshold. And the model gives incrementally combined predictions if no singleton belief meets the bar.

The learning of singleton belief is implemented as evidence acquisition on a Dirichlet prior. The loss of EDL is: 

$$L_{EDL}=\sum_{i=1}^{K}y_{i}[log(\sum_{j=1}^{K}α_{j})-log(α_{i})]$$

where $y = [y_{1}, y_{2}, . . . , y_{i}, . . . , y_{K}]^{T}$ is one-hot class label for a sample $x$, and $α = [α_{1}, α_{2}, . . . α_{K}]^{T}$ are parameters of a Dirichlet distribution $Dir(·|α)$.

Different from EDL, class evidence is replaced with belief. Hence, $α$ is directly calculated from singleton beliefs and overall uncertainty. It is derived as: 

$$α_{i}=KU_{b_{i}}+1=\dfrac{1-\sum_{j=1}^{K}b_{j}}{K}Kb_{i}+1$$

where $b_{i}$ could be obtained from Eq. 6. During inference, all opinions could be directly predicted by performing combinations on the output of plausibility functions.

## 2.2. Our interpretation 

@TODO: Explain the parts that were not clearly explained in the original paper and how you interpreted them.

# 3. Experiments and results

## 3.1. Experimental setup

In the original paper, a variety of experiments were conducted to test the proposed method's effectiveness in handling uncertainties in visual recognition systems. The experiments were designed to demonstrate the separation of two sources of uncertainty, indicate the correct class on misclassified samples with estimated confusion, and apply ignorance to compare with other methods on the task open-set detection.

The authors used the ResNet-18 as the backbone for their experiments, except on synthetic data and open-set detection. The dimension of the extracted feature was set to 512. For the proposed method, they applied the sigmoid activation on the last linear layer to work as multiple plausibility functions. They found both EDL and their method to be more sensitive to the learning rate. Specifically, they set the learning rate for both methods to 0.004 with a momentum of 0.9 for the batch size of 128. The $\lambda_{KL}$ in Eq. 13 annealed to 0 with epochs with the maximum coefficient of 0.05, and $\lambda_{reg}$ was set to 1.

For our study, we have decided to focus specifically on the CIFAR-10 and CIFAR-100 datasets for the open-set detection task. These datasets are directly obtained from $\texttt{torchvision.datasets.CIFAR10}$ and $\texttt{torchvision.datasets.CIFAR100}$ respectively.

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

The CIFAR-100 dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 coarse classes.

This focus will allow us to thoroughly investigate and understand the performance of the proposed method on these particular datasets. We have not made any changes to the original experimental setup and have followed the same procedures and settings as described in the paper to ensure the accuracy of our results. Our goal is to verify the effectiveness of the proposed method as presented in the original paper.

## 3.2. Running the code

@TODO: Explain your code & directory structure and how other people can run it.

## 3.3. Results

@TODO: Present your results and compare them to the original paper. Please number your figures & tables as if this is a paper.

# 4. Conclusion

@TODO: Discuss the paper in relation to the results in the paper and your results.

# 5. References

@TODO: Provide your references here.

# Contact

@TODO: Provide your names & email addresses and any other info with which people can contact you.
