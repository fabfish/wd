I would have liked to see more discussion of limitations. Where would these relationships break down?

I’d like a greater background/intuition for REH as well as how this method would transfer/compare with other regularization methods like dropout or batch normalization.

关于本研究的直觉： （回答Limitations）

在模型训练中，衰减的学习率通过减小探索的步长，精细化的让算法收敛到局部极小值。而权重衰减则从抑制参数的角度入手，让算法达到相同的目的。于是，很自然的想法是学习率和权重衰减在整个优化系统中是否存在同样的作用，这促使我们研究这两种超参数在训练中的相互影响。因此，亟待找到一种等价关系来建立起这两者间的关系，泛化性能就是一种可量化的潜在备选方案。

关于正则化等价性的直觉： （回复Weakness2，Key Question 1）

我们考虑一个简单的凸优化目标，对于不同的两种正则方式，其共同的目标是通过约束条件，让函数的解落入特定的（例如平坦）区域，以获得更好的泛化性能。当它们使得函数的解都落在了最优性能处时，我们相信这两种不同的正则方式对函数达到最优目标的约束力是相同的。

举一个通俗的例子：当我们使用导弹打击一个目标时，无论采用光学制导还是热制导，这两种制导方式对最终达成击中目标所施加的“引导能力”是一样的。再例如：为了逼近同一个目标函数，我们既可以使用泰勒展式，也可以使用小波函数，尽管它们的逼近方式不同，但逼近的效果是一样的。

局限性： （回复Weakness1，Limitations）

尽管在实验中，我们在更一般的设定下验证了超参数间的关系，但在理论上，我们的等价关系构造和理论推均在凸假设下讨论。对于非凸假设下的理论研究仍需要进一步探讨。

此外，我们要求2种不同算法的最终效果要一致，于是这产生第二个局限性：本文在适当的假设下仅能得到算法的上界，而并非是上确界，这使得
和
之间建立的并不是严格的等式关系。于是，我们在文中使用近似的的关系来表示（表达式19和表达式20），同时我们称该理论研究提供了一种启发式的理论依据。

关于是否适用于其他正则化方法 （回复Key Question 1）

在本文的研究框架适用于，能够基于一致稳定分析建立泛化界的正则化方法，如dropout，我们可以对参数矩阵施加一个稀疏化因子s以模拟丢弃的元素，并在泛化界中显式的体现s的收紧作用，然后利用该泛化界与超参数建立等价关系。然而，对于无法量化的指标（如早停），则不适用于该研究框架。

对Muon的推测（回复Key Question 2）

Key Questions For Authors:

The regularization equivalence hypothesis seems like a fairly general principle. I would have liked to see more justification/intuition for why it would be the case. Should this hold across almost all regularization methods or are there particular ones, such as dropout, early stopping, normalization, etc where it would not be applicable? Are there other experimental settings or other models that lead the authors to this intuition?

I’d be interested in the author's speculation on the muon optimizer. Do you think the claims would hold as well in this case?

Limitations:

The authors have an impact statement. The authors' claims are well substantiated with evidence, but they could have gone deeper into limitations. For instance, by scoping out the areas where their results would not hold or giving intuition where their results would hold in wider circumstances.

We sincerely thank the reviewer for the careful, rigorous, and insightful reading of our manuscript. We especially appreciate the reviewer’s positive assessment of the overall direction of the paper. These comments are very important to us.

Intuition for This Study

In model training, a decaying learning rate reduces the step size of exploration, enabling the algorithm to converge finely to a local minimum. Weight decay, on the other hand, achieves the same goal by penalizing the parameters. A natural question is whether the learning rate and weight decay play equivalent roles in the overall optimization system. This motivated us to investigate the interplay between these two hyperparameters during training. Thus, it is important to find an equivalence that connects them, and generalization performance serves as a quantifiable candidate.

Intuition for Regularization Equivalence (Response to Weakness 2, Key Question 1)

Consider a simple convex optimization objective. For two different regularization methods, their common goal is to constrain the solution to a certain region (e.g., a flat region) to achieve better generalization. When both methods cause the solution to achieve optimal performance, we believe that the "constraining power" they impose on the objective to reach optimality is equivalent.

To use an analogy: when using a missile to strike a target, whether optical guidance or thermal guidance is used, the “guidance capability” applied to achieve the final hit is the same. Similarly, to approximate the same target function, we can use either Taylor expansion or wavelet functions; although the approaches differ, the approximation effect is the same.

Limitations (Response to Weakness 1, Limitations)

Although we empirically validate the relationship between hyperparameters under general settings, our theoretical construction of the equivalence and the theoretical analysis are both conducted under convex assumptions. Further theoretical investigation under non-convex assumptions remains necessary.

Additionally, our approach requires the final effects of the two different algorithms to be consistent, which introduces a second limitation: under appropriate assumptions, we can only obtain an upper bound on the generalization error, and this bound is not tight. Consequently, the relationship we establish between 
 and 
 is not a strict equality. We therefore use an approximate relationship in the paper (Eqs. (19) and (20)) and note that this theoretical study provides a heuristic foundation.

Applicability to Other Regularization Methods (Response to Key Question 1)

Our framework applies to regularization methods for which a generalization bound can be established via uniform stability analysis. For example, for dropout, we can impose a sparsification factor 
 on the parameter matrix to simulate dropped elements, and explicitly reflect the tightening effect of 
 in the generalization bound. An equivalence can then be built using that bound together with hyperparameters. However, this framework is not applicable to unquantifiable criteria such as early stopping.

Speculation on the Muon Optimizer (Response to Key Question 2)

When discussing optimizers in the framework of stability analysis, these arguments assume that gradients are bounded and the problem is convex. The interaction between the learning rate and weight decay in Muon (or similar momentum-based optimizers) may still exhibit a similar scaling pattern. This is because the gradient term captures the essence of Muon but is directly constrained by existing upper bounds, whereas weight decay acts on the parameters, so its effect is 
 of the final result (for details on the derivation, see the response to key Question 2 in Reviewer 6Su6). We conjecture that the core idea of balancing 
 and 
 can still provide insightful guidance, although rigorous generalization would require substantial additional analysis.

Weaknesses

Hyperparameters in SGDM need a little introduction before Theorem 4.11.
感谢您的建议，我们将在手稿中3.2节SGD算法的后面添加介绍SGDM的算法的内容。
We denote by SGDM the following updates：
, 
, where 
 and 
 denotes the stochastic gradient w.r.t. 
.

Remark 5.2, which claims that the stability of SWA is half of SGD without SWA, is not well-grounded. The difference in the order of a constant does not indicate a significant difference in the asymptotic sense. (Similarly, 
 is in the order of a constant, so it would be hard to conclude that momentum advances stability.) The comparison of two upper bounds (instead of upper vs lower bounds) makes the argument weaker.
正如我们在Remark 4.7中所讨论的，Zhang et al. (2022)[1]证明了SGD的界
是紧的，这意味着，即便是我们仅提供了SWA和SGDM的上界，若它们的上界小于SGD的结果，即可说明在现有的假设下，这两个算法提升了稳定性。

对于这句话“the stability of SWA is half of SGD”的表述，我们将在正式的手稿中修正为“SWA improves the generalization for SGD algorithm”。

The current figure on the interaction of 
 and 
 can only be interpreted qualitatively, as it is hard to see a clear diagonal patter. The experiment can be replaced with the following plot: Each curve of the plot represents the loss / accuracy of one weight decay parameter 
. The X-axis is 
, and the Y-axis is the test accuracy or the validation loss. If the scaling law of 
 is correct when 
 is fixed, then different curves should achieve the minimal loss / maximal accuracy at the same 
. If this plot is added and looks reasonable, I will raise my rating.
我们已经按照你的建议重新调整了这张图的呈现方式，具体效果展示在XX。

Key Questions For Authors:

How to intuitively interpret the relationship the dependence of 
 on 
 and 
?
事实上，基于稳定性分析这一框架，因子
由动量项的累积而产生。正如我们所看到的，当T趋于无穷，因子
趋于1，这意味着该泛化上界与SGD的上界趋于一致，具有几乎相同的泛化性能。这是因为当迭代步数足够大，处于训练后期的算法逐渐收敛至极小值，此时SGD和SGDM对于损失面上的探索行为是趋于一致的，因为此时的损失地形比较平缓，它们的梯度震荡都不会很大。相反，当T是一个较小的有限整数时，算法均处于训练的前期，损失地形中充满了尖锐的极小值，导致SGD的震荡明显，进而对噪声反映处较大的波动；而SGDM由于动量的累积效应，更新幅度被抑制，体现出良好的稳定性，因此产生了更小的泛化界。

At the end of Section 4, the paper claims that the introduction of 
 reduces the conditions of 
 under the optimal performance. I am confused about how it is inferred from the theorems.
本文基于稳定性分析这一框架衡量算法的泛化性能。基于该框架，算法的更新性质在凸假设下需满足非扩张性（Lemma4.4），以满足算法的现实行为。当SGD满足这一假设时，学习率需满足
（附录，证明A.1）。然而，当引入
后，SGD-WD仍需要首先满足非扩张假设，但此时学习率需满足
（附录，证明B.1，26式）)。这说明，当引入
后，需要更小的学习率以满足当前的假设。

[1] Stability of SGD: Tightness analysis and improved bounds. (Zhang et al., 2022)

Respond to Weakness 1

Thank you for the suggestion. We will add an introduction to the SGDM algorithm following the SGD description in Section 3.2 of the manuscript.
We denote by SGDM the following updates:
,
,
where 
 and 
 denotes the stochastic gradient with respect to 
.

Respond to Weakness 2

As discussed in Remark 4.7, Zhang et al. (2022) [1] proved that the bound 
 for SGD is tight. This implies that even though we only provide upper bounds for SWA and SGDM, if these upper bounds are smaller than those of SGD, it still demonstrates that the two algorithms improve stability under the given assumptions.

Regarding the statement “the stability of SWA is half of SGD”, we will revise it in the final manuscript to: “SWA improves the generalization performance over the SGD algorithm.”

[1] Stability of SGD: Tightness analysis and improved bounds. (Zhang et al., 2022)

Respond to Weakness 3

We have replotted the figure following your suggestion. XXXXXX

Respond to Key Questions 1

Under the stability analysis framework, the factor 
 arises from the accumulation of momentum. As we observe, when 
, 
, meaning that the generalization upper bound approaches that of SGD, leading to nearly identical generalization performance. This is because when the number of iterations is sufficiently large, the algorithm converges to a minimum in the later stage of training, and the exploration behavior of SGD and SGDM on the loss surface becomes similar; the loss landscape becomes relatively flat, and gradient fluctuations are small for both.

Conversely, when 
 is a small finite integer, the algorithm operates in the early stage of training, where the loss landscape contains sharp minima. In this case, SGD exhibits significant oscillations and is more sensitive to noise, while SGDM, due to the cumulative effect of momentum, suppresses the update magnitude and demonstrates better stability, resulting in a smaller generalization bound.

Respond to Key Questions 2

This paper evaluates generalization performance using stability. Within this framework, the update rule must satisfy non-expansiveness under convex assumptions to reflect realistic algorithm behavior (Lemma 4.4). For SGD, this requires 
 (see Appendix, Proof A.1). After introducing 
, SGD-WD must also satisfy the non-expansiveness assumption, but the required condition becomes 
 (see Appendix, Proof B.1, Eq. (26)). This indicates that the inclusion of 
 imposes a stricter upper bound on the learning rate to satisfy the assumption. Thus, the coupling between 
 and 
 is theoretically revealed, and the condition on 
 becomes more restrictive under optimal performance considerations.

Weakness:

Some existing works that are closely related to this paper are not cited in this paper, for instance, the generalization analysis of SGD with weight decay.
感谢你的建议，我们将在相关工作章节中继续补充关于权重衰减的泛化的现有研究的相关工作，例如包括但不限于以下文献：
[1]Weight decay with tailored Adam on scale-invariant weights for better generalization.(Jia et al., 2022)
[2]Towards Better Generalization: Weight Decay Induces Low-rank Bias for Neural Networks.(Chen et al., 2024)
[3]Three mechanisms of weight decay regularization.(Zhang et al., 2018)

The claims in this paper do not clearly specify the range to which they apply. For example, Question 1 in fact goes beyond the results presented in the paper. Also, this paper should include proper summaries that clearly state its actual contributions.
本文的贡献 (请看引言的贡献总结部分)

1 We derive and systematically compare the generalization bounds for SGD, SGD with Weight Decay (SGDWD), SGD with Momentum (SGDM), and SGD with Momentum and Weight Decay (SGDM-WD) based on
stability as shown in Table 1. Our theoretical analysis reveals that weight decay improves the generalization bound and enforces more stringent learning rate conditions for SGD and SGDM, respectively.
2 We derive a unified scaling rule, 
, grounded in the heuristic alignment of stability bounds. This law theoretically justifies empirical observations regarding the coupling of weight decay and training duration. This finding corroborates the empirical results reported in [1].
3 We explicitly quantify the relationship between explicit regularization 
 and implicit regularization B. We show that as B increases, 
 must be scaled up to maintain the total regularization strength.
4 We perform validation experiments, and the observation aligns with our finding that weight decay enhances generalization but also constrains the optimal learning rate. In addition, we validate the proposed hyperparameter scaling law across different architectures and task domains, including ResNet-18 for vision and Qwen3-0.6B for language modeling.
本文结论的适用范围
1 Although we empirically validate the relationship between hyperparameters under general settings, our theoretical construction of the equivalence and the theoretical analysis are both conducted under convex assumptions. Further theoretical investigation under non-convex assumptions remains necessary.
2 Our framework applies to regularization methods for which a generalization bound can be established via uniform stability analysis. For example, for dropout, we can impose a sparsification factor 
 on the parameter matrix to simulate dropped elements, and explicitly reflect the tightening effect of 
 in the generalization bound. An equivalence can then be built using that bound together with hyperparameters. However, this framework is not applicable to unquantifiable criteria such as early stopping.
关于问题1内容的说明
本文呈现SGD-WD的泛化界主要有两方面原因：
1 研究权重衰减在稳定性分析中如何收紧sgd的泛化界，这为后面的建立超参数的缩放规则提供依据。
2 基于该泛化界，通过正则等价性假设建立超参数间的解析关系。
[1]How to set AdamW’s weight decay as you scale model and dataset size.(X. Wang and L. Aitchison,2025)

The clarity of this paper needs to be improved. For example, the abstract could be more concise and should clearly summarize the main idea of the paper, leaving more details for the main text. The main text could include some concise summaries and takeaways. Some minor typos are listed in the Questions section.
感谢你的建议，我们将进一步概括并高亮文章中的总结和要点内容，使得文章思路更清晰。在这篇文章中，我们试图以问题为导向的写作方式呈现给读者。对于文章的整体结构：

以提出问题开始：如Q1，引入权重衰减对算法的泛化性能造成什么影响（Question 1）；Q2，权重衰减如何影响超参数？（Question 2）
以回答问题展开：如回答Q1，通过平稳性分析建立算法的泛化界（section 4）；回答Q2，通过正则等价性假设建立超参数之间的解析表达（section 5）。
以总结问题结束：如总结Q1，权重衰减提升算法的泛化性能（Answer to Q1）；总结Q2，权重衰减与学习率成反比，与Batch size成正比（Answer to Q2）。
Key Questions For Authors:

For the first question (Q1) mentioned in this paper, what about the theoretical analysis of the coupling for optimal λ in the non-convex setting? There are existing papers focusing on the generalization of SGD with weight decay in the non-convex setting, which are not mentioned in the paper.
在非凸设置下,Xie et al.(2020)[1]从收敛性的角度入手，提供了
的关系，以展示权重衰减可以加速收敛。Wang et al.(2025)[2]从理解AdamW算法的角度，提供了相同的观点，用于AdamW超参数的设置。然而，我们与上述文章的主要区别在于，本文以启发式的理论推导为依据，重点研究超参数间的协同缩放规则，并从实验上验证它们。
正如对weakness1的回复内容，我们将继续补充关于SGD-WD算法在非凸设置下的泛化性的研究。
[1] Understanding and scheduling weight decay. (Xie et al., 2020)
[2] How to set AdamW’s weight decay as you scale model and dataset size.(X. Wang and L. Aitchison,2025)

How to compare the influence of weight decay in SGDM and SGD? Do the results and proposed scaling rules in this paper also hold for other algorithms?
权重衰减对SGD和SGDM都有提升泛化的作用。

在一致稳定性分析框架下，
对SGD和SGDM的影响的是相同的。因为
的作用在这两种算法的分析中均体现在参数的收缩项中
中，而这部分是权重衰减的定义，也是SGD和SGDM共有的部分。SGD和SGDM算法的区别主要体现在梯度项中是否包含动量，但梯度项与
无关，所以导致
对SGD和SGDM的影响是一致的，均体现为泛化界的
 
。

本文更希望通过对比权重衰减对算法的影响（即对比SGD和SGD—WD，SGDM和SGDM—WD），以体现超参数
的引入，需要我们重新思考
对算法泛化性的影响和超参数之间的交互关系。

How does the quantitative scaling law incorporate the batch size 
 established? by empirical observation? How does it differ from that in existing papers?
正如本文在（19）式中所建立的解析关系，其中依赖迭代次数
。记迭代次数
和轮数
，我们根据天然存在的转换关系
(5.2节，第一句话)，其中
为样本量，
为批量大小。

（20）式展示了包含
的超参数缩放关系，它的建立依据本文提出的正则等价性这一启发式的假设，通过一致稳定性分析框架将两个不同正则算法的泛化界联系起来，并基于此建立了该缩放规则。

相比于基于实验观测的研究[1],我们的工作提供了启发性的理论解释，并在适当的假设下通过数学推导得到了理论的解析关系。相比于研究AdamW算法的工作[2]，其更希望借助该依赖关系来更好的理解AdamW算法本身，而本文更关注超参数间的协同关系，并尝试将该缩放关系推广到基础算法训练过程中的超参数设置中。

[1]How to Set the Batch Size for Large-Scale Pre-training?(Zhou et al., 2026)
[2]How to set AdamW’s weight decay as you scale model and dataset size.(X. Wang and L. Aitchison,2025)

Minor questions:

Line 212: “The Theorem 31 shows that”
Line 381: “Table 2 (in Appendix C)” Does it refer to the Table 2 on Page 7?
Line 311: “As shown in Eq. 19 and 20”
Line 299, 306: “Theorem 5.1”
Line 174&178: “SGD-DW”, Line 226: “SGDM-DW”?

感谢你细致而严谨的帮我们指出这些书写错误。
Line 212：The Theorem 31 改为 The Theorem 4.11
Line 381: Table 2 refers to the Table 2 on Page 7
Line 311: “As shown in Eq. 19 and 20” 改为 As shown in expressions (19) and (20)
Line 299, 306: Theorem 5.1 改为 Lemma 5.1
Line 174&178: “SGD-DW” 改为 SGD-WD
Line 226: “SGDM-DW” 改为 SGDM-WD

Weakness:

It is completely unclear to me why the generalization bounds of SWA and SGD with weight decay should match. There is no evidence or support provided in this work. The "regularization equivalence" should have been justified, so that it might indeed hold in practice. Without such a justification, the claims made in the paper sound like some heuristic that might give a correct answer. Given that all the theory in this work is based on the aforementioned hypothesis, it raises questions about the work's contributions.
关于本研究的动机：

在模型训练中，衰减的学习率通过减小探索的步长，精细化的让算法收敛到局部极小值。而权重衰减则从抑制参数的角度入手，让算法达到相同的目的。于是，很自然的想法是学习率和权重衰减在整个优化系统中是否存在同样的作用，这促使我们研究这两种超参数在训练中的相互影响。因此，亟待找到一种等价关系来建立起这两者间的关系，泛化性能就是一种可量化的潜在备选方案。

关于正则等价性的直觉：

我们考虑一个简单的凸优化目标，对于不同的两种正则方式，其共同的目标是通过约束条件，让函数的解落入特定的（例如平坦）区域，以获得更好的泛化性能。当它们使得函数的解都落在了最优性能处时，我们相信这两种不同的正则方式对函数达到最优目标的约束力是相同的。

举一个通俗的例子：当我们使用导弹打击一个目标时，无论采用光学制导还是热制导，这两种制导方式对最终达成击中目标所施加的“引导能力”是一样的。再例如：为了逼近同一个目标函数，我们既可以使用泰勒展式，也可以使用小波函数，尽管它们的逼近方式不同，但逼近的效果是一样的。

为什么匹配SWA和SGD-WD的泛化界：

在机器学习中，权重衰减是一种典型的显式正则项，其通过约束优化目标的参数，以获得良好的泛化性能。同时，目前较为流行的SWA算法通过引导模型到达更平坦的区域以获得良好的泛化性能，可以视为一种在优化目标解空间中的隐式正则方法。因此，我们在简单的凸优化目标假设下，通过相同的泛化能力作为等价关系，将这两种具有不同约束行为的算法联系到一起。

Equalising the bounds for two different algorithms can also be done when the generalization bounds are tight. There is no discussion around the tightness of the derivations. If one bound is tight, but the second one is not, then equalising them is meaningless.
正如你所关心的，我们在适当的假设下仅能得到算法的上界，而并非是上确界，这使得
和
之间建立的并不是严格的等式关系。于是，我们在文中使用近似的的关系来表示（表达式19和表达式20），同时我们称该理论研究提供了一种启发式的理论依据。

I also find the way the authors write to be misleading and confusing. The authors claim that weight decay improves generalization bounds, but later equate it to the generalization bound of SWA, which is the same as that of vanilla SGD.
这篇文章通过问题导向的写作方式，从泛化性能的角度入手，探讨了超参数
引入，对算法及其他超参数的影响，旨在为超参数在实验中的设置及协同缩放提供启发性的理解。文章第四章，从理论的视角研究了
对泛化界的收紧作用，回答为什么权重衰减可以提升泛化界。第五章，通过正则等价性假设研究了超参数之间的相互影响，其中该等价性关系要求SWA与SGD-WD的泛化界相等，以建立
与其他超参数的显式关系。

最后，SWA的泛化界是
[2,3,4]，而SGD的泛化界是
[2]，且该结果是紧的[1]。这充分说明了SWA具有更小的泛化界和更好的泛化性能。我们引入SWA的目的主要是将其作为一种可以改善算法泛化性能的隐式正则方法，并以此来建立等价关系。

Some of the theorems were already obtained in the literature. However, the need for them in the paper is questionable. For example, I do not see the reason why the generalization bound for vanilla SGD [Hardt et al., 2016] is provided if it is never used later. The same question regarding the generalization bound for SGD with momentum. This confuses the reader, in my view.
我们引用了SGD的理论结果，并不是想作为我们的理论贡献之一，而是为了在与SGD-WD、SWA等算法的泛化界比较中更清晰的呈现出结果上的差异。而引入动量SGD原因之一是在动量背景下，引入超参数
后，
仍然影响动量SGD的泛化性质和算法内的相关超参数；原因之二是动量SGD作为带有动量项算法的代表，我们希望以此来展示本文的研究结果具有一定的推广能力。

The paper is badly written. There are many typos and potential mistakes. Therefore, the paper might be LLM-written or heavily rely on it. Just to mention some of them:
5.1 
 is used in the abstract, but never defined. This makes the reading less clear. Maybe 
 is a token budget or some other quantity.

代表算法的迭代步数。

5.2 
 depends on the iteration counter 
, but is always used as a constant in the derivations, which is not fully true. The authors should be more accurate with this quantity. I encourage them to redo the calculations and take into account that 
 depends on 
 in the bounds. This might bring an additional parameter to the rule 
. The authors also refer to this constant as "small", with respect to what?

5.3 The definition of SWA is vague. What is 
? I expected the averaging of all iterates, not only 
 and 
.

这是一个笔误，感谢你指出这一点。事实上，等式8应该被写为
 
，其中
代表第
步迭代对应的模型参数。正如你所期望的，
代表所有
个参数平均。

5.4 Typos: Theorem 31 does not exist in the paper (line 213), Theorem 5.1 does not exist (line 299), Lemma 3.6 in [Hardt et al., 2016] does not exist (line 197),

这是一个tex中的格式错误，感谢你指出这一点。我们已经在手稿中重新修正了它。

5.5 Definition 3.1 is incorrect from a mathematical point of view. How can 
 and 
 come with and? Similar concerns about Definition 3.2

Unknown environment 'definition'

Unknown environment 'definition'

5.6 In (31), the authors should bound the terms 
 and 
 separately (although the bounds are the same) to make the proof fully correct. A similar applies to (40).

感谢你指出这一点，我们将在手稿中补充关于
的证明过程，并在后文中出现的位置，标注它们的出处。

5.7 In line 675 (proof of Theorem 4.11), the authors assume that some inequality holds which involves 
 and 
. The authors claim that it leads to a restriction on the learning rate of the form 
. I do not see how it is done. I encourage the authors to provide more details around this part of the proof. Without this clarification, I currently consider the proof to be wrong. A similar applies to line 731 (proof of Theorem 4.13).

关于675行和731行中对于学习率的限制，这源自于算法在凸函数假设下满足非扩张性质，具体如下：
1.在一致稳定性分析的框架下，要建立算法的泛化界，需要满足两个基本假设：（1）凸假设下，算法满足非扩张性质(Lemma 4.4)；（2）非凸假设下，算法满足
-扩张性质。
2.事实上，675行和731行中对于学习率的限制来自于使得非扩张性质“
”成立，所推导出的学习率需要满足的条件。
也就是说，(35)式中
 
 
成立，使得基本假设非扩张性质成立，而这构成了学习率索要满足的条件。

Experiments are not averaged over several random seeds to remove the effect of the noise.
我们增加了XXX

Empirical claims regarding the interplay between 
 in practice are based on small number of runs. Therefore, I do not find them convincing.
我们额外在XXX

In general, the experiments section is badly written; it is hard to understand which algorithms' performance is reported, what the setting is, etc. I encourage the authors to add more details to improve readability. I also think the provided empirical evidence is insufficient to support the claims. The authors should add more training configurations and average over several runs to be able to say that the theory is predictive.
我们XXXX

Key Questions For Authors:

Which algorithm is reported in Figure 1?
图1展示了最优学习率下，SGD、SGD-WD、SGDM和SGDM-WD的最优性能。旨在展示：

引入
，模型的最优精度会提升。（SGD VS SGD-WD，SGDM VS SGDM-WD）
引入
，取得最优精度的模型，对应学习率会减小。
Can the authors provide details on why inequalities in lines 675 and 731 hold?
请看Weakness5.7的回复
Limitations:

The authors mention that the current theory does not cover non-convex functions.
In my view, there are more limitations that should have been mentioned:
由于空间原因，请看回复审稿人oX4Y的关于局限性的回答。

Why equivalenece hypothesis holds
请看Weakness1的回复

More sophisticated empirical study to support theoretical claims
请看Weakness6、7、8的回复

Response to W1.

Regarding the motivation for this study and why the regular equivalence holds, due to space constraints, please refer to the response to Reviewer oX4Y about Intuition.
Why match the generalization bounds of SWA and SGD-WD?
In machine learning, weight decay is a typical explicit regularizer that constrains parameters to achieve good generalization. SWA, a popular method, guides the model to flatter regions, which can be viewed as an implicit regularizer on the solution space. Under a simple convex objective, we use the same generalization capability as an equivalence to connect these two algorithms with different constraint behaviors.

W2.
As you rightly pointed out, at present we can only obtain an upper bound on the generalization error, and it is not tight. Consequently, the relationship we establish between 
 and 
 is not a strict equality. We therefore use an approximate relationship in the paper (expressions (19) and (20)) and note that this theoretical study provides a heuristic foundation.

W3.

This paper adopts a problem-oriented writing style. Section 4 theoretically examines how 
 tightens the generalization bound, answering why weight decay improves generalization. Section 5 uses the regularization equivalence hypothesis to study the interplay among hyperparameters, requiring that the generalization bounds of SWA and SGD-WD be equated to establish an explicit relationship between 
 and other hyperparameters.
Finally, the generalization bound of SWA is 
 [1,2], while that of SGD is 
 [3], and the latter is tight [4]. This clearly shows that SWA has a smaller generalization bound and better generalization performance. We introduce SWA primarily as an implicit regularization method that improves generalization, serving as a basis for establishing the equivalence.
[1] Stability analysis and generalization bounds of adversarial training. (Xiao et al., 2022)

[2] Generalization analysis of stochastic weight averaging with general sampling. (Wang et al., 2024)

[3] Train faster, generalize better: Stability of stochastic gradient descent. (Hardt et al., 2016)

[4] Stability of SGD: Tightness analysis and improved bounds. (Zhang et al., 2022)

W4.
We cite the theoretical results for SGD not as a claimed contribution, but to clearly highlight differences when comparing with the bounds of SGD-WD, SWA, and others. We introduce momentum SGD for two reasons: (1) to show that 
 still affects generalization and hyperparameters in the presence of momentum; (2) to demonstrate that our findings generalize to algorithms with momentum.

W5

5.1 
 represents the number of iterations.
5.2 Factor 
 arises from the accumulation of momentum and is already in its simplest form in the current analysis. For a real number of training steps T, 
 must be strictly less than 1, since we cannot allow the algorithm to run indefinitely.
5.3 This is a typo. Eq. (8) should be written as 
 
, where 
 is the model parameter at iteration 
. As you expected, 
 represents the average of all 
 parameters.
5.4 These are formatting errors in the TeX source. We have corrected them in the manuscript.
5.5 We say that 
 is uniformly Lipschitz continuous in 
 if there exists a constant 
 such that for all 
 and all 
, |F(u, z) - F(v, z)| \leq G |u - v|, where 
 is the Euclidean norm.
5.6 Thank you. We will add the separate bounding for 
 in the proof and indicate its derivation.
5.7 The learning rate condition arises from the non-expansiveness property under convex assumptions (Lemma 4.4). Specifically, for the inequality 
 
 
 to hold (which ensures non-expansiveness), we derive the condition 
.
W6. We have addressed this concern by conducting extensive multi-seed experiments. All results are now averaged over 2 random seeds (seed=42, seed=123), reported as mean ± half-range. Across 3 architectures (ResNet-18, VGG-16, ResNet-50), we performed a total of approximately **640 independent training runs**. In the stable hyperparameter region, the half-range is consistently below 0.5%, confirming that our conclusions are robust to random seed variation. For ResNet-18, we further performed 4 independent runs (2 seeds × 2 runs per seed, 332 total runs), where peak accuracies across runs differ by less than 0.22%. Full details are provided in the supplementary reports (resnet18\_4run\_report, phase2\_3\_report, resnet50\_report).

W7. We have substantially expanded the empirical evidence. Beyond the original ResNet-18 experiments, we now validate the three theoretical predictions on **two additional architectures**: VGG-16 (no residual connections, 14.8M params) and ResNet-50 (deeper residual network, 23.5M params), each with 2 random seeds. Key findings across all 3 architectures and ~640 runs:
- **Stability boundary ordering** (Exp 1): SGD+WD exhibits the widest stable LR range, SGDM+WD the tightest — consistent across all 3 architectures. Both SGD and SGDM+WD have η\*=0.1, while SGD+WD peaks at η\*=0.5–1.0. VGG-16 shows a tighter boundary than ResNets (consistent with a larger Lipschitz constant L due to no skip connections).
- **η–λ inverse relationship** (Exp 2): The anti-diagonal heatmap pattern is preserved across all architectures. At η=0.2, all 8 independent runs (R18×4 + VGG×2 + R50×2) unanimously select λ\*=5e-4. The optimal λ\* ranges match for all 5 tested learning rates across architectures.
- **Batch size scaling** (Exp 3): Under the linear LR scaling rule, optimal λ\* ∈ [5e-4, 1e-3] for all batch sizes across all architectures. At B=64 and B=512, all 8 runs unanimously select λ\*=1e-3.

W8. We appreciate the suggestion regarding the clarity of the experimental section. Due to the page limit of the current submission, we are unable to revise the main text at this stage. We will improve the writing in the camera-ready version, where each experiment will clearly specify the optimizer variant (SGD / SGD+WD / SGDM+WD), the hyperparameter grid, and the evaluation metric. More detailed descriptions of the experimental setup and results will be provided in the appendix.

Regarding the concern about insufficient configurations, following your suggestion, we have now conducted additional experiments on **three architectures** (ResNet-18, VGG-16, ResNet-50) with 2 random seeds each (~640 total runs). The expanded configurations include:
- **Exp 1** (24 configs × 3 architectures × 2 seeds): Accuracy-vs-LR curves for SGD, SGD+WD, and SGDM+WD.
- **Exp 2** (35 configs × 3 architectures × 2 seeds): Full η–λ heatmaps with row-wise optimal λ\* marked.
- **Exp 3** (24 configs × 3 architectures × 2 seeds): Batch size scaling tables with the linear LR rule applied.
All three theoretical predictions are consistently confirmed across architecturally diverse models (residual vs. non-residual, shallow vs. deep), providing strong evidence that the theory is predictive beyond a single architecture and seed. Full results are included in the appendix.

KQ1 Figure 1 shows the optimal performance of SGD, SGD-WD, SGDM, and SGDM-WD under the optimal learning rate. It demonstrates two points:

Introducing 
 improves the optimal accuracy (SGD vs. SGD-WD, SGDM vs. SGDM-WD).
Introducing 
 reduces the learning rate that achieves optimal accuracy.
KQ2 Please refer to our response to Weakness 5.7.

Limitations Due to space constraints, please

refer to our response to Reviewer oX4Y regarding limitations
see W1
see W6–8