OpenReview
.net
Search articles, authors and reviews...
Notifications40
Activity
Tasks
Zhiyuan Yu 
back arrowGo to NeurIPS 2026 Conference homepage
Rethinking Weight Decay: A Stability-Based View on Hyperparameters Coupling
Download PDF
Peng Wang, Zhiyuan Yu, Shengchao Hu, Shi Fu, Enneng Yang, Guodong Zheng, Qiyang Zhou, Qixin Zhang, Yan Sun, Shiwei Liu, Li Shen 
03 May 2026 (modified: 28 May 2026)
NeurIPS 2026 Conference Submission
Conference, Senior Area Chairs, Area Chairs, Reviewers, Authors
Revisions
CC BY 4.0
Abstract:
Weight decay introduces a hyperparameter 
 that improves stability but disrupts the balance among optimization hyperparameters (learning rate 
, batch size 
), making optimal tuning substantially more difficult. While prior work has focused on scaling laws for 
 and 
, 
 is typically treated as a fixed constant, overlooking its impact on optimization dynamics. In this work, we systematically study how 
 affects the stability of the optimization system, as well as its coupling with 
 and 
. Using a stability-based analysis, we derive stability bounds for basic optimizers such as stochastic gradient descent (SGD) and SGD with momentum (SGDM), showing that weight decay can tighten upper bounds but also impose stricter constraints on 
. We further propose a heuristic regularization equivalence hypothesis, under which we derive the analytic coupling of hyperparameters (
 
), consistent with recent empirical results that interpret AdamW as an exponential moving average (EMA). Extending this analysis to the batch size, we show that the optimal product 
 scales with 
. Experiments across diverse architectures and tasks, from ResNet-18, ResNet-50, and VGG-16 on vision tasks to Qwen3-0.6B on language modeling, validate our theoretical and heuristic findings.

Checklist Confirmation: I confirm that I have included a paper checklist in the paper PDF.
Supplementary Material:  zip
Responsible Reviewing: We acknowledge the responsible reviewing obligations as authors.
Primary Area: Deep learning advancements (e.g., architectures, optimizers, representation learning)
Secondary Area: Theory (e.g., learning theory, theory of deep learning, algorithmic game theory)
Contribution Type: Negative Results: The main contribution is in understanding a negative result. (The significance and originality bar for these contributions is high.)
Academic Integrity: I acknowledge that I have read the NeurIPS Handbook and commit to adhering to all policies in the Handbook (https://neurips.cc/Conferences/2026/MainTrackHandbook), the NeurIPS Code of Conduct and the NeurIPS Academic Integrity Policy.
LLM Usage: Editing (e.g., grammar, spelling, word choice)
Declaration: I confirm that the above information is accurate.
Reviewer Nomination:  Peng Wang
Submission Number: 17464
Discussion
Filter by reply type...
Filter by author...
Search keywords...

Sort: Newest First
10 / 10 replies shown
Add:
Meta Review of Submission17464 by Area Chair vXFZ
Meta Reviewby Area Chair vXFZ21 Jul 2026, 17:04 (modified: 24 Jul 2026, 02:00)Senior Area Chairs, Area Chairs, Authors, Reviewers Submitted, Program Chairs, Area Chair vXFZRevisions
Metareview:
This paper proposes an empirical study of weight decay as a regularization parameter for Transformers. It also introduces several empirical laws and attempts to validate them on standard benchmarks.

Strengths

The paper proposes a method for selecting the weight-decay parameter instead of treating it as a fixed quantity.
Important Weaknesses, that need to be carefully treated:

I agree with reviewer SijV that, although the paper does not put much emphasis on this point, it essentially recovers existing results through a different approach. The paper feels somewhat re-heated, and I also agree with reviewer eC8H that there might be even more related results that are not sufficiently acknowledged.
I am not sure that the theoretical analysis actually supports the experiments, as it relies strongly on assumptions such as 
-smoothness, which may not reflect well the practical setting considered in the paper.
The paper seems motivated by practical considerations, but I do not see any convincing argument explaining why a practitioner should use the proposed method rather than simpler existing approaches.
The discussion about momentum is not fully complete. The paper would have benefited from contrasting its results with concurrent work, but this opportunity is currently missed.
Add:
Official Review of Submission17464 by Reviewer eC8H
Official Reviewby Reviewer eC8H26 Jun 2026, 18:30 (modified: 29 Jul 2026, 02:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer eC8HRevisions
Summary:
The study of weight decay hyperparameter 
 is often ignored in the optimization literature. Practitioners either tune this hyperparameter using expensive grid search or set this as a fixed default value e.g. 
. To address this gap, the authors study the effect of weight decay 
 in optimization dynamics. This paper derives a stability bound for SGD and SGDM showing that weight decay can tighten upper bounds and also impose stricter constraints on the learning rate 
. The authors also propose a heuristic regularization-equivalence hypothesis. The authors further argue that the optimal product of regularization hyperparameter and step size i.e. 
 should scale as the batch size 
. They provide experiments to validate their claim.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths

This paper studies an important aspect of modern optimization algorithms, which is often ignored in the literature.
The paper suggests a method for choosing the 
 instead of setting it as a fixed quantity.
They provide theoretical bounds on the generalization error of SGD and SGDM in the presence of weight decay.
Experiments to verify the proposed claim.
Weaknesses:

The paper suggests that the product of the weight decay parameter and step size should scale as the inverse of the total number of iterations. This implies that it is enough to tune the step size according to the number of iterations.

It is not clear to me why we should care about uniform stability. How does this notion translate to better generalization?

Quality: 3: good
Clarity: 3: good
Significance: 4: excellent
Originality: 4: excellent
Questions:
How is the generalization performance of SGD and SGDM with this analysis? Is SGDM better at generalization?
What is the impact of the momentum parameter on generalization?
Limitations:
To my understanding, the primary limitation is the fact that the product of the weight decay parameter and step size should scale as the inverse of the total number of iterations. This means it is enough to just tune the step size while keeping the weight decay constant. However, the paper starts with the motivation of choosing weight decay correctly.

Rating: 4: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
NA

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Rebuttal by Authors
Rebuttalby Authors (Qixin Zhang, Peng Wang, Qiyang Zhou, Shi Fu, +7 more)28 Jul 2026, 10:48 (modified: 28 Jul 2026, 21:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Response to Weakness 1 and the stated limitation
We thank the reviewer for raising this important point. We agree that our theoretical result constrains the product of the learning rate and weight decay. However, this does not imply that weight decay is redundant or that one can always fix 
 and tune only 
.

For decoupled weight decay, the update takes the form 
. Although the contraction term depends on the product (\eta_t\lambda_t), the learning rate (\eta_t) also scales the gradient update (g_t), whereas (\lambda_t) only controls parameter contraction. Therefore, changing (\eta_t) to compensate for a fixed (\lambda_t) simultaneously changes the optimization dynamics, including the effective gradient step and stochastic noise scale. The two hyperparameters are thus coupled but not interchangeable.

To examine this issue empirically, we conducted a dense 
 learning-rate–weight-decay grid search on ResNet-18/CIFAR-100 with a fixed training budget of 100 epochs, batch size 128, momentum 0.9, and two random seeds. For each learning rate, we define the accuracy envelope 
.

The results provide three relevant observations.

First, the weight decay that maximizes accuracy changes systematically with the learning rate. As (\eta) increases from (0.005) to (0.5), the optimal weight decay decreases from 
 to 
: 
.

Thus, the optimal solutions do not lie on a single fixed-
 row. A log-log regression of the grid maximizers gives a slope of approximately (-0.76), showing a clear inverse relationship between the two hyperparameters. In contrast, a redundant weight-decay parameter would produce no systematic dependence of 
 on 
.

Second, fixing a commonly used weight decay can lead to a non-negligible loss. For example, with the standard choice 
, the best accuracy over all learning rates is (77.31%), compared with the joint optimum of (78.50%). Moreover, relative to the accuracy envelope, this fixed choice incurs an average gap of (1.44) percentage points and a maximum gap of (3.76) points across the tested learning rates. Therefore, tuning only the learning rate does not allow a prespecified weight decay to consistently recover the best achievable performance.

Third, after optimizing 
 for each 
, the accuracy envelope remains relatively stable over a 100-fold range of learning rates, varying from (75.75%) to (78.50%). This indicates that coupling weight decay with the learning rate improves robustness to the learning-rate choice. The purpose of our theory is precisely to characterize this coupling, rather than to suggest that either hyperparameter can be removed.

We will clarify this distinction in the revision and add the complete grid, the accuracy-envelope analysis, and the corresponding figure to the main text. In particular, we will revise our discussion to emphasize that the theoretical scaling law constrains the contraction strength 
, while the individual values of 
 and 
 remain important because they play different roles in the optimization dynamics.

0.005	0.01	0.02	0.05	0.1	0.2	0.3	0.5
73.20	73.78	73.81	74.20	74.18	74.69	74.89	75.34
73.44	74.22	74.56	75.31	75.43	75.69	75.70	75.97
 (common default)	73.80	75.15	75.96	77.09	77.31	76.36	75.75	74.47
74.75	76.23	77.39	78.26	77.47	75.53	73.66	67.47
75.66	77.02	78.16	78.50	76.22	72.41	63.16	38.19
77.04	77.67	77.73	75.78	68.47	41.05	21.97	2.50
77.56	77.58	76.65	69.22	44.20	5.42	2.45	1.66
77.25	75.81	69.88	43.62	3.77	2.06	1.30	1.35
55.65	17.00	5.32	2.69	2.31	1.00	1.00	1.00
envelope 
77.56	77.67	78.16	78.50	77.47	76.36	75.75	75.97
W2: Why does uniform stability matter for generalization?
It is not clear to me why we should care about uniform stability. How does this notion translate to better generalization?

Response. We thank the reviewer for raising this important question. Our objective is to gain a better understanding, from the perspective of generalization, of the coupling among hyperparameters after introducing weight decay in order to preserve the generalization properties of the entire dynamical system.

Uniform stability measures the sensitivity of a learning algorithm to replacing one example in its training set. It is relevant here because it is directly connected to generalization. Definition 4.1 and Theorem 4.2 (Hardt et al., 2016, Theorem 2.2) show that once an optimizer is shown to be 
-uniformly stable, the same 
 upper-bounds its expected empirical-to-population risk gap. Therefore, we can quantitatively characterize the algorithm’s generalization ability based on stability. This is the precise sense in which stability translates to generalization in our analysis.

Weight decay changes how these perturbations accumulate. Without weight decay, the perturbation recursion is additive, yielding the familiar SGD dependence 
 
 With weight decay, the recursion contains the contraction factor 
: 
 
 Summing the resulting geometric series gives 
 
, which no longer grows linearly with (T). This contraction mechanism is the main theoretical result: weight decay limits the accumulation of data perturbations across optimization steps.

Q1: SGD versus SGDM
How is the generalization performance of SGD and SGDM with this analysis? Is SGDM better at generalization?

Response. Our analysis does not establish that SGDM generalizes better than SGD. Although the SGDM stability bound contains a momentum-dependent factor 
, it does not imply a universal generalization advantage for SGDM. The two bounds are derived under different assumptions: the SGDM result requires strong convexity, whereas the SGD result assumes only convexity. Their sufficient learning-rate conditions also differ.

Our experiments likewise do not provide a controlled comparison of momentum. They were designed to examine how weight decay changes stability and hyperparameter coupling within each base optimizer. Because the reported SGD and SGDM configurations do not hold all other hyperparameters fixed, any observed difference cannot be attributed solely to momentum. We therefore conclude only that the contraction mechanism induced by weight decay extends from SGD to SGDM. Neither our theory nor the current experiments demonstrate that SGDM generally has better generalization performance than SGD.

Q2: Impact of the momentum coefficient
What is the impact of the momentum parameter on generalization?

Response. In our generalization analysis, the role of momentum is represented by the factor 
 in the theoretical framework. This factor describes how the exponential averaging of past gradients (i.e., 
)changes the propagation of a one-sample perturbation. Once the momentum coefficient 
 is fixed, the factor t is always less than 1; in the theoretical framework, this does indeed reduce the generalization upper bound. However, it is important to note that the results based on momentum hold under the assumption of strong convexity.

Second, the momentum changes the admissible learning-rate region by satisfying the non-expansiveness condition. A value of 
 that smooths short-term gradient perturbations can simultaneously require a different 
 to remain in the stable regime.

Add:
Official Review of Submission17464 by Reviewer SijV
Official Reviewby Reviewer SijV26 Jun 2026, 01:12 (modified: 29 Jul 2026, 02:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer SijVRevisions
Summary:
The paper studies how weight decay couples with learning rate and batch size through a uniform-stability lens. It derives generalization bounds for SGD/SGDM with and without WD, arguing WD tightens the bound from O(T/n) to O(1/n) at the cost of a stricter η constraint. It then proposes a heuristic "Regularization Equivalence" between SGD-WD and SWA, yielding the coupling lambda ~= 1/(lr*T).

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths:

The paper's general presentation is done well; the story, figures, sections etc. are very clear.
The authors do reasonable efforts for experiments and robustness.
I generally cannot comment on the theory as it is out of my focus area / capacity, but the use of uniform stability to rederive bounds with weight decay seem like a clean step and result.
Weaknesses:

Framing: The paper misses some important literature on weight decay in modern optimization for deep learning: in modern nets WD's main role is in optimization dynamics, not capacity control. Kosson et al. (ICML 2024, arXiv:2305.17212 and ICLR 2026 arXiv:2510.19093) show WD balances the effective learning rate, since it prevents uncontrolled weight norm growth and lets weight vectors rotate at controlled speeds in an equilibrium state, ie. stabilizing update dynamics, with the optimum keeping ηλ ≈ const.
I do not think the experiments on Qwen finetuning are meaningful. In my experience (unless I'm missing something) WD is rarely a load-bearing hyperparameter in standard LoRA practice (defaults are typically zero). To me it's therefore unclear that the η–λ coupling demonstrated here reflects something where WD is doing meaningful work at the magnitudes swept. Unfortunately, this would require proper GPT style full training runs.
Novelty: without putting a lot of emphasis on this, the paper essentially recovers existing results with a different approach. (This is of course interesting in its own right, but has to nonetheless be mentioned). Also, there's plenty of reused results, and it is not clear to me what exactly is a entirely new result.
Minor inconsistencies: The conclusion states lambda ≈ 2/lrT (line 283); the body says 1/lrT (Eq. 16). Plus minor typos ("bitch size", ...).
Quality: 3: good
Clarity: 3: good
Significance: 2: not good
Originality: 2: not good
Questions:
This is perhaps a very direct question, but for me to understand better: how much of this theory (in your honest view) entirely novel and how much of it is recombination of existing results?
It seems there are many different versions of WD relationships floating around, and sometimes it's proportionality, sometimes a precise law. Could you comment on this?
Limitations:
yes

Rating: 3: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
none

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Rebuttal by Authors
Rebuttalby Authors (Qixin Zhang, Peng Wang, Qiyang Zhou, Shi Fu, +7 more)28 Jul 2026, 14:33 (modified: 28 Jul 2026, 21:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
SijV-W1: Missing literature
Reviewer comment. Framing: The paper misses .... Kosson et al. (ICML 2024, arXiv:2305.17212 and ICLR 2026 arXiv:2510.19093) ....

Response. Thank you for pointing out these two related works; we will include them in the manuscript. Additionally, please refer to our response to Weakness 3 for a discussion of how this paper differs from those two papers.

SijV-W2: Does the Qwen-LoRA experiment meaningfully test weight decay?
Reviewer comment. I do not think the experiments on Qwen finetuning are meaningful. In my experience (unless I'm missing something) WD is rarely a load-bearing hyperparameter in standard LoRA practice (defaults are typically zero). To me it's therefore unclear that the (\eta)-(\lambda) coupling demonstrated here reflects something where WD is doing meaningful work at the magnitudes swept. Unfortunately, this would require proper GPT style full training runs.

Response. We thank the reviewer for raising this important concern. We agree with the phenomenon you observed in your experiment and that weight decay is not the dominant factor in certain tasks.

Nevertheless, weight decay remains nontrivial when only the LoRA parameters are optimized. Let the effective adapter be (\Delta W=sBA). Under the decay-only AdamW update, 
, and therefore 
. Thus, (\eta\lambda) directly controls the dynamics of the effective LoRA update even though the pretrained backbone is frozen.

This effect is also consistent with recent LoRA-specific theory. Kim et al. [1] show that weight decay biases LoRA optimization toward low-rank, small-magnitude global solutions, while Jacobs et al. [2] demonstrate that explicit regularization can have a persistent effect on LoRA’s implicit bias. More broadly, results for factorized matrix optimization connect factor-wise weight decay to a nuclear-norm-type bias [3].

We will also narrow our claim: Exp. 4 demonstrates that the coupling occurs in a controlled Qwen3-0.6B LoRA setting; it is not intended to claim that every standard LoRA recipe must use nonzero weight decay.

[1] LoRA Training Provably Converges to a Low-Rank Global Minimum Or It Fails Loudly (But it Probably Won’t Fail), 2025 ICML.
[2] Mirror, Mirror of the Flow: How Does Regularization Shape Implicit Bias? 2025 ICML.
[3] Weight decay induces low-rank attention layers, 2024 NeurIPS.
SijV-Q1 and W3: How much is genuinely new and how much is recombination?
Reviewer question. This is perhaps a very direct question, but for me to understand better: how much of this theory (in your honest view) is entirely novel and how much of it is recombination of existing results?

Response. Our honest assessment is that the work is primarily a new synthesis and derivation from a generalization perspective. The paper's own contribution has three layers:

A novel yet heuristic approach: the “regularization equivalence” proposal, as illustrated by the two examples given in the introduction. This proposal aligns the stability metrics of SGD-WD and SWA. It provides a generalization-driven approach for exploring the coupled effects of hyperparameters.
Theoretical Derivation Based on Stability: Although analyses of SGD [1] and SGDM [2,3] under stability have already been conducted, the results under weight decay (SGD-WD and SGDM-WD) constitute the core theoretical contribution of this paper, particularly regarding the impact of introducing weight decay on the algorithms’ generalization bounds and learning rate conditions.
New empirical evidence in this submission: the CIFAR sweeps across several architectures and the exploratory LoRA study. These show consistency with a coupled band.
The distinction from the closest related work can be summarized briefly:

Compared with [1,2,3], which study focuses solely on the algorithm's generalization performance using stability, without examining the impact of weight decay on the interactions between the algorithm and other hyperparameters.

Compared with [4], which studies weight norms and rotational equilibrium in scale-invariant neural networks, , whereas we examined the effects of weight decay on generalization performance and other hyperparameters based on stability. The two analyses share the factor 
, but concern different notions of stability.

Compared with [5], which studies how weight decay stabilizes representation updates and enables learning-rate transfer across model widths. The theoretical basis of 
P is grounded in the limit behavior of gradients controlled by the tensor program, whereas we rely on stability analysis to examine the stability of the algorithm’s output following perturbations.

Compared with [6], which interprets AdamW as an EMA and transfers its decay timescale across model and dataset scales. In contrast, we analyze SGD/SGDM and investigate the coupling effects among hyperparameters through “regularization equivalence.” Our results are not limited to AdamW’s memory window but instead explore the basic algorithm from a stability perspective.

[1] Train faster, generalize better, Stability of stochastic gradient descent, ICML 2016.
[2] On the Generalization of Stochastic Gradient Descent with Momentum, JMLR 2024.
[3] Stochastic Gradient Descent with Momentum is Algorithmically Stable, Arxiv 2026.
[4] How Weight Decay Balances Learning Across Neural Networks, ICML 2024.
[5] Weight Decay may matter more than muP for Learning Rate Transfer in Practice, ICLR 2026.
[6] How to set AdamW’s weight decay as you scale model and dataset size, ICML 2025.
SijV-W4 (Part I) and Q2: Inconsistent constants
Reviewer question. It seems there are many different versions of WD relationships floating around, and sometimes it's proportionality, sometimes a precise law. Could you comment on this?

Response. Thank you for point this. We will clarify this hierarchy as follows.

1. Controlled approximations
For decoupled WD, 
 is an exact algorithm update. However, before match the generalization bounds of SGD-WD and SWA, we need to account for two major approximations factors:

Our analysis uses uniform stability to derive an upper bound on the expected generalization gap. This step may introduce slack because the stability quantity only upper-bounds, rather than exactly characterizes, the generalization error.
The resulting bounds contain constants that depend on the assumptions and proof technique.
Therefore, in the above content, we use upper-bound control.

2. Heuristic or empirical scaling statements
Our Eq. (16), 
 
, is obtained by matching the two generalization upper bounds under the proposed Regularization Equivalence hypothesis. It should therefore be interpreted as a heuristic proportionality with an unspecified constant 
, rather than as a precise or universal optimality law. Empirical claims such as 
 likewise describe a trend in a specified scaling regime; their constants and sometimes their exponents can depend on model size, data scale, optimizer, and schedule.

3. Statement in line 283
Although directly equating the displayed bounds may produce a numerical factor such as (2), this factor should not be interpreted as universal because the comparison is based on upper bounds and a heuristic equivalence hypothesis. We will revise line 283 and unify it as 
 
.

SijV-W4 (Part II) Typographical errors
Thank you for point this. We will correct “bitch size” to “batch size”.

Add:
Official Comment by Reviewer SijV
Official Commentby Reviewer SijV03 Aug 2026, 17:43Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thanks a lot for the response and the clarifications.

Regarding LoRA: of course, weight decay does have a direct influence on the effective update step; this was never questioned. Instead, my point was more directed towards the claim of 'realistic LLM settings' as described in the paper. For instance, what if weight decay was turned off on the LoRA experiments, and LR tuned? How much would the accuracy differ?

Regarding the related work and the context within the broader literature, it's clear that there are differences. The important point is that they all give the same (or almost the same) takeaway, irrelevant of how they were achieved. Also, for clarification: [4] is not at all limited to scale invariant networks, and [5] is in fact showing that the muP assumptions are overwritten by weight decay (so they way you describe these works is incorrect); [6] also looks at the interaction with batch size, which is the important overlap. To be clear, I am not an author of any of these papers.

Together with the AC's and other reviewers' comments, I am still of the opinion that the paper is claiming some practical motivation, but essentially recovers existing results through a different theoretical approach (and does not, really, "rethink" weight decay and how it should be used). If the emphasis was on the theoretical aspects (and motivate more why they are relevant), the assessment (& merit) could be different). I would therefore keep my score.

Add:
Official Review of Submission17464 by Reviewer xkCF
Official Reviewby Reviewer xkCF22 Jun 2026, 09:22 (modified: 29 Jul 2026, 02:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer xkCFRevisions
Summary:
The paper studies how weight decay interacts with learning rate, batch size, and training length. The main point is that weight decay should not be treated as a fixed constant as it changes the stability of the optimizer - so, the other hyperparameters have to move with it.

The authors analyze SGD and SGD with momentum using uniform stability. They show that weight decay can improve the generalization bound. But it also makes the allowable learning rate smaller.

They then use a regularization-equivalence argument to derive simple coupling rules. In particular, weight decay should scale roughly like 
, where 
 is the learning rate and 
 is the number of training steps. When batch size is included, the rule suggests that larger batches require larger effective weight decay.

Contribution Type: General: Most submissions will fall into this type.
Strengths And Weaknesses:
Strengths

The paper studies an important practical issue: weight decay is not independent of learning rate, batch size, and training length. The main idea is clean - weight decay improves stability, but it also changes the stable range of the learning rate. The stability analysis gives a useful explanation for why 
 should be coupled.

The batch-size extension is also intuitive. Larger batches reduce stochastic noise, so stronger explicit regularization may be needed.

The experiments support the qualitative claims. The good regions in the 
 grid form a coupled band. The trend is shown on CIFAR models and also in a Qwen LoRA fine-tuning setting.

Weaknesses

The key coupling law is not a rigorous optimality result. It comes from matching stability upper bounds under a heuristic regularization-equivalence assumption.

The practical scaling laws are close to prior work on AdamW timescales, batch-size scaling, and learning-rate-aware weight decay. The paper should make the distinction from these works much sharper.

The experiments validate the trend, but they do not show that the proposed rule is better than existing scaling rules or scheduling methods. The contribution is more of a unifying stability-based explanation than a fundamentally new hyperparameter law.

Quality: 3: good
Clarity: 4: excellent
Significance: 2: not good
Originality: 2: not good
Questions:
The main scaling rule is derived by matching stability upper bounds. Can the authors clarify why this matching should predict the optimal weight decay, rather than only giving a heuristic? It would be really useful if the authors can give a sharper justification or clearly states the result as a heuristic.

It would be good if the authors can highlight the novelty of this paper compared to prior work on AdamW timescales, batch-size scaling of weight decay, and learning-rate-aware weight decay schedules? Please give a direct comparison of assumptions, formulas, and /or predictions.

Can the authors compare their rule against at least one existing scaling rule or scheduled weight decay baseline? A small CIFAR-scale experiment would be sufficient.

The theory assumes convex/smooth losses, but the experiments are non-convex neural networks. Can the authors explain which parts of the theory they expect to survive in the non-convex setting?

Can the authors report how sensitive the proposed rule is to constants hidden in 
? This will help show whether the practical usefulness depends on whether (C) is stable across architectures, datasets, and optimizers.

Limitations:
Yes

Rating: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Ethical Concerns: NO or VERY MINOR ethics concerns only
Paper Formatting Concerns:
N/A

Code Of Conduct Acknowledgement: Yes
Responsible Reviewing Acknowledgement: Yes
Add:
Rebuttal by Authors
Rebuttalby Authors (Qixin Zhang, Peng Wang, Qiyang Zhou, Shi Fu, +7 more)28 Jul 2026, 19:47 (modified: 28 Jul 2026, 21:38)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, AuthorsRevisions
Rebuttal:
Weakness 1 and Question 1: The coupling law is not a rigorous optimality result
Response. We appreciate the opportunity to clarify this point, and we agree with the reviewer’s interpretation. Our reasoning consists of three steps:

Uniform stability directly upper-bounds the expected gap between empirical and population risks.
The relevant stability scales are 
 for SGD-WD and 
 for the comparison averaging procedure under the stated assumptions.
If two well-tuned regularization mechanisms operate at comparable stability scales, matching these scales suggests the balance 
.
Only Steps 1 and 2 are theorem-level statements. Step 3 is the “Regularization Equivalence” hypothesis. It provides a principled direction for reducing the hyperparameter search space, but it is not an optimality theorem. Matching two upper bounds cannot, by itself, prove that the corresponding hyperparameters minimize either the true expected generalization gap or the population risk. Upper bounds can be loose, equality of bounds is neither necessary nor sufficient for equality of the underlying risks, and population risk also contains optimization error and approximation bias.

Therefore, in the paper, we state that our research findings are heuristic, while the coupling 
 effects among hyperparameters are approximate, and that 
 may depend on factors such as the network architecture, dataset, optimizer, and training schedule.

Weakness 2 and Question 2: Insufficient distinction from prior scaling work
Response. We thank the reviewers for raising this comparison.

Wang and Aitchison did not establish a general scaling law for weight decay. Instead, by drawing a formal analogy between AdamW and EMA, they defined the AdamW memory timescales as 
 
 
, and empirically observed that, under their experimental settings, a well-performing value of 
 transfers approximately across model and dataset scales. We will therefore revise the manuscript to avoid the overly strong statement that Wang and Aitchison directly established or verified 
.

Our contribution is different in nature. Under the assumptions of Theorem 4.8, we derive a uniform-stability bound for SGD with weight decay and obtain the separate sufficient condition 
. We further extend this stability analysis to SGDM under stronger assumptions.

More broadly, our study investigates how weight decay affects the generalization behavior of SGD and SGDM, whereas Wang and Aitchison focus on the memory timescale of AdamW. Methodologically, our results are derived through uniform-stability analysis, supplemented by explicit assumptions to characterize the coupling among hyperparameters, while their relationships arise from a formal analogy with EMA. Although the resulting formulas are related, they concern different optimization algorithms, rely on different assumptions, characterize different quantities, and lead to distinct empirical interpretations.

Weakness 3 and Question 3: No demonstrated superiority over prior rules or schedules
Response. (We will present numerical experiments to illustrate this point.)

Question 4: What should survive in non-convex neural networks?
Response. Under non-convexity, a similar one-step stability analysis can be applied to SGD with weight decay (SGD-WD). This yields a generalization bound of 
 compared with 
 for standard SGD. However, the non-convex analysis requires prescribing a time-dependent learning-rate schedule, such as
. After substituting this schedule into the final bound, the learning rate no longer appears as an independent quantity. Consequently, the resulting expression cannot reveal the detailed coupling among the learning rate, weight decay, and other hyperparameters. For this reason, we did not include the non-convex result in the manuscript.

Nevertheless, the convex and non-convex analyses share several structural properties, and the convex result still provides useful intuition for neural networks. The shrinkage mechanism is exact at the update level. Decoupled weight decay contracts the parameter-dependent component of each update by the factor 
, regardless of whether the objective is convex. This means that 
 and 
 jointly determine the strength of parameter contraction throughout optimization. Their product should remain within an appropriate range: excessive contraction may impair generalization, whereas insufficient contraction may provide little regularization or stability benefit. This observation is consistent with the main argument of our paper.

Accordingly, the qualitative relation 
 is expected to remain relevant in non-convex neural networks. However, the proportionality coefficient 
 is unlikely to be universal. Unlike the constant assumed in the convex analysis, it may depend on the network architecture, optimization dynamics, parameter distribution, and data distribution.

Question 5: Sensitivity and transferability of 
Response. For the same configuration (fixed tasks and optimizer), the constant 
 remains relatively stable. As shown in Figure 1 of the manuscript, the joint values of the hyperparameters 
 and 
, for different individual values, are concentrated within a narrow red interval.

For different configurations, the present theory does not predict a universal 
, and the submitted experiments do not establish one. The constant absorbs at least three effects: looseness and unmatched constants in the stability bounds, the optimizer and momentum convention, and task/model-dependent optimization bias.

Add:
Additional experiment for Weakness 3 and Question 3
Official Commentby Authors (Qixin Zhang, Peng Wang, Qiyang Zhou, Shi Fu, +7 more)30 Jul 2026, 14:07Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Weakness 3 and Question 3: No demonstrated superiority over prior rules or schedules
Response. We have added a controlled scheduled-weight-decay comparison on ResNet-18/CIFAR-100. We distinguish two questions: whether scheduling only 
 helps when the learning rate is fixed, and whether applying the same schedule multiplier to both the learning rate and the weight-decay coefficient, i.e., 
 and 
, improves over the cosine-LR/constant-
 configuration used in our main experiments.

For the first comparison, we fix 
, 
 epochs, and 
 throughout training. We compare a constant 
 with cosine, linear, drop-step, and cosine-with-restarts schedule shapes adapted from Loshchilov and Hutter. Specifically, the scheduled methods use 
, where 
 is respectively cosine decay to zero, linear decay to zero, a 
 step-decay schedule at 50% and 75% of training, or cosine decay with a restart at epoch 50. Every method receives the same five-point budget 
.

The table reports the peak test accuracy over this common grid; values in parentheses are percentage-point changes relative to constant WD for the same optimizer.

Optimizer	Fixed	Cosine WD	Linear WD	Step-decay WD	Cosine-restart WD
SGD, momentum 
73.20	73.50 (+0.30)	73.22 (+0.02)	74.24 (+1.04)	73.43 (+0.23)
SGDM, momentum 
66.67	71.34 (+4.67)	70.32 (+3.65)	73.10 (+6.43)	70.13 (+3.46)
The best Step-decay settings use 
 for SGD and 
 for SGDM. These results indicate that a fixed WD, even when tuned over the same grid, does not necessarily yield the best empirical performance; searching over time-varying WD schedules remains beneficial, especially for SGDM, where step-decay WD improves the peak accuracy by 6.43 percentage points.

Because a constant LR is not our main training configuration, we also performed a stricter contextual comparison for SGDM. Following the AdamW/SGDW formulation more closely, we apply the same multiplier jointly, 
 and 
, and compare it with our default cosine LR plus a constant 
:

Method, SGDM, 
Best test accuracy	
 or selection rule
Cosine LR + constant WD, grid oracle	77.28	
Joint cosine multiplier	76.42	
Joint linear multiplier	76.17	
Joint drop-step multiplier	75.36	
Joint cosine-restart multiplier	75.11	
Applying schedules to both LR and WD achieves competitive performance, indicating a meaningful interaction between their dynamics. The second table confirms that LR and WD should be considered jointly in practical hyperparameter selection. However, jointly scheduling both quantities is not necessarily optimal: among all configurations evaluated under the same 100-epoch protocol and search grid, our cosine-LR protocol with a grid-searched constant WD achieves the highest test accuracy of 77.28%, outperforming all joint LR-WD schedules. These results suggest that coordination does not require both LR and WD to vary over time; rather, WD should be selected conditional on the LR trajectory. Thus, in this experiment, cosine LR combined with a properly tuned constant WD provides the most effective strategy. This is also consistent with the setup described in our paper.

Add:
 Replying to Additional experiment for Weakness 3 and Question 3
Official Comment by Reviewer xkCF
Official Commentby Reviewer xkCF03 Aug 2026, 02:05Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors
Comment:
Thank you for the detailed response. I still have a few follow-ups:

Eq. (17) gives
 
Since Exp. 3 imposes 
, this predicts approximately constant 
, not that both 
 and 
 should increase. Table 4 also shows 
 remaining within a narrow range. Could you clarify the exact empirical claim: is it 
, or 
 only when 
 is fixed?

I appreciate the additional schedule comparison. However, jointly setting
makes 
, so it does not preserve the coupling proposed by the paper. If possible, can you compare against a schedule with approximately constant 
, such as 
, or match methods by the same cumulative contraction 
?

The rebuttal acknowledges that (C) is not universal. Can you provide one held-out test in which (C) is estimated on one setting and used to predict 
 for another architecture, batch size, or learning rate? This would show whether the rule actually reduces tuning relative to a two-dimensional grid.

Add:
About OpenReview
Contact
FAQ
Hosting a Venue
Sponsors
Terms of Use / Privacy Policy
All Venues
Donate
News
OpenReview is a long-term project to advance science through improved peer review with legal nonprofit status. We gratefully acknowledge the support of the OpenReview Sponsors. © 2026 OpenReview

Rethinking Weight Decay: A Stability-Based View on Hyperparameters Coupling | OpenReview