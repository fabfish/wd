OpenReview
.net
Search articles, authors and reviews...
Notifications20
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
4 / 4 replies shown
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
Official Reviewby Reviewer eC8H26 Jun 2026, 18:30 (modified: 23 Jul 2026, 23:05)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer eC8HRevisions
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
Official Review of Submission17464 by Reviewer SijV
Official Reviewby Reviewer SijV26 Jun 2026, 01:12 (modified: 23 Jul 2026, 23:05)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer SijVRevisions
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
Official Review of Submission17464 by Reviewer xkCF
Official Reviewby Reviewer xkCF22 Jun 2026, 09:22 (modified: 23 Jul 2026, 23:05)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer xkCFRevisions
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

