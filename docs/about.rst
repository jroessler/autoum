About AutoUM
===========================

``AutoUM`` is a Python Framework for automatically evaluating various uplift modeling algorithms to estimate individual treatment
effects. It provides an interface that allows to estimate the Uplift (also called Conditional Average Treatment Effect (CATE), or
Individual Treatment Effect (ITE)) from **experimental data** with a **binary treatment** indiciator (customers are either treated or
not) and a **binary response** variable (customers either respond or not).

Typical use cases include:

- (Proactive) churn prevention
- Up-/Cross-selling
- Customer acquisition

The framework currently supports the following approaches:

- "General" Two Model Approach
- X-Learner (Künzel et al. 2019)
- Lai's Generalization (Kane et al. 2014)
- Class Variable Transformation (Jaskowski and Jaroszewicz 2012)
- Delta-Delta Pi (DDP) (Hansotia & Rukstales 2002)
- Uplift Random Forest with KL-, ED-, CHI- based splitting criterion (Rzepakowski and Jaroszewicz 2012)
- Contextual Treatment Selection (Zhao et al. 2017)
- Interaction Tree (Su et al. 2009)
- Conditional Interaction Tree (Su et al. 2012)
- Generalized Random Forest (Athey et al. 2019)
- S-Learner (e.g., Künzel et al. 2019)
- Treatment Dummy Approach (Lo 2002)
- Bayesian Causal Forest (Hahn et al. 2020)
- R-Learner (Nie and Wager 2020)
- Traditional Approach (without control group)