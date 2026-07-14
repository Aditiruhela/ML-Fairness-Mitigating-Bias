# Mitigating Bias in ML Models using MinDiff

## Problem
Classification models can show unequal performance across demographic
groups even when overall accuracy looks fine. This project applies
TensorFlow's MinDiff technique to reduce that disparity in a text
classification model.

## Approach
- Trained a baseline Keras classifier and measured performance gap
  across [group A] vs [group B] using False Positive Rate (FPR) and
  Equal Opportunity Difference as fairness metrics.
- Applied MinDiff regularization during training to penalize the model
  for unequal error rates between groups.
- Re-evaluated the same metrics post-mitigation to quantify improvement.

## Results
| Metric | Before MinDiff | After MinDiff |
|---|---|---|
| FPR Gap | [X]% | [X]% |
| Equal Opportunity Diff | [X] | [X] |

## Key takeaway
[e.g., "Reduced FPR disparity by X% with only a Y% drop in overall
accuracy — showing fairness and performance don't have to fully trade off."]

## Tech stack
TensorFlow, TF Model Remediation (MinDiff), Keras, Vertex AI
