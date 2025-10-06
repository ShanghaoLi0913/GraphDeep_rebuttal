# Hallucination Case Study Analysis Report

Generated on: 2025-07-27 21:05:32
Input file: experiment_records/inference_results/llama2-7b/colab_dev_simple.jsonl
Total samples analyzed: 5000

## Executive Summary

This report analyzes different types of hallucinations based on the PRD×SAS quadrant classification:

- **Q1 (High PRD, High SAS)**: 短路径过拟合 - Models over-rely on shortest paths but maintain semantic alignment
- **Q2 (Low PRD, High SAS)**: 理想情况 - Low path dependence with high semantic alignment (ideal behavior)
- **Q3 (Low PRD, Low SAS)**: 语义脱节 - Semantic misalignment with unfocused reasoning
- **Q4 (High PRD, Low SAS)**: 路径误导 - High path dependence but semantic errors

## Quadrant Analysis

| Quadrant | Count | Hit Rate (%) | Hallucination Rate (%) | Avg PRD | Avg SAS | Avg TUS |
|----------|-------|--------------|------------------------|---------|---------|----------|
| Q1 High Prd High Sas | 966 | 90.5 | 9.5 | 0.7521 | 0.4210 | 0.8761 |
| Q2 Low Prd High Sas | 1534 | 95.0 | 5.0 | 0.7010 | 0.4524 | 0.8505 |
| Q3 Low Prd Low Sas | 966 | 77.8 | 22.2 | 0.7071 | 0.3403 | 0.8536 |
| Q4 High Prd Low Sas | 1534 | 89.1 | 10.9 | 0.7538 | 0.3441 | 0.8769 |

## Representative Cases

### 短路径过拟合 (High PRD, High SAS)

#### Case 1
- **Question**: what films did David Carradine act in
- **Model Answer**: death race, death race, deathsport, death race, circle of iron, lone wolf mcquade, bound for glory, crime zone
- **Gold Answer**: big stan, death race 2000, lone wolf mcquade, q, bound for glory, gray lady down, circle of iron, deathsport, the warrior and the sorceress, crime zone
- **Correct**: ❌
- **PRD Score**: 0.9019
- **SAS Score**: 0.3773
- **TUS Score**: 0.9520

#### Case 2
- **Question**: what does John Davidson star in
- **Model Answer**: john davidson
- **Gold Answer**: the one and only, genuine, original family band
- **Correct**: ❌
- **PRD Score**: 0.8141
- **SAS Score**: 0.4182
- **TUS Score**: 0.9064

#### Case 3
- **Question**: what does W.C. Fields star in
- **Model Answer**: w.c. fields starred in the old fashioned way, never give a sucker an even break, and the big broadcast of
- **Gold Answer**: never give a sucker an even break, the old fashioned way, the big broadcast of 1938
- **Correct**: ❌
- **PRD Score**: 0.7951
- **SAS Score**: 0.3739
- **TUS Score**: 0.8984


### 理想情况 (Low PRD, High SAS)

#### Case 1
- **Question**: what was Billy Kent the director of
- **Model Answer**: billy kent
- **Gold Answer**: the oh in ohio
- **Correct**: ❌
- **PRD Score**: 0.6792
- **SAS Score**: 0.4825
- **TUS Score**: 0.8398

#### Case 2
- **Question**: what does Robin Harris star in
- **Model Answer**: robin harris
- **Gold Answer**: house party, bébé's kids
- **Correct**: ❌
- **PRD Score**: 0.7008
- **SAS Score**: 0.4801
- **TUS Score**: 0.8515

#### Case 3
- **Question**: what is a film directed by Neil Thompson
- **Model Answer**: neil thompson
- **Gold Answer**: clubbed
- **Correct**: ❌
- **PRD Score**: 0.6645
- **SAS Score**: 0.4770
- **TUS Score**: 0.8321


### 语义脱节 (Low PRD, Low SAS)

#### Case 1
- **Question**: what was Omar Naim the director of
- **Model Answer**: omar naim
- **Gold Answer**: the final cut
- **Correct**: ❌
- **PRD Score**: 0.7101
- **SAS Score**: 0.3736
- **TUS Score**: 0.8555

#### Case 2
- **Question**: what films did Geoffrey Sax direct
- **Model Answer**: geoffrey sax directed white noise.
- **Gold Answer**: white noise
- **Correct**: ❌
- **PRD Score**: 0.6937
- **SAS Score**: 0.3733
- **TUS Score**: 0.8476

#### Case 3
- **Question**: what does Miles Fisher star in
- **Model Answer**: miles fisher
- **Gold Answer**: final destination 5, believe me
- **Correct**: ❌
- **PRD Score**: 0.7186
- **SAS Score**: 0.3718
- **TUS Score**: 0.8595


### 路径误导 (High PRD, Low SAS)

#### Case 1
- **Question**: what films did Eve act in
- **Model Answer**: eve acted in the woodsman.
- **Gold Answer**: the woodsman
- **Correct**: ❌
- **PRD Score**: 0.9055
- **SAS Score**: 0.3644
- **TUS Score**: 0.9531

#### Case 2
- **Question**: what does Brittany Murphy appear in
- **Model Answer**: comedy
- **Gold Answer**: 8 mile, girl, interrupted, clueless, just married, don't say a word, spun, uptown girls, riding in cars with boys, little black book, cherry falls, the ramen girl, neverwas, abandoned, deadline, bongwater, tribute, david and lisa
- **Correct**: ❌
- **PRD Score**: 0.8008
- **SAS Score**: 0.3258
- **TUS Score**: 0.9014

#### Case 3
- **Question**: which movies did Gérard Depardieu direct
- **Model Answer**: paris, je t'aime
- **Gold Answer**: paris, je t'aime
- **Correct**: ❌
- **PRD Score**: 0.7974
- **SAS Score**: 0.3136
- **TUS Score**: 0.8984


## Key Insights

### Hallucination Mechanisms

#### Q1 High PRD High SAS
- Sample count: 966
- Hallucination rate: 9.5%
- **Mechanism**: Over-reliance on shortest paths with good semantic alignment
- **Interpretation**: Models follow logical reasoning paths but may miss broader context

#### Q2 Low PRD High SAS
- Sample count: 1534
- Hallucination rate: 5.0%
- **Mechanism**: Balanced reasoning with good semantic alignment
- **Interpretation**: Ideal behavior - models integrate multiple information sources

#### Q3 Low PRD Low SAS
- Sample count: 966
- Hallucination rate: 22.2%
- **Mechanism**: Unfocused reasoning with poor semantic alignment
- **Interpretation**: Models generate plausible but semantically ungrounded responses

#### Q4 High PRD Low SAS
- Sample count: 1534
- Hallucination rate: 10.9%
- **Mechanism**: Path-dependent but semantically incorrect reasoning
- **Interpretation**: Models follow paths but misinterpret semantic content

