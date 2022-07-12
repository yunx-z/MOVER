# MOVER
Code and data for [MOVER: Mask, Over-generate and Rank for Hyperbole Generation](https://aclanthology.org/2022.naacl-main.440/) (NAACL 2022)

## Data Preparation
Request HYPO-XL and HYPO-L from this [link](https://drive.google.com/file/d/1va8yoS8X5XQaCmVKTU9MuhemI125x1Ev/view?usp=sharing)

Request HYPO.tsv from this [paper](https://aclanthology.org/D18-1367/)

Preprocess HYPO dataset
`python preprocess_data.py`

Prepare masked hyperbolic spans:
`python unexpected_syntax_mask.py`

## Model Training and Inference
Train MOVER:
`bash run_bart.sh`

Inference with MOVER:
`bash infer_run_bart.sh`

## Baseline Implementations
- R1:
`python ir_hypo_baseline.py`

- R3:
`python retrieve_rank_baseline.py`

## Automatic Evaluation
`python evaluate_result.py`
