# Thanks for running the experiments Marco.

In the following you find the arguments for our experiments.
FYI: We do not use loss weighting yet (0.5/0.5)

# ToDos for you:
1) Pull git repo (https://github.com/LaurinSiefermann/open_clip_based_thesis/tree/my_branch)
2) Install conda env (open_clip_based_thesis/environment.yml)
3) Commect to wandB (get your API key here: https://wandb.ai/settings)
4) Search and replace "/home/lsiefermann/open_clip_based_thesis/myUtils/" with your path to pkl datasets.
5) Search and replace /scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/ with "/scratch/datasets/medical/MIMIC/"
6) Search and replace "/scratch1/lsiefermann/logs/" with where you want to save the logs
7) Replace gpu number before running an experiment

# 1  Report level - ChexPert 
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#1-Report-level-ChexPert] --gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic.pkl --val-data /scratch/mcipriano/laurin/val_mimic.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --pkl-text-key REPORT --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100 --lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --loss-type single_feature --similarity-decision-1 chexPert-labels --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model --save-pairings 

# 2  Report level - Chexpert / Sentence level - Chexpert
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#2-Report-level-Chexpert___Sentence-level-Chexpert] -- gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic.pkl --val-data /scratch/mcipriano/laurin/val_mimic.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100--lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --loss-type multiple_features --similarity-decision-1 chexPert-labels --similarity-decision-2 chexPert-labels --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model  --save-pairings 

# 3  Report level - Chexpert / Sentence level - sBERT
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#3-Report-level-Chexpert___Sentence-level-sBERT] --gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic.pkl --val-data /scratch/mcipriano/laurin/val_mimic.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100 --lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --text-similarity-model all-distilroberta-v1 --loss-type multiple_features --similarity-decision-1 chexPert-labels --similarity-decision-2 text_similarity_model --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model --save-pairings 

# 4  Report level - sBERT 
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#4-Report-level-sBERT] --gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic.pkl --val-data /scratch/mcipriano/laurin/val_mimic.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --pkl-text-key REPORT --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100  --lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --text-similarity-model all-distilroberta-v1 --loss-type single_feature --similarity-decision-1 text_similarity_model --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb  --save-only-best-model --save-pairings 

# 5  Report level - sBERT / Sentence level - Chexpert
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#5-Report-level-sBERT___Sentence-level-Chexpert] -- gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic.pkl --val-data /scratch/mcipriano/laurin/val_mimic.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100 --lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --text-similarity-model all-distilroberta-v1 --loss-type multiple_features --similarity-decision-1 text_similarity_model --similarity-decision-2 chexPert-labels --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model --save-pairings 

# 6  Report level - sBERT / Sentence level - sBERT
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#6-Report-level-sBERT___Sentence-level-sBERT] --gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic.pkl --val-data /scratch/mcipriano/laurin/val_mimic.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100 --lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --text-similarity-model all-distilroberta-v1 --loss-type multiple_features --similarity-decision-1 text_similarity_model --similarity-decision-2 text_similarity_model --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model --save-pairings 

# 7  Sentence level - ChexPert 
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#7-Sentence-level-ChexPert] --gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic_sentence_image.pkl --val-data /scratch/mcipriano/laurin/val_mimic_sentence_image.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100  --lr 5e-5 --wd 1e-4--warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --loss-type single_feature --similarity-decision-1 chexPert-labels --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model --save-pairings 

# 8  Sentence level - sBERT
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#8-Sentence-level-sBERT] --gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic_sentence_image.pkl --val-data /scratch/mcipriano/laurin/val_mimic_sentence_image.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --pkl-text-key sentences --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100  --lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image --text-similarity-model all-distilroberta-v1 --loss-type single_feature --similarity-decision-1 text_similarity_model --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model --save-pairings 

# 9 Report level - CLIP LOSS 
/home/mcipriano/projects/laurin/src/training/main.py --exp-desc [#9-Report-level-CLIP-LOSS] --gpu 0 --train-data /scratch/mcipriano/laurin/train_mimic.pkl --val-data /scratch/mcipriano/laurin/val_mimic.pkl --mimimc-5x200 /scratch/mcipriano/laurin/eval_mimic_5x200.pkl --dataset-type pkl --pkl-img-key jpg_paths --pkl-text-key REPORT --img-base-path /scratch/datasets/medical/MIMIC/ --workers 4 --epochs 25 --batch-size 100 --lr 5e-5 --wd 1e-4 --warmup 2500 --model bioClinicalBERT-timm-swin_small_patch4_window7_224 --pretrained-image  --loss-type clip --similarity-decision-1 None --zeroshot-frequency 1 --logs /scratch/mcipriano/results/laurin/ --report-to wandb --save-only-best-model --save-pairings 


