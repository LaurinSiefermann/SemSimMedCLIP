# Thanks for running the experiments Marco.

In the following you find the arguments for our experiments.
FYI: We do not use loss weighting yet (0.5/0.5)

# ToDos for you:
1) Pull git repo (https://github.com/LaurinSiefermann/open_clip_based_thesis/tree/my_branch)
2) Install conda env (open_clip_based_thesis/environment.yml)
3) Commect to wandB (get your API key here: https://wandb.ai/settings)
4) Search and replace "/home/lsiefermann/open_clip_based_thesis/myUtils/" with your path to pkl datasets.
5) Search and replace "/scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/" with "/scratch/datasets/medical/MIMIC/"
6) Search and replace "/scratch1/lsiefermann/logs/" with where you want to save the logs
7) Replace gpu number before running an experiment

# 1  Report level - ChexPert 
#####       RDY       #####  
main.py \
--gpu 0 \
--train-data /home/lsiefermann/open_clip_based_thesis/myUtils/train_mimic.pkl \
--val-data /home/lsiefermann/open_clip_based_thesis/myUtils/val_mimic.pkl \
--mimimc-5x200 /home/lsiefermann/open_clip_based_thesis/myUtils/eval_mimic_5x200.pkl \
--dataset-type pkl \
--pkl-img-key jpg_paths \
--pkl-text-key REPORT \
--img-base-path /scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/ \
--workers 4 \
--epochs 25 \
--batch-size 200 \
--lr 5e-5 \
--wd 1e-4 \
--warmup 2500 \
--model bioClinicalBERT-timm-swin_small_patch4_window7_224 \
--pretrained-image 
--text-similarity-model all-distilroberta-v1 \
--loss-type single_feature \
--similarity-decision-1 chexPert-labels \
--zeroshot-frequency 1 \
--logs /scratch1/lsiefermann/logs/ \
--report-to wandb \
--save-only-best-model \
--save-pairings \

# 2  Report level - Chexpert / Sentence level - Chexpert
##### WAIT FOR LABLER ##### 

# 3  Report level - Chexpert / Sentence level - sBERT
#####       RDY       ##### 
main.py \
--gpu 0 \
--train-data /home/lsiefermann/open_clip_based_thesis/myUtils/train_mimic.pkl \
--val-data /home/lsiefermann/open_clip_based_thesis/myUtils/val_mimic.pkl \
--mimimc-5x200 /home/lsiefermann/open_clip_based_thesis/myUtils/eval_mimic_5x200.pkl \
--dataset-type pkl \
--pkl-img-key jpg_paths \
--img-base-path /scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/ \
--workers 4 \
--epochs 25 \
--batch-size 200 \
--lr 5e-5 \
--wd 1e-4 \
--warmup 2500 \
--model bioClinicalBERT-timm-swin_small_patch4_window7_224 \
--pretrained-image 
--text-similarity-model all-distilroberta-v1 \
--loss-type multiple_features \
--similarity-decision-1 chexPert-labels \
--similarity-decision-2 text_similarity_model \
--zeroshot-frequency 1 \
--logs /scratch1/lsiefermann/logs/ \
--report-to wandb \
--save-only-best-model \
--save-pairings \

# 4  Report level - sBERT 
#####       RDY       ##### 
main.py \
--gpu 0 \
--train-data /home/lsiefermann/open_clip_based_thesis/myUtils/train_mimic.pkl \
--val-data /home/lsiefermann/open_clip_based_thesis/myUtils/val_mimic.pkl \
--mimimc-5x200 /home/lsiefermann/open_clip_based_thesis/myUtils/eval_mimic_5x200.pkl \
--dataset-type pkl \
--pkl-img-key jpg_paths \
--pkl-text-key REPORT \
--img-base-path /scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/ \
--workers 4 \
--epochs 25 \
--batch-size 200 \
--lr 5e-5 \
--wd 1e-4 \
--warmup 2500 \
--model bioClinicalBERT-timm-swin_small_patch4_window7_224 \
--pretrained-image 
--text-similarity-model all-distilroberta-v1 \
--loss-type single_feature \
--similarity-decision-1 text_similarity_model \
--zeroshot-frequency 1 \
--logs /scratch1/lsiefermann/logs/ \
--report-to wandb \
--save-only-best-model \
--save-pairings \

# 5  Report level - sBERT / Sentence level - Chexpert
##### WAIT FOR LABLER ##### 

# 6  Report level - sBERT / Sentence level - sBERT
#####       RDY       ##### 
main.py \
--gpu 0 \
--train-data /home/lsiefermann/open_clip_based_thesis/myUtils/train_mimic.pkl \
--val-data /home/lsiefermann/open_clip_based_thesis/myUtils/val_mimic.pkl \
--mimimc-5x200 /home/lsiefermann/open_clip_based_thesis/myUtils/eval_mimic_5x200.pkl \
--dataset-type pkl \
--pkl-img-key jpg_paths \
--img-base-path /scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/ \
--workers 4 \
--epochs 25 \
--batch-size 200 \
--lr 5e-5 \
--wd 1e-4 \
--warmup 2500 \
--model bioClinicalBERT-timm-swin_small_patch4_window7_224 \
--pretrained-image 
--text-similarity-model all-distilroberta-v1 \
--loss-type multiple_features \
--similarity-decision-1 text_similarity_model \
--similarity-decision-2 text_similarity_model \
--zeroshot-frequency 1 \
--logs /scratch1/lsiefermann/logs/ \
--report-to wandb \
--save-only-best-model \
--save-pairings \

# 7  Sentence level - ChexPert 
##### WAIT FOR LABLER ##### 

# 8  Sentence level - sBERT
#####       RDY       ##### 
main.py \
--gpu 0 \
--train-data /home/lsiefermann/open_clip_based_thesis/myUtils/train_mimic_sentence_image \
--val-data /home/lsiefermann/open_clip_based_thesis/myUtils/val_mimic_sentence_image \
--mimimc-5x200 /home/lsiefermann/open_clip_based_thesis/myUtils/eval_mimic_5x200.pkl \ 
--dataset-type pkl \
--pkl-img-key jpg_paths \
--pkl-text-key sentences \
--img-base-path /scratch1/MIMIC/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/ \
--workers 4 \
--epochs 25 \
--batch-size 200 \
--lr 5e-5 \
--wd 1e-4 \
--warmup 2500 \
--model bioClinicalBERT-timm-swin_small_patch4_window7_224 \
--pretrained-image 
--text-similarity-model all-distilroberta-v1 \
--loss-type single_feature \
--similarity-decision-1 text_similarity_model \
--zeroshot-frequency 1 \
--logs /scratch1/lsiefermann/logs/ \
--report-to wandb \
--save-only-best-model \
--save-pairings \

# ! 9 Report level - CLIP LOSS !
#####       RDY       ##### 
# TODO. 


