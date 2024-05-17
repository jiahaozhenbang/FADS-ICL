# FADS-ICL
Source code for the paper "Feature-Adaptive and Data-Scalable In-Context Learning" in ACL 2024 

<div  align="center">  
<img src="./FADS-ICL.png" alt="Framework of FADS-ICL" align=center />
</div>  

## Preparation
### Environment
The code is tested under torch==1.12.0 and transformers==4.20.1, though the requirement of spefic version is not very strict, run with no bugs, then you are set.
### Model
Prepare your LLM ([gpt2](https://huggingface.co/gpt2-xl/tree/main) or opt) in `./llm/`, I personally prefer download them myself and configure the local path in scripts.
### Data
[Download](https://drive.google.com/file/d/1Yh2blPkJvMtdm5xWKoHr2fLp2i2Bn5Ir/view?usp=share_link) dataset and unzip them in `./data`.\
The structure of the project looks like:
```
.
├── run_icl.sh
├── run_knnprompting.sh
├── icl.py
├── knn_prompting.py
├── utils
│   ├── anchor.py
│   ├── dataset.py
│   ├── __init__.py
│   └── template.py
├── llm
│   └── gpt2-xl
│       ├── config.json
│       ├── merges.txt
│       ├── pytorch_model.bin
│       ├── tokenizer.json
│       └── vocab.json
└── data
    └── sst2
        ├── dev_subsample.jsonl
        ├── test.jsonl
        └── train.jsonl
```

## Run
Run kNNPrompting or In-Context Learning as follows, check the configuration in the script including dataset, llm, seed, etc.
```
bash run_knnprompting.sh
```
or
```
bash run_icl.sh
```
## Results
As the entire framework is training-free, you shall get **exact** results w.r.t. random seeds as follows (invariant to different environment):

| Seed                                | 1      | 2      | 3      | 4      | 5      |
| ----------------------------------- | ------ | ------ | ------ | ------ | ------ |
| **In-Context Learning** (gpt2-xl)   | 0.8438 | 0.8125 | 0.7227 | 0.8633 | 0.8242 |
| **KNN Prompting** (gpt2-xl, N=1024) | 0.8711 | 0.8867 | 0.8906 | 0.8711 | 0.8906 |

Full results are listed in the paper (see Table 8 and others).

## Citation
 * If you have any quesitons, feel free to open an issue.
 * If you find this repo useful, please cite us as:
```
@inproceedings{
xu2023knn,
title={\$k\${NN} Prompting: Beyond-Context Learning with Calibration-Free Nearest Neighbor Inference},
author={Benfeng Xu and Quan Wang and Zhendong Mao and Yajuan Lyu and Qiaoqiao She and Yongdong Zhang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=fe2S7736sNS}
}
```

## Finetune

After the above steps are completed, modify the path parameters of the [script](https://github.com/jiahaozhenbang/SCOPE/blob/main/train.sh) and run:

`bash train.sh`

## Inference

Please modify the path parameters of the [script](predict.sh) and run:

`bash predict.sh`

## Citation

If you find this work is useful for your research, please cite our papers:

#### Improving Chinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity

```bibtex
@inproceedings{li-etal-2022-improving-chinese,
    title = "Improving {C}hinese Spelling Check by Character Pronunciation Prediction: The Effects of Adaptivity and Granularity",
    author = "Li, Jiahao  and
      Wang, Quan  and
      Mao, Zhendong  and
      Guo, Junbo  and
      Yang, Yanyan  and
      Zhang, Yongdong",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.emnlp-main.287",
    pages = "4275--4286",
    abstract = "Chinese spelling check (CSC) is a fundamental NLP task that detects and corrects spelling errors in Chinese texts. As most of these spelling errors are caused by phonetic similarity, effectively modeling the pronunciation of Chinese characters is a key factor for CSC. In this paper, we consider introducing an auxiliary task of Chinese pronunciation prediction (CPP) to improve CSC, and, for the first time, systematically discuss the adaptivity and granularity of this auxiliary task. We propose SCOPE which builds upon a shared encoder two parallel decoders, one for the primary CSC task and the other for a fine-grained auxiliary CPP task, with a novel adaptive weighting scheme to balance the two tasks. In addition, we design a delicate iterative correction strategy for further improvements during inference. Empirical evaluation shows that SCOPE achieves new state-of-the-art on three CSC benchmarks, demonstrating the effectiveness and superiority of the auxiliary CPP task. Comprehensive ablation studies further verify the positive effects of adaptivity and granularity of the task.",
}

```