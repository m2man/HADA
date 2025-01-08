# HADA: A Graph-based Amalgamation Framework in Image-Text Retrieval

## Introduction
Our [paper](https://arxiv.org/abs/2301.04742) has been accepted at ECIR'23

HADA is a framework that combines any pretrained SOTA models in image-text retrieval (ITR) to produce a better result. A special feature in HADA is that this framework only introduces a tiny number of additonal trainable parameters. Thus, it does not required multiple GPUs to train HADA or external large-scale dataset (although pretraining may further improve the performance).

In this repo, we used HADA to combine 2 SOTA models including ALBEF and LightningDOT. The total recall was increase by 3.6% on the Flickr30k dataset. Therefore, it needs to clone ALBEF and LightningDOT and extract their feature first.

## Extracted Features
We uploaded the extracted feature and pretrained model [here](https://drive.google.com/drive/folders/13NnWfIa_1HAcWbRn5_R9Nkibnq6zKN0G?usp=sharing). Or you can run the extraction again by using files in **ALBEF** and **DOT** directory.

## MLFlow-UI tracking
We used **mlflow-ui** to keep track the performance between configurations. Please modify or remove this related-part if you do not want to use.

## Train and Evaluate
Remember to update the path in the config files in **HADA** folders. Then you can train or evaluate by the file `run_exp.py`

```python
# Train
python run_exp.py -cp HADA_m_extend/Config/C5.yml -rm train

# Test
python run_exp.py -cp HADA_m_extend/Config/C5.yml -rm test
```

## LAVIS as Backbones
We created a sub-repository for applying HADA using [LAVIS](https://github.com/salesforce/LAVIS) as backbones [here](https://github.com/m2man/HADA-LAVIS).

## Contact
For any issue or comment, you can directly email me at manh.nguyen5@mail.dcu.ie

For citation, you can add the bibtex as following:
```
@inproceedings{hada_ecir2023,
  title={HADA: A Graph-based Amalgamation Framework in Image-text Retrieval},
  author={Nguyen, Manh-Duy and Nguyen, Binh T and Gurrin, Cathal},
  booktitle={Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2--6, 2023, Proceedings, Part I},
  pages={717--731},
  year={2023}
}
```
