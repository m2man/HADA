# HADA: A Graph-based Amalgamation Framework in Image-Text Retrieval

## Introduction
HADA is a framework that combines any pretrained SOTA models in image-text retrieval (ITR) to produce a better result. A special feature in HADA is that this framework only introduces a tiny number of additonal trainable parameters. Thus, it does not required multiple GPUs to train HADA or external large-scale dataset (although pretraining may further improve the performance).

In this repo, we used HADA to combine 2 SOTA models including ALBEF and LightningDOT. The total recall was increase by $3.6\%$ on the Flickr30k dataset. Therefore, it needs to clone ALBEF and LightningDOT and extract their feature first.

## Extracted Features
We uploaded the extracted feature here. Or you can run the extraction again by using files in **ALBEF** and **DOT** directory.

## Train and Evaluate
Remember to update the path in the config files in **HADA** folders. Then you can train or evaluate by the file `run_exp.py`

```python
# Train
python run_exp.py -cp HAMA_m_extend/Config/C5.yml -rm train

# Test
python run_exp.py -cp HAMA_m_extend/Config/C5.yml -rm test
```

## Contact
For any issue or comment, you can directly email me at manh.nguyen5@mail.dcu.ie