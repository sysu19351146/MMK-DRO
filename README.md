# MMK-DRO
This repository includes the PyTorch implementation of our paper: A Prior Knowledge-guided Distributionally Robust Optimization-based Adversarial Training Strategy for Medical Image Classification

First, you need to install the required packages， simply run:

```
pip install -r requirements.txt
```

Then, download and extract the datasets:

- [SARS-COV-2](https://www.kaggle.com/datasets/plameneduardo/sarscov2-ctscan-dataset)
- [COVID19-C](https://www.kaggle.com/datasets/quinn777/covid19c)
- [HAM10000](https://challenge.isic-archive.com/data/#2018)
- [Bcn20000](https://challenge.isic-archive.com/data/#2019)


Now, you can train the model after set the dataset path

Run the following command for SARS-COV-2 [Visformer-tiny](https://drive.google.com/file/d/1n9LwZX8Y2LLKzkVqI-euKDdSeXCn35vB/view?usp=share_link):

```
sh run_vis_sars.sh
```

Run the following command for SARS-COV-2 [Deit-tiny](https://drive.google.com/file/d/1DbZ-4R72zzVzAfNmRpY5o_Ic7mg97EZ7/view?usp=sharing):

```
sh run_deit_sars.sh
```
