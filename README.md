# Image editing of light and color from a single image: a baseline framework
The 30th Color and Imaging Conference (CIC30) (Oral)

[paper](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/30/1/33)

## Getting Started
- Clone this repo
- Install [PyTorch](http://pytorch.org) (torch==1.10.2) and other dependencies
- Install the requirements.txt.

## Relighting Surreal Dataset (RLSID)
The proposed dataset is released in this
[link](http://datasets.cvc.uab.cat/CiC/RLSID.html).

## Training
To train the model, use the below command:
```bash
python train_1to3.py
```

## Testing
To test the model, you need to modify which_experiment and epoch in options/test_quantitative_options.py first, 
then use the below command:
```bash
python test_quantitative.py
```

