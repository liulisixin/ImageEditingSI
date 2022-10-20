# Image editing of light and color from a single image: a baseline framework
The 30th Color and Imaging Conference (CIC30) (Oral)

[arXiv]

## Getting Started
- Clone this repo
- Install [PyTorch](http://pytorch.org) (torch==1.10.2) and other dependencies
- Install the requirements.txt.

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

