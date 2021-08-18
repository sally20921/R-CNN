# R-CNN-based custom object detectors
This repository implements a R-CNN-based object detectors on a custom dataset with detailed comments.

[doc](https://github.com/sally20921/R-CNN/docs/DOC.md)

## R-CNN Architecture
![Screen Shot 2021-08-18 at 6 47 26 AM](https://user-images.githubusercontent.com/38284936/129805064-5b4c7a2b-b3a7-40cb-8571-9001a1d804fc.png)

1. define a VGG backbone
2. normalized crop => pretrained model => features
3. attach a linear layer with sigmoid to the VGG backbone (predicting class)
4. attach another linear layer to predict bb offset
5. define loss for each of the two output (class, bb offset)
6. train the model 


## Install
```bash
pip install -q --upgrade selectivesearch torch_snippets
pip install numpy pandas scikit-image scipy
pip install torch torchvision opencv-python 
pip install kaggle --upgrade
```

## Prepare Data:

`kaggle.json` is the file you can get by clicking on Create New API token in your personal account.

```
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
ls ~/.kaggle
chmod 600 /root/.kaggle/kaggle.json
kaggle datasets download -d sixhky/open-images-bus-trucks
unzip -qq open-images-bus-trucks.zip
```
