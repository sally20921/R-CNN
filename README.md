# R-CNN-based custom object detectors
This repository implements a R-CNN-based object detectors on a custom dataset.
## Working details of R-CNN
![Screen Shot 2021-08-18 at 6 47 26 AM](https://user-images.githubusercontent.com/38284936/129805064-5b4c7a2b-b3a7-40cb-8571-9001a1d804fc.png)

1. Extract region proposals from an image.
2. Resize (warp) all the extracted regions to get images of the same size.
3. Pass the resized region proposals through a network 
