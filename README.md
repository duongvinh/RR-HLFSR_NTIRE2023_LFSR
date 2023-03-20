# RR-HLFSR_NTIRE2023_LFSR
Residual in Resiual Learning based Hybrid Light Field Image Super-Resolution Network (RR-HLFSR)
## News
[2023-03-20]: We released our RR-HLFSR, which is participated in [NTIRE2023](https://cvlai.net/ntire/2023/). Our RR-HLFSR is an enhanced version of our [HLFSR](https://github.com/duongvinh/HLFSR-SSR)


## Results
We share the pre-trained models and the SR LF images generated by our RR-HLFSR model for 4x LF spatial SR, which are avaliable at https://drive.google.com/drive/u/2/folders/160KS4l5jWEehJ0KtgOg0T6pdlhgTyWbg

## Code
### Dependencies
* Python 3.6
* Pyorch 1.3.1 + torchvision 0.4.2 + cuda 92
* Matlab

### Dataset
We use the processed data by [LF-DFnet](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9286855), including EPFL, HCInew, HCIold, INRIA and STFgantry datasets for training and testing. Please download the dataset in the official repository of [LF-DFnet](https://github.com/YingqianWang/LF-DFnet).

### Prepare Training and Test Data
* To generate the training data, please first download the five datasets and run:
  ```matlab
  Generate_Data_for_Training.m
* To generate the test data, run:
  ```matlab
  Generate_Data_for_Test.m
### Train
* Run:
  ```python
  python train.py  --model_name RR_HLFSR --angRes 5 --scale_factor 4 --n_groups 10 --n_blocks 15 --channels 64  --crop_test_method 3  
### Test
* Run:
  ```python
  python test.py --model_name RR_HLFSR --angRes 5 --upscale_factor 4 --n_groups 10 --n_blocks 15 --channels 64  --crop_test_method 1 --self_ensemble True  --model_path [pre-trained dir]
  
[Important note]: 

1) For our HLFSR method, the performance is following “the larger image patch size is the better”. For example, if we keep the whole image as an input of our network (i.e., crop_test_method is fixed equal to 1), it can be achieved the best performance. This is because our proposed network components require an adequate size of an input image to better exploit the pixel correlations in a larger receptive field. To get the same performance as reported in this NTIRE2023 LFSR challenge, we need to set the default crop_test_method equal to 1.

2) We are using the geometric self-ensemble method to further improve the performance in NTIRE2023 LFSR challenges

3) We may need to turn off the calculate PSNR/SSIM by settings "--test_NTIRE2023_LFSR 1" since there is no ground true HR images during the testing phase.
  
  
  
## Citation
If you find this work helpful, please consider citing the following papers:<br> 
```Citation
@article{
  title={Light Field Super-Resolution Network Using Joint Spatio-Angular and Epipolar Information},
  author={Vinh Van Duong, Thuc Nguyen Huu, Jonghoon Yim, and Byeungwoo Jeon},
  journal={ IEEE Transactions on Computational Imaging},
  year={2023},
  publisher={IEEE}
}
```
```Citation
@InProceedings{LF-InterNet,
  author    = {Wang, Yingqian and Wang, Longguang and Yang, Jungang and An, Wei and Yu, Jingyi and Guo, Yulan},
  title     = {Spatial-Angular Interaction for Light Field Image Super-Resolution},
  booktitle = {European Conference on Computer Vision (ECCV)},
  pages     = {290-308},
  year      = {2020},
}

```Citation
@article{LF-DFnet,
  author  = {Wang, Yingqian and Yang, Jungang and Wang, Longguang and Ying, Xinyi and Wu, Tianhao and An, Wei and Guo, Yulan},
  title   = {Light Field Image Super-Resolution Using Deformable Convolution},
  journal = {IEEE Transactions on Image Processing},
  volume  = {30),
  pages   = {1057-1071},
  year    = {2021},
}

```
## Acknowledgement
Our work and implementations are inspired and based on the following projects: <br> 
[Basic-LFSR](https://github.com/ZhengyuLiang24/BasicLFSR)<br> 
[LF-DFnet](https://github.com/YingqianWang/LF-DFnet)<br> 
[LF-InterNet](https://github.com/YingqianWang/LF-InterNet)<br>
We sincerely thank the authors for sharing their code and amazing research work!

## Contact
if you have any question, please contact me through email duongvinh@skku.edu
