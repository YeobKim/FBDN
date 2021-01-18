# FBDN
Residual Learning Using Features Block and Dual Network for Image Denoising (IPIU 2021)

This is a PyTorch implementation of the [33nd Workshop on Image Processing and Image Understanding (IPIU 2021)](http://www.ipiu.or.kr/2021/index.php) paper, [Residual Learning Using Features Block and Dual Network for Image Denoising](https://github.com/YeobKim/FBDN/blob/main/Residual%20Learning%20Using%20Features%20Block%20and%20Dual%20Network%20for%20Image%20Denoising%20(IPIU%202021).pdf)

## Abstract
In this experiment, we propose Image Denoising neural networks using Feature Extracting Block and Dual Network among Convolutional Neural Networks (CNN). The proposed network has a feature extraction block that extracts features of input images, and a structure that combines input and extracted features to share two results across a dual-path network. Experiments are conducted by eliminating noise by using images added with three levels of Additive White Gaussian Noise (AWGN) with noise levels of 15, 25, and 50 as input to the network. Experimental results show that the proposed network showed higher results than existing algorithms and networks in the objective evaluation, Peak Signal-to-Noise Ratio (PSNR), and that it also preserves in-image detail more sharply than existing algorithms and networks in the subjective evaluation.

## Proposed Algorithm
- Network Architecture
![Network](https://user-images.githubusercontent.com/59470033/104935503-ae856b80-59ee-11eb-8b5d-b55fa56e6b63.png)
- Results of Layer 1 and Layer 2
![Layer1_Result](https://user-images.githubusercontent.com/59470033/104937387-0d4be480-59f1-11eb-982c-f443e82c7b42.png)
![Layer2_Result](https://user-images.githubusercontent.com/59470033/104937397-0f15a800-59f1-11eb-9aff-3e426bf89d03.png)

## Dataset
We used the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) datasets for training the proposed network.   Furthermore, the datasets for testing the proposed network are BSD68 datasets and Set12 datasets.   There are other options you can choose.Please refer to dataset.py.

## Experimental Results
![psnr_result](https://user-images.githubusercontent.com/59470033/104938297-38830380-59f2-11eb-8596-d50717bfb0f9.png)
![image_result](https://user-images.githubusercontent.com/59470033/104938965-28b7ef00-59f3-11eb-8193-65e24c3ce87a.png)

## Contact
If you have any questions, please contact athurk94111@gmail.com.

