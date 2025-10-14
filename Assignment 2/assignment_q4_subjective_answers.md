### a) Image Reconstruction:
We implemented two separate functions: **factorize_with_SGD(image, k)** and **factorize_with_ALS(image, k)** that factorizes the given image using Stochastic Gradient Descent and Alternating Least Squares respectively.

Both the functions are used to reconstruct images in cases where a block is missing and when random subset of pixels are missing. Reconstruction is done using ranks: 4, 8, 16, 32, and 64. 

As expected, RMSE decreases and PSNR increases with increasing rank.

### b) Data Compression:
We created three separate patches for each of the three cases: 
single color **(P1)**, 2-3 different colors **(P2)** and atleast 5 different colors **(P3)**.

These patches are factorized into low rank matrices using **factorize_with_Adam(image, k)** function and reconstructed. Their reconstruction losses are recorded.

For smaller ranks, the reconstruction losses are in the order: **P1<P2<P3** (i.e. patch with more number of different colors has higher reconstrucion loss). As rank increases, the reconstruction loss approaches zero for all patches.