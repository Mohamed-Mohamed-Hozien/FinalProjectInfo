# Image Compression Report

## [Source Code]([url](https://github.com/Mohamed-Mohamed-Hozien/FinalProjectInfo/tree/master))

## Test Cases

The code was tested on a dataset of images from categories "animals", "faces", and "nature", loaded from specified training and test directories. The output below shows the Mean Squared Error (MSE) and compression ratio for each test image under both RGB and YUV compression methods, followed by a summary of average values.

### RGB Compression Results
- Image 1: MSE: 4.33, Ratio: 4.00:1
- Image 2: MSE: 20.33, Ratio: 4.00:1
- Image 3: MSE: 10.79, Ratio: 4.00:1
- Image 4: MSE: 100.38, Ratio: 4.00:1
- Image 5: MSE: 6.35, Ratio: 4.00:1
- ...
- Image 29: MSE: 43.36, Ratio: 4.00:1
- Image 30: MSE: 155.52, Ratio: 4.00:1

### YUV Compression Results
- Image 1: MSE: 7.43, Ratio: 8.00:1
- Image 2: MSE: 26.65, Ratio: 8.00:1
- Image 3: MSE: 15.35, Ratio: 8.00:1
- Image 4: MSE: 109.94, Ratio: 8.00:1
- Image 5: MSE: 8.90, Ratio: 8.00:1
- ...
- Image 29: MSE: 109.95, Ratio: 8.00:1
- Image 30: MSE: 186.62, Ratio: 8.00:1

### Summary
- **RGB**: Average MSE: 48.81, Compression Ratio: 4.00:1
- **YUV**: Average MSE: 82.28, Compression Ratio: 8.00:1

## Discussion of Findings

### Overview
This project compares two image compression methods using vector quantization with a codebook size of 256: one in RGB color space and the other in YUV color space. The performance is evaluated based on Mean Squared Error (MSE), which measures reconstruction quality (lower is better), and compression ratio, which indicates the degree of data reduction (higher is better).

### Methodology
- **RGB Compression**:
    - Each image is split into R, G, and B components.
    - 2x2 blocks are extracted from each component and clustered using k-means to create three codebooks (one per channel).
    - During testing, each 2x2 block is replaced by the index of the closest codebook entry, and decompression reconstructs the image using these codebook entries.
- **YUV Compression**:
    - Images are converted to YUV color space, separating luminance (Y) from chrominance (U and V).
    - The Y component is processed at full resolution, while U and V are subsampled (4:2:0 scheme) before compression.
    - Separate codebooks are created for Y, U, and V, and decompression involves upsampling U and V before converting back to RGB.

### Results
- **RGB Method**:
    - Achieves an average MSE of 48.81, indicating relatively good reconstruction quality.
    - Consistently delivers a compression ratio of 4:1, as each 2x2 block per channel is reduced to a single index, and there are three channels.
- **YUV Method**:
    - Yields a higher average MSE of 82.28, suggesting greater distortion in reconstructed images.
    - Achieves a higher compression ratio of 8:1, due to subsampling of U and V components, reducing the total number of indices needed.

### Analysis
- **Compression Ratio**:
    - The RGB method compresses each channel equally, resulting in a compressed size of `(w/2) * (h/2) * 3` indices, yielding a 4:1 ratio compared to the original `w * h * 3` bytes.
    - The YUV method benefits from 4:2:0 subsampling, where U and V are reduced to a quarter of their original resolution before compression, leading to a compressed size of `(w/2) * (h/2) + 2 * (w/4) * (h/4)` indices, resulting in an 8:1 ratio.
- **Reconstruction Quality (MSE)**:
    - The RGB method preserves more detail by treating all color channels equally without subsampling, leading to lower MSE.
    - The YUV method’s higher MSE is largely due to the subsampling of chrominance channels (U and V), which can cause color bleeding or loss of detail, especially in images with sharp color transitions. Notable outliers (e.g., YUV MSE of 386.67 vs. RGB MSE of 9.65) highlight cases where subsampling severely impacts quality.
- **Trade-off**:
    - The YUV method doubles the compression ratio but sacrifices reconstruction quality, reflecting a trade-off between storage efficiency and visual fidelity.
    - The RGB method offers better quality but less compression, suitable for applications prioritizing image accuracy over file size reduction.

### Insights
- **Color Space Impact**:
    - RGB’s direct representation avoids information loss from subsampling, benefiting quality.
    - YUV leverages human visual sensitivity to luminance over chrominance, allowing higher compression via subsampling, but the fixed codebook size (256) may not adequately capture chrominance variability after subsampling.
- **Potential Improvements**:
    - Adjusting codebook sizes (e.g., larger for Y in YUV) could balance quality and compression.
    - Using advanced subsampling (e.g., averaging instead of simple decimation) or interpolation (e.g., bilinear upsampling) might reduce YUV’s MSE.
    - Varying codebook sizes to match compression ratios could enable a fairer quality comparison.

### Conclusion
The RGB compression method achieves better reconstruction quality (average MSE of 48.81) with a compression ratio of 4:1, while the YUV method offers superior compression (8:1 ratio) at the cost of higher distortion (average MSE of 82.28). The choice between these methods depends on application needs: RGB for quality-critical scenarios, and YUV for storage-constrained environments.
