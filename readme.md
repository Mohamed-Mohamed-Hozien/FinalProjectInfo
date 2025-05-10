# Image Compression Report

## Source Code

Below is the Java code used for implementing and comparing RGB and YUV image compression methods using vector quantization with a codebook size of 256.

```java
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import javax.imageio.ImageIO;

public class Main {
    private static final int CODEBOOK_SIZE = 256;
    private static final int MAX_ITER = 20;

    private static List<byte[]> redCodebook;
    private static List<byte[]> greenCodebook;
    private static List<byte[]> blueCodebook;
    private static List<byte[]> yCodebook;
    private static List<byte[]> uCodebook;
    private static List<byte[]> vCodebook;

    private static final String TRAIN_PATH = "C:\\Users\\Hozien\\Desktop\\UST-CSAI\\Year 3 Semester 2\\DSAI 325 Information Theory\\Project\\untitled\\test";
    private static final String TEST_PATH  = "C:\\Users\\Hozien\\Desktop\\UST-CSAI\\Year 3 Semester 2\\DSAI 325 Information Theory\\Project\\untitled\\train";
    private static final List<String> CATEGORIES = Arrays.asList("animals", "faces", "nature");

    public static void main(String[] args) throws Exception {
        System.out.println("=== RGB Compression (k=" + CODEBOOK_SIZE + ") ===");
        trainRGB();
        TestResult rgbResult = testRGB();

        System.out.println("\n=== YUV Compression (k=" + CODEBOOK_SIZE + ") ===");
        trainYUV();
        TestResult yuvResult = testYUV();

        System.out.printf("\nSummary (codebook size=%d):\n", CODEBOOK_SIZE);
        System.out.printf("RGB => MSE: %.2f, Compression Ratio: %.2f:1\n",
                rgbResult.mse, rgbResult.ratio);
        System.out.printf("YUV => MSE: %.2f, Compression Ratio: %.2f:1\n",
                yuvResult.mse, yuvResult.ratio);
    }

    private static void trainRGB() throws Exception {
        List<BufferedImage> imgs = loadImages(TRAIN_PATH);
        List<byte[]> rBlocks = new ArrayList<>(), gBlocks = new ArrayList<>(), bBlocks = new ArrayList<>();
        for (BufferedImage img : imgs) {
            byte[][] r = extractComponent(img, 'R');
            byte[][] g = extractComponent(img, 'G');
            byte[][] b = extractComponent(img, 'B');
            collectBlocks(r, rBlocks);
            collectBlocks(g, gBlocks);
            collectBlocks(b, bBlocks);
        }
        redCodebook   = kMeans(rBlocks, CODEBOOK_SIZE, MAX_ITER);
        greenCodebook = kMeans(gBlocks, CODEBOOK_SIZE, MAX_ITER);
        blueCodebook  = kMeans(bBlocks, CODEBOOK_SIZE, MAX_ITER);
    }

    private static TestResult testRGB() throws Exception {
        List<BufferedImage> imgs = loadImages(TEST_PATH);
        double totalMse = 0, totalRatio = 0;
        for (BufferedImage img : imgs) {
            int w = img.getWidth(), h = img.getHeight();
            byte[][] r = extractComponent(img, 'R');
            byte[][] g = extractComponent(img, 'G');
            byte[][] b = extractComponent(img, 'B');

            byte[][] cr = compress(r, redCodebook);
            byte[][] cg = compress(g, greenCodebook);
            byte[][] cb = compress(b, blueCodebook);

            byte[][] dr = decompress(cr, redCodebook);
            byte[][] dg = decompress(cg, greenCodebook);
            byte[][] db = decompress(cb, blueCodebook);

            BufferedImage recon = combineRGB(dr, dg, db);
            double mse   = calculateMSE(img, recon);
            double ratio = calculateRatio(w, h, true);
            System.out.printf("RGB MSE: %.2f, Ratio: %.2f:1\n", mse, ratio);
            totalMse   += mse;
            totalRatio += ratio;
        }
        return new TestResult(totalMse/imgs.size(), totalRatio/imgs.size());
    }

    private static void trainYUV() throws Exception {
        List<BufferedImage> imgs = loadImages(TRAIN_PATH);
        List<byte[]> yBlocks = new ArrayList<>(), uBlocks = new ArrayList<>(), vBlocks = new ArrayList<>();
        for (BufferedImage img : imgs) {
            byte[][][] yuv = splitYUV(img);
            collectBlocks(yuv[0], yBlocks);
            collectBlocks(subsample(yuv[1]), uBlocks);
            collectBlocks(subsample(yuv[2]), vBlocks);
        }
        yCodebook = kMeans(yBlocks, CODEBOOK_SIZE, MAX_ITER);
        uCodebook = kMeans(uBlocks, CODEBOOK_SIZE, MAX_ITER);
        vCodebook = kMeans(vBlocks, CODEBOOK_SIZE, MAX_ITER);
    }

    private static TestResult testYUV() throws Exception {
        List<BufferedImage> imgs = loadImages(TEST_PATH);
        double totalMse = 0, totalRatio = 0;
        for (BufferedImage img : imgs) {
            int w = img.getWidth(), h = img.getHeight();
            byte[][][] yuv = splitYUV(img);
            byte[][] cy = compress(yuv[0], yCodebook);
            byte[][] cu = compress(subsample(yuv[1]), uCodebook);
            byte[][] cv = compress(subsample(yuv[2]), vCodebook);

            byte[][] dy = decompress(cy, yCodebook);
            byte[][] du = upsample(decompress(cu, uCodebook));
            byte[][] dv = upsample(decompress(cv, vCodebook));
            BufferedImage recon = combineYUV(dy, du, dv);

            double mse   = calculateMSE(img, recon);
            double ratio = calculateRatio(w, h, false);
            System.out.printf("YUV MSE: %.2f, Ratio: %.2f:1\n", mse, ratio);
            totalMse   += mse;
            totalRatio += ratio;
        }
        return new TestResult(totalMse/imgs.size(), totalRatio/imgs.size());
    }

    private static List<BufferedImage> loadImages(String rootPath) throws Exception {
        List<BufferedImage> images = new ArrayList<>();
        for (String cat : CATEGORIES) {
            File dir = new File(rootPath, cat);
            if (!dir.exists() || !dir.isDirectory()) continue;
            for (File f : dir.listFiles()) {
                String name = f.getName().toLowerCase();
                if (name.endsWith(".jpg") || name.endsWith(".png")) {
                    BufferedImage img = ImageIO.read(f);
                    if (img != null) {
                        int w = img.getWidth();
                        int h = img.getHeight();
                        w = (w / 4) * 4;
                        h = (h / 4) * 4;
                        if (w > 0 && h > 0) {
                            img = img.getSubimage(0, 0, w, h);
                            images.add(img);
                        }
                    }
                }
            }
        }
        return images;
    }

    private static byte[][] extractComponent(BufferedImage img, char c) {
        int w = img.getWidth(), h = img.getHeight();
        byte[][] comp = new byte[h][w];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int rgb = img.getRGB(x, y);
                comp[y][x] = (byte)((c=='R'? (rgb>>16)&0xFF : c=='G'? (rgb>>8)&0xFF : rgb&0xFF));
            }
        return comp;
    }

    private static void collectBlocks(byte[][] comp, List<byte[]> blocks) {
        int h = comp.length, w = comp[0].length;
        for (int y = 0; y < h - 1; y += 2)
            for (int x = 0; x < w - 1; x += 2)
                blocks.add(new byte[]{comp[y][x], comp[y][x+1], comp[y+1][x], comp[y+1][x+1]});
    }

    private static List<byte[]> kMeans(List<byte[]> blocks, int k, int maxIter) {
        int n = blocks.size();
        if (n == 0) return new ArrayList<>();
        int actualK = Math.min(k, n);
        Random rnd = new Random();
        List<byte[]> centers = new ArrayList<>();
        for (int i = 0; i < actualK; i++) centers.add(blocks.get(rnd.nextInt(n)));
        int[] assign = new int[n];
        for (int it = 0; it < maxIter; it++) {
            boolean changed = false;
            for (int i = 0; i < n; i++) {
                int idx = findClosest(blocks.get(i), centers);
                if (idx != assign[i]) { assign[i] = idx; changed = true; }
            }
            if (!changed) break;
            centers = updateCentroids(blocks, assign, actualK);
        }
        return centers;
    }

    private static int findClosest(byte[] b, List<byte[]> centers) {
        double minD = Double.MAX_VALUE; int best=0;
        for (int i=0;i<centers.size();i++) {
            double d=0; for (int j=0;j<4;j++) { int diff = (b[j]&0xFF)-(centers.get(i)[j]&0xFF); d+=diff*diff; }
            if (d<minD) {minD=d;best=i;}
        }
        return best;
    }

    private static List<byte[]> updateCentroids(List<byte[]> blocks, int[] assign, int k) {
        List<List<byte[]>> cls = new ArrayList<>(); for (int i=0;i<k;i++) cls.add(new ArrayList<>());
        for (int i=0;i<assign.length;i++) cls.get(assign[i]).add(blocks.get(i));
        List<byte[]> out=new ArrayList<>();
        for (List<byte[]> c : cls) {
            if (c.isEmpty()) { out.add(new byte[4]); continue; }
            int[] s=new int[4]; for (byte[] b : c) for (int j=0;j<4;j++) s[j]+=b[j]&0xFF;
            byte[] avg=new byte[4]; for (int j=0;j<4;j++) avg[j]=(byte)(s[j]/c.size()); out.add(avg);
        }
        return out;
    }

    private static byte[][] compress(byte[][] comp, List<byte[]> codebook) {
        int h=comp.length, w=comp[0].length;
        byte[][] out=new byte[h/2][w/2];
        for (int y=0;y<h/2;y++) for (int x=0;x<w/2;x++) {
            byte[] blk={comp[y*2][x*2],comp[y*2][x*2+1],comp[y*2+1][x*2],comp[y*2+1][x*2+1]};
            out[y][x]=(byte)findClosest(blk,codebook);
        }
        return out;
    }

    private static byte[][] decompress(byte[][] in, List<byte[]> codebook) {
        int h=in.length*2, w=in[0].length*2;
        byte[][] out=new byte[h][w];
        for (int y=0;y<in.length;y++) for (int x=0;x<in[0].length;x++) {
            byte[] blk=codebook.get(in[y][x]&0xFF);
            out[y*2][x*2]=blk[0]; out[y*2][x*2+1]=blk[1]; out[y*2+1][x*2]=blk[2]; out[y*2+1][x*2+1]=blk[3];
        }
        return out;
    }

    private static byte[][][] splitYUV(BufferedImage img) {
        int w=img.getWidth(), h=img.getHeight();
        byte[][] y=new byte[h][w], u=new byte[h][w], v=new byte[h][w];
        for (int i=0;i<h;i++) for (int j=0;j<w;j++) {
            int rgb=img.getRGB(j,i), r=(rgb>>16)&0xFF, g=(rgb>>8)&0xFF, b=rgb&0xFF;
            y[i][j]=(byte)(0.299*r+0.587*g+0.114*b);
            u[i][j]=(byte)((-0.147*r-0.289*g+0.436*b)+128);
            v[i][j]=(byte)((0.615*r-0.515*g-0.100*b)+128);
        }
        return new byte[][][]{y,u,v};
    }

    private static byte[][] subsample(byte[][] c) {
        int h=c.length, w=c[0].length; byte[][] o=new byte[h/2][w/2];
        for (int i=0;i<h/2;i++) for (int j=0;j<w/2;j++) o[i][j]=c[i*2][j*2];
        return o;
    }

    private static byte[][] upsample(byte[][] c) {
        int h = c.length * 2, w = c[0].length * 2;
        byte[][] o = new byte[h][w];
        for (int i = 0; i < c.length; i++) {
            for (int j = 0; j < c[0].length; j++) {
                byte v = c[i][j];
                o[i * 2][j * 2] = v;
                o[i * 2][j * 2 + 1] = v;
                o[i * 2 + 1][j * 2] = v;
                o[i * 2 + 1][j * 2 + 1] = v;
            }
        }
        return o;
    }

    private static BufferedImage combineRGB(byte[][] r, byte[][] g, byte[][] b) {
        int h=r.length, w=r[0].length; BufferedImage img=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
        for (int i=0;i<h;i++) for (int j=0;j<w;j++) img.setRGB(j,i,((r[i][j]&0xFF)<<16)|((g[i][j]&0xFF)<<8)|(b[i][j]&0xFF));
        return img;
    }

    private static BufferedImage combineYUV(byte[][] y, byte[][] u, byte[][] v) {
        int h=y.length, w=y[0].length; BufferedImage img=new BufferedImage(w,h,BufferedImage.TYPE_INT_RGB);
        for (int i=0;i<h;i++) for (int j=0;j<w;j++) {
            int Y=y[i][j]&0xFF, U=(u[i][j]&0xFF)-128, V=(v[i][j]&0xFF)-128;
            int rr=clamp(Y+(int)(1.140*V)), gg=clamp(Y-(int)(0.395*U)-(int)(0.581*V)), bb=clamp(Y+(int)(2.032*U));
            img.setRGB(j,i,(rr<<16)|(gg<<8)|bb);
        }
        return img;
    }

    private static double calculateMSE(BufferedImage o, BufferedImage c) {
        int w=Math.min(o.getWidth(),c.getWidth()), h=Math.min(o.getHeight(),c.getHeight());
        double sum=0; for (int i=0;i<h;i++) for (int j=0;j<w;j++) {
            int or=o.getRGB(j,i), cr=c.getRGB(j,i);
            int dr=((or>>16)&0xFF)-((cr>>16)&0xFF), dg=((or>>8)&0xFF)-((cr>>8)&0xFF), db=(or&0xFF)-(cr&0xFF);
            sum+=dr*dr+dg*dg+db*db;
        }
        return sum/(w*h*3);
    }

    private static double calculateRatio(int w,int h,boolean rgb) {
        double orig=3.0*w*h;
        double comp = rgb ? (w/2.0)*(h/2.0)*3 : (w/2.0)*(h/2.0) + 2*(w/4.0)*(h/4.0);
        return orig/comp;
    }

    private static int clamp(int v) { return v<0?0:v>255?255:v; }

    private static class TestResult { double mse, ratio; TestResult(double m,double r){mse=m;ratio=r;} }
}
```

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