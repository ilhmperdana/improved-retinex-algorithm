Retinex Algorithm

Retinex is derived from "retina" (eye) and "cortex" (brain). It was first proposed by Edwin Land in 1964. The Retinex algorithm is designed to reduce the effect of environmental illumination and extract the reflectance component of an image. The primary goal of the Retinex algorithm is to enhance image quality by compensating for illumination effects, allowing the reflectance of the image to be more accurately captured. This process mimics how the human visual system (eyes and brain) perceives and interprets scenes. The equations are provided below:

$$
R_{(x, y)} = \log(I_{(x, y)}) - \log((F_{(x, y)} * I_{(x, y)})
$$

where I_{(x, y)} is intensity of the pixel in x,y coordinates and F_{(x,y)} is centter surround function. Retinex image to the substraction of pixel and

# Single Scale Retinex

SSR is the vanilla of the retinex algorithm. "Single Scale" means only using single constant or scale to calculate average of center-surround if the given pixel coordinates x,y

The original paper uses the Gaussian Function Gσ​ as a center-surround function. The Gaussian Function or Gaussian Filter serves to smooth the image. The Gaussian Filter provides an estimate of global illumination \log((G_{(x, y)} * I_{(x, y)}) by smoothing variations in pixel illumination with given scale (\sigma).

$$
R_{(x, y)} = \log(I_{(x, y)}) - \log((G_{(x, y)} * I_{(x, y)})
$$

You might be wondering why the logarithmic function is used. This is because it aligns with human perception of illumination. At low illumination levels, changes in light have a significant effect, whereas at high illumination levels, changes in light have a smaller effect or may even be almost negligible.

My Code implementation at single_scale_retinex.py

The image above is example of SSR on ExDark dataset [1] with \sigma = 180

# Multi-Scale Retinex

Multi-Scale Retinex is same as SSR above, unless its scale. SSR only using 1 type of scale while MSR using **multi-scale**. It means MSR is weighted average of n single-scale retinex for different scale or \sigma values

$$
MSR_i(x,y) = \sum_{n=1}^{N} w_n R_i{(x, y)}
$$

The image above is example of MSR on ExDark dataset [1] with \sigma = [40,80,160]

# MSRCR



![](C:\Users\ACER\AppData\Roaming\marktext\images\2025-01-12-02-31-39-image.png)

soon~
