import cv2
import numpy as np



def process_channel_ssr(channel, sigma):
    kernel_size = (0,0)
   
    channel = np.array(channel)+1.0
    print(type(channel))
    Is = (np.log10(channel)-np.log10(cv2.GaussianBlur(channel, kernel_size, sigma) + 1.0))
    print(Is)
    return Is

def MSR(img, sigma_scales):
    
    image = cv2.imread(img)  # Baca gambar (BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
    
    # Separate color channel
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    enhanced_image_msr = np.zeros(image.shape)
    for sigma in sigma_scales:
    # Process each channel
        R_out = process_channel_ssr(R, sigma)
        G_out = process_channel_ssr(G, sigma)
        B_out = process_channel_ssr(B, sigma)
        enhanced_image = cv2.merge([R_out, G_out, B_out])
        enhanced_image_msr = enhanced_image_msr + enhanced_image
    
    enhanced_image_msr = enhanced_image_msr / (len(sigma_scales))
    enhanced_image_msr = cv2.normalize(enhanced_image_msr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    
    # Normalization 
#    R_out = cv2.normalize(R_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#    print(R_out)
#    G_out = cv2.normalize(G_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
#    B_out = cv2.normalize(B_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combine each channel into 1 image
    
    return enhanced_image_msr



img = "D:/research/retinex/retinex-algorithm/ExDark/Bicycle/2015_00409.jpg"
sigma_scales = [20, 60, 80]
enhanced_image_msr = MSR(img, sigma_scales)

cv2.imshow('Enhanced Image', cv2.cvtColor(enhanced_image_msr, cv2.COLOR_BGR2RGB))
cv2.imshow('Input Image', cv2.imread(img))




cv2.waitKey(0)
cv2.destroyAllWindows()