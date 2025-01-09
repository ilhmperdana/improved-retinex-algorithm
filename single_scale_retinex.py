import cv2
import numpy as np



def process_channel_ssr(channel):
    kernel_size = (0,0)
    sigma = 100
    channel = np.array(channel)+1.0
    print(type(channel))
    Is = (np.log10(channel)-np.log10(cv2.GaussianBlur(channel, kernel_size, sigma) + 1.0))
    print(Is)
    return Is

def SSR(img):
    
    img = cv2.imread(img)  # Baca gambar (BGR format)
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
    
    # Separate color channel
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    
    # Process each channel
    R_out = process_channel_ssr(R)
    G_out = process_channel_ssr(G)
    B_out = process_channel_ssr(B)
    
    # Normalization 
    R_out = cv2.normalize(R_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    print(R_out)
    G_out = cv2.normalize(G_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    B_out = cv2.normalize(B_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combine each channel into 1 image
    enhanced_image = cv2.merge([R_out, G_out, B_out])
    return enhanced_image


image_path= "D:/research/retinex/retinex-algorithm/ExDark/Bicycle/2015_00409.jpg"

enhanced_image = SSR(image_path)

cv2.imshow('Enhanced Image', cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
cv2.imshow('Input Image', cv2.imread(image_path))




cv2.waitKey(0)
cv2.destroyAllWindows()