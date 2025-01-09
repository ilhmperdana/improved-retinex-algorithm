import cv2
import numpy as np
from sklearn.preprocessing import normalize

# Eq (14) : calculate Is
def calculate_sharpen_image(L_w):
    kernel_size = (3,3)
    sigma=1.0
    Is = (L_w + normalize((L_w - cv2.GaussianBlur(L_w, kernel_size, sigmaX=sigma)), axis=1, norm = 'l1'))
    return Is

# Eq (17): Calculate log-average brightness
def calculate_log_average_brightness(L_w, delta=1e-3):
    log_luminance = np.log(delta + L_w)
    L_w_avg = np.exp(np.mean(log_luminance))  # Persamaan (17)
    return L_w_avg

# Eq (16): Calculate global adaptation brightness
def calculate_global_adaptation(L_w, L_w_avg, L_w_max):
    L_g = np.log(L_w / L_w_avg + 1) / np.log(L_w_max / L_w_avg + 1)  # Persamaan (16)
    return L_g

# Eq (18): Calculate Guided filtering using OpenCV
def guided_filter(Lg, radius=15, epsilon=1e-2):
    Lg = Lg.astype(np.float32)
    H_g = cv2.ximgproc.guidedFilter(guide=Lg, src=Lg, radius=radius, eps=epsilon)
    return H_g

# Eq (19): Calculate β
def calculate_offset(L_g, lambda_param=0.5):

    beta = lambda_param * L_g  # Persamaan (19)
    return beta

# Eq (20): Calculate α
def calculate_contrast_factor(L_g, L_g_max, eta=1.0):
    alpha = (1 + eta * (L_g / L_g_max)) ** (1 + (L_g_max / (L_g_max + eta * L_g)))  # Persamaan (20)
    return alpha

# Eq (21): Calculate Reflection R(x, y)
def calculate_reflection(L_g, H_g, alpha, beta):
    R = alpha * np.log((L_g / H_g) + beta)  # Persamaan (21)
    return R

# Eq (22): Calculate Final Output
def calculate_final_output(R, L_w, I_S):
    I_out = (R / L_w) * I_S  # Persamaan (22)
    return I_out

# Retinex Algorithm Process
def process_channel(channel):
    
    L_w = channel
    L_w_sharp = calculate_sharpen_image(L_w)
    L_w_avg = calculate_log_average_brightness(L_w)
    L_w_max = np.max(L_w)
    L_g = calculate_global_adaptation(L_w, L_w_avg, L_w_max)
    H_g = guided_filter(L_g)
    beta = calculate_offset(L_g)
    L_g_max = np.max(L_g)
    alpha = calculate_contrast_factor(L_g, L_g_max)
    R = calculate_reflection(L_g, H_g, alpha, beta)
    I_out = calculate_final_output(R, L_w, L_w_sharp)
    return I_out

# Implementasi lengkap Improved Retinex Algorithm untuk RGB
def improved_retinex_rgb(image_path):

    image = cv2.imread(image_path)  # Baca gambar (BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
    
    # Separate color channel
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    
    # Process each channel
    R_out = process_channel(R)
    G_out = process_channel(G)
    B_out = process_channel(B)
    
    # Normalization 
    R_out = cv2.normalize(R_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    G_out = cv2.normalize(G_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    B_out = cv2.normalize(B_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combine each channel into 1 image
    enhanced_image = cv2.merge([R_out, G_out, B_out])
    return enhanced_image


image_path = "D:/research/Underwater Image/raw-890/raw-890/8_img_.png"
enhanced_image = improved_retinex_rgb(image_path)
image = cv2.imread(image_path)

# show image
cv2.imshow('Enhanced RGB Image', cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
cv2.imshow('Image awal', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
