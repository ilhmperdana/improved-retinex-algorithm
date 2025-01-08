import cv2
import numpy as np
from sklearn.preprocessing import normalize

# Persamaan (14) : calculate Is

def calculate_sharpen_image(L_w):
    kernel_size = (3,3)
    sigma=1.0
    Is = (L_w + normalize((L_w - cv2.GaussianBlur(L_w, kernel_size, sigmaX=sigma)), axis=1, norm = 'l1'))
    return Is

# Persamaan (17): Menghitung log-average brightness
def calculate_log_average_brightness(L_w, delta=1e-3):
    """
    Menghitung log-average brightness (Lw) berdasarkan Persamaan (17).
    """
    log_luminance = np.log(delta + L_w)
    L_w_avg = np.exp(np.mean(log_luminance))  # Persamaan (17)
    return L_w_avg

# Persamaan (16): Menghitung global adaptation brightness
def calculate_global_adaptation(L_w, L_w_avg, L_w_max):
    """
    Menghitung global adaptation brightness (Lg(x, y)) berdasarkan Persamaan (16).
    """
    L_g = np.log(L_w / L_w_avg + 1) / np.log(L_w_max / L_w_avg + 1)  # Persamaan (16)
    return L_g

# Persamaan (18): Guided filtering menggunakan OpenCV
def guided_filter(Lg, radius=15, epsilon=1e-2):
    """
    Guided filtering untuk menghitung H_g(x, y) berdasarkan Persamaan (18).
    """
    Lg = Lg.astype(np.float32)
    H_g = cv2.ximgproc.guidedFilter(guide=Lg, src=Lg, radius=radius, eps=epsilon)
    return H_g

# Persamaan (19): Menghitung offset β
def calculate_offset(L_g, lambda_param=0.5):
    """
    Menghitung offset β berdasarkan Persamaan (19).
    """
    beta = lambda_param * L_g  # Persamaan (19)
    return beta

# Persamaan (20): Menghitung faktor kontras α
def calculate_contrast_factor(L_g, L_g_max, eta=1.0):
    """
    Menghitung faktor kontras (α) berdasarkan Persamaan (20).
    """
    alpha = (1 + eta * (L_g / L_g_max)) ** (1 + (L_g_max / (L_g_max + eta * L_g)))  # Persamaan (20)
    return alpha

# Persamaan (21): Menghitung komponen refleksi R(x, y)
def calculate_reflection(L_g, H_g, alpha, beta):
    """
    Menghitung komponen refleksi (R(x, y)) berdasarkan Persamaan (21).
    """
    R = alpha * np.log((L_g / H_g) + beta)  # Persamaan (21)
    return R

# Persamaan (22): Menghitung gambar akhir yang ditingkatkan
def calculate_final_output(R, L_w, I_S):
    """
    Menghitung gambar akhir yang ditingkatkan berdasarkan Persamaan (22).
    """
    I_out = (R / L_w) * I_S  # Persamaan (22)
    return I_out

# Proses Retinex untuk saluran tunggal
def process_channel(channel):
    """
    Menerapkan Improved Retinex Algorithm pada satu saluran gambar.
    Args:
        channel (np.ndarray): Saluran tunggal gambar (R, G, atau B).
    Returns:
        np.ndarray: Saluran gambar hasil.
    """
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
    """
    Implementasi Improved Retinex Algorithm untuk gambar RGB.
    Args:
        image_path (str): Path ke file gambar.
    Returns:
        np.ndarray: Gambar hasil yang ditingkatkan dalam format RGB.
    """
    image = cv2.imread(image_path)  # Baca gambar (BGR format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Konversi ke RGB
    
    # Pisahkan saluran warna
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    
    # Proses masing-masing saluran
    R_out = process_channel(R)
    G_out = process_channel(G)
    B_out = process_channel(B)
    
    # Normalisasi hasil agar sesuai untuk tampilan
    R_out = cv2.normalize(R_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    G_out = cv2.normalize(G_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    B_out = cv2.normalize(B_out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Gabungkan kembali saluran warna
    enhanced_image = cv2.merge([R_out, G_out, B_out])
    return enhanced_image

# Jalankan algoritma Improved Retinex untuk gambar RGB
image_path = "D:/research/Underwater Image/raw-890/raw-890/13_img_.png"
enhanced_image = improved_retinex_rgb(image_path)
image = cv2.imread(image_path)

# Tampilkan hasil dengan cv2.imshow
cv2.imshow('Enhanced RGB Image', cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB))
cv2.imshow('Image awal', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
