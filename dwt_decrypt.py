import pywt
import numpy as np
from PIL import Image

def text_to_bits(text):
    byte_arr = text.encode('utf-8')
    binary_str = ''.join(format(byte, '08b') for byte in byte_arr)
    return binary_str

def bits_to_text(binary_str):
    bytes_list = []
    for i in range(0, len(binary_str), 8):
        byte_str = binary_str[i:i+8]
        if len(byte_str) < 8:
            byte_str += '0' * (8 - len(byte_str))
        byte = int(byte_str, 2)
        bytes_list.append(byte)
    return bytes(bytes_list).decode('utf-8', errors='ignore')

def decode_message(image_path, wavelet='haar', subband='LH'):
    # Đọc ảnh chứa tin nhắn
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float64)
    
    if img_array.shape[0] % 2 != 0 or img_array.shape[1] % 2 != 0:
        raise ValueError("Kích thước ảnh phải chẵn để thực hiện DWT.")
    
    # Tách kênh Red (hoặc kênh đã nhúng tin)
    red_channel = img_array[:, :, 0]
    
    # Áp dụng DWT trên kênh Red
    coeffs_red = pywt.dwt2(red_channel, wavelet)
    cA_red, (cH_red, cV_red, cD_red) = coeffs_red
    
    # Chọn subband đã nhúng tin
    if subband == 'LH':
        selected_subband = cH_red
    elif subband == 'HL':
        selected_subband = cV_red
    elif subband == 'HH':
        selected_subband = cD_red
    else:
        raise ValueError("Subband không hợp lệ. Chọn LH, HL hoặc HH.")
    
    flat_subband = selected_subband.flatten()
    
    # Đọc độ dài tin nhắn (32 bit đầu)
    length_bits = ''.join(str(int(coeff) & 1) for coeff in flat_subband[:32])
    message_length = int(length_bits, 2)
    
    if message_length < 0 or message_length > len(flat_subband) - 32:
        raise ValueError("Độ dài tin nhắn không hợp lệ.")
    
    # Đọc nội dung tin nhắn
    message_bits = []
    for i in range(32, 32 + message_length):
        coeff = flat_subband[i]
        message_bits.append(str(int(coeff) & 1))
    
    return bits_to_text(''.join(message_bits))

# Ví dụ sử dụng:
decoded = decode_message('encrypted.png', subband='LH')
print("successfully decoded")
print('Tin giải mã:', decoded)