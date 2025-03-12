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

def encode_message(image_path, message, output_path, wavelet='haar', subband='LH'):
    # Đọc ảnh với kênh Alpha (nếu có)
    img = Image.open(image_path)
    img_array = np.array(img, dtype=np.float64)
    
    # Kiểm tra xem ảnh có kênh Alpha không
    if img_array.shape[2] == 4:
        has_alpha = True
        alpha_channel = img_array[:, :, 3]
    else:
        has_alpha = False
    
    # Tách các kênh màu
    red_channel = img_array[:, :, 0]
    green_channel = img_array[:, :, 1]
    blue_channel = img_array[:, :, 2]
    
    # Kiểm tra kích thước ảnh
    if red_channel.shape[0] % 2 != 0 or red_channel.shape[1] % 2 != 0:
        raise ValueError("Kích thước ảnh phải chẵn để thực hiện DWT.")
    
    # Áp dụng DWT trên kênh Red (hoặc kênh khác tùy chọn)
    coeffs_red = pywt.dwt2(red_channel, wavelet)
    cA_red, (cH_red, cV_red, cD_red) = coeffs_red
    
    # Chọn subband để nhúng tin
    if subband == 'LH':
        selected_subband = cH_red
    elif subband == 'HL':
        selected_subband = cV_red
    elif subband == 'HH':
        selected_subband = cD_red
    else:
        raise ValueError("Subband không hợp lệ. Chọn LH, HL hoặc HH.")
    
    # Chuyển tin nhắn thành bit và thêm độ dài
    message_bits = text_to_bits(message)
    message_length = len(message_bits)
    if message_length > (2**32 - 1):
        raise ValueError("Tin nhắn quá dài để sử dụng 32 bit lưu độ dài.")
    length_bits = format(message_length, '032b')
    all_bits = length_bits + message_bits
    
    # Kiểm tra dung lượng subband
    flat_subband = selected_subband.flatten()
    if len(all_bits) > len(flat_subband):
        raise ValueError("Tin nhắn quá dài cho subband đã chọn.")
    
    # Nhúng bit vào hệ số wavelet
    for i, bit in enumerate(all_bits):
        coeff = flat_subband[i]
        integer_part = int(coeff)
        fractional_part = coeff - integer_part
        new_integer = (integer_part & ~1) | int(bit)
        flat_subband[i] = new_integer + fractional_part
    
    modified_subband = flat_subband.reshape(selected_subband.shape)
    
    # Tái tạo lại các hệ số
    if subband == 'LH':
        modified_coeffs_red = (cA_red, (modified_subband, cV_red, cD_red))
    elif subband == 'HL':
        modified_coeffs_red = (cA_red, (cH_red, modified_subband, cD_red))
    else:
        modified_coeffs_red = (cA_red, (cH_red, cV_red, modified_subband))
    
    # Áp dụng inverse DWT trên kênh Red
    stego_red = pywt.idwt2(modified_coeffs_red, wavelet)
    
    # Kết hợp lại các kênh màu và kênh Alpha (nếu có)
    if has_alpha:
        stego_array = np.stack([stego_red, green_channel, blue_channel, alpha_channel], axis=-1)
    else:
        stego_array = np.stack([stego_red, green_channel, blue_channel], axis=-1)
    
    # Chuẩn hóa và lưu ảnh
    stego_array = np.clip(stego_array, 0, 255).astype(np.uint8)
    Image.fromarray(stego_array).save(output_path)

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
encode_message('Deadpool-PNG-Photo.png', 'hello', 'output_transparent.png', subband='LH')
decoded = decode_message('output_transparent.png', subband='LH')
print('Tin giải mã:', decoded)