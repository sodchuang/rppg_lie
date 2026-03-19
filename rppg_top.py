import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def extract_bvp_from_video(video_path, roi=(100, 100, 200, 200), fps=30):
    # 打開視頻文件
    cap = cv2.VideoCapture(video_path)
    
    # 獲取fps
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps <= 0:
        actual_fps = fps  # 如果無法獲取fps，使用預設值

    # 初始化變量
    frame_count = 0
    r_signal = []
    g_signal = []
    b_signal = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 提取感興趣區域 (ROI) 的顏色信號
        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]
        mean_color = np.mean(roi_frame, axis=(0, 1))  # RGB通道的均值
        b_signal.append(mean_color[0])  # 藍色通道
        g_signal.append(mean_color[1])  # 綠色通道
        r_signal.append(mean_color[2])  # 紅色通道
        
        frame_count += 1
    
    cap.release()
    
    # 將信號轉換為numpy數組
    r_signal = np.array(r_signal)
    g_signal = np.array(g_signal)
    b_signal = np.array(b_signal)
    
    # 平滑信號
    r_smoothed = gaussian_filter1d(r_signal, sigma=5)
    g_smoothed = gaussian_filter1d(g_signal, sigma=5)
    b_smoothed = gaussian_filter1d(b_signal, sigma=5)
    
    # 畫出RGB通道的平滑信號
    plt.figure(figsize=(15, 5))
    plt.subplot(3, 1, 1)
    plt.plot(r_smoothed, color='red')
    plt.title('Red Channel Signal')
    plt.subplot(3, 1, 2)
    plt.plot(g_smoothed, color='green')
    plt.title('Green Channel Signal')
    plt.subplot(3, 1, 3)
    plt.plot(b_smoothed, color='blue')
    plt.title('Blue Channel Signal')
    plt.tight_layout()
    plt.show()
    
    # 進行傅立葉變換以提取心跳信號
    def analyze_signal(signal, fps):
        if len(signal) <= 1:
            return None
        
        freq = np.fft.fftfreq(len(signal), d=1/fps)
        fft_values = np.fft.fft(signal)
        magnitude = np.abs(fft_values)
        peak_indices = find_peaks(magnitude[1:len(magnitude)//2], height=0.5)[0]
        if len(peak_indices) > 0:
            main_frequency = freq[peak_indices[0] + 1]  # 加1因為freq從1開始
            return main_frequency
        else:
            return None
    
    # 分析每個通道的頻率
    r_frequency = analyze_signal(r_smoothed, actual_fps)
    g_frequency = analyze_signal(g_smoothed, actual_fps)
    b_frequency = analyze_signal(b_smoothed, actual_fps)
    
    # 輸出主要的心跳頻率
    print(f"紅色通道主要心跳頻率: {r_frequency:.2f} Hz" if r_frequency else "紅色通道未檢測到心跳信號")
    print(f"綠色通道主要心跳頻率: {g_frequency:.2f} Hz" if g_frequency else "綠色通道未檢測到心跳信號")
    print(f"藍色通道主要心跳頻率: {b_frequency:.2f} Hz" if b_frequency else "藍色通道未檢測到心跳信號")
    
    return r_smoothed, g_smoothed, b_smoothed

# 使用範例
video_path = 'C:\\Users\\User\\Downloads\\MAFW-main\\MAFW-main\\samples-gif\\fear_09246.gif'
r_bvp_signal, g_bvp_signal, b_bvp_signal = extract_bvp_from_video(video_path)
