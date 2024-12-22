import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import convolve, butter, filtfilt, savgol_filter
from sklearn.metrics import mean_squared_error

# Генерация исходного сигнала (с добавленным шумом)
np.random.seed(3454)
t = np.linspace(0, 10, 500)
signal = np.sin(t)  # Исходный сигнал
noise = np.random.normal(0, 0.5, size=t.shape)  # Шум
noisy_signal = signal + noise  # Сигнал с шумом

# Функции фильтрации
def moving_average_filter(signal, window_size):
    return convolve(signal, np.ones(window_size)/window_size, mode='same')

# Фильтр Butterworth (низкочастотный фильтр)
def butter_lowpass_filter(signal, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)

# Параметры фильтрации
window_size = 10  # Для скользящего окна
sigma = 2  # Для гауссового фильтра
median_size = 5  # Для медианного фильтра
cutoff_frequency = 2  # Частота среза для Butterworth фильтра
sampling_frequency = 100  # Частота дискретизации для фильтра Butterworth
polyorder = 3  # Порядок полинома для фильтра Савицкого-Голея

# Применение фильтров
smoothed_signal_ma = moving_average_filter(noisy_signal, window_size)
smoothed_signal_gaussian = gaussian_filter1d(noisy_signal, sigma)
smoothed_signal_median = median_filter(noisy_signal, size=median_size)
smoothed_signal_butterworth = butter_lowpass_filter(noisy_signal, cutoff_frequency, sampling_frequency)
smoothed_signal_savgol = savgol_filter(noisy_signal, window_length=11, polyorder=polyorder)

# Метрики
mse_ma = mean_squared_error(signal, smoothed_signal_ma)
mse_gaussian = mean_squared_error(signal, smoothed_signal_gaussian)
mse_median = mean_squared_error(signal, smoothed_signal_median)
mse_butterworth = mean_squared_error(signal, smoothed_signal_butterworth)
mse_savgol = mean_squared_error(signal, smoothed_signal_savgol)

# Визуализация
plt.figure(figsize=(15, 12))

plt.subplot(3, 2, 1)
plt.plot(t, noisy_signal, label='Noisy Signal')
plt.plot(t, signal, label='Original Signal', linewidth=2, color='black')
plt.title('Original and Noisy Signal')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(t, smoothed_signal_ma, label=f'Moving Average Filter\nMSE: {mse_ma:.4f}')
plt.plot(t, signal, label='Original Signal', linewidth=2, color='black')
plt.title('Moving Average Filter')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(t, smoothed_signal_gaussian, label=f'Gaussian Filter\nMSE: {mse_gaussian:.4f}')
plt.plot(t, signal, label='Original Signal', linewidth=2, color='black')
plt.title('Gaussian Filter')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(t, smoothed_signal_median, label=f'Median Filter\nMSE: {mse_median:.4f}')
plt.plot(t, signal, label='Original Signal', linewidth=2, color='black')
plt.title('Median Filter')
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(t, smoothed_signal_butterworth, label=f'Butterworth Filter\nMSE: {mse_butterworth:.4f}')
plt.plot(t, signal, label='Original Signal', linewidth=2, color='black')
plt.title('Butterworth Lowpass Filter')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(t, smoothed_signal_savgol, label=f'Savgol Filter\nMSE: {mse_savgol:.4f}')
plt.plot(t, signal, label='Original Signal', linewidth=2, color='black')
plt.title('Savitzky-Golay Filter')
plt.legend()

plt.tight_layout()
plt.show()

# Вывод метрик
print(f'Mean Squared Error (Moving Average): {mse_ma:.4f}')
print(f'Mean Squared Error (Gaussian Filter): {mse_gaussian:.4f}')
print(f'Mean Squared Error (Median Filter): {mse_median:.4f}')
print(f'Mean Squared Error (Butterworth Filter): {mse_butterworth:.4f}')
print(f'Mean Squared Error (Savitzky-Golay Filter): {mse_savgol:.4f}')
