import cv2 as cv
import numpy as np
import matplotlib
# Обязательно ДО импорта pyplot переключаем бэкенд для работы без графического интерфейса
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

class ImageAnalyzer:
    def __init__(self, filepath):
        self.path = Path(filepath)
        # Загрузка изображения сразу в градациях серого
        self.img_gray = cv.imread(str(self.path), cv.IMREAD_GRAYSCALE)
        if self.img_gray is None:
            raise FileNotFoundError(f"Не удалось загрузить изображение: {self.path}")

    def get_hist_and_cdf(self, img):
        # Вычисление гистограммы и кумулятивной функции распределения (CDF)
        hist, bins = np.histogram(img.flatten(), 256, [0, 256])
        cdf = hist.cumsum()
        # Нормализация CDF для корректного отображения поверх гистограммы
        cdf_normalized = cdf * float(hist.max()) / cdf.max() if cdf.max() > 0 else cdf
        return hist, cdf_normalized

    def equalize(self):
        # Глобальная эквализация гистограммы по требованиям лабораторной
        return cv.equalizeHist(self.img_gray)

    def visualize_comparison(self, img_equalized, output_dir):
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Анализ экспозиции: {self.path.name}", fontsize=14, fontweight='bold')

        images = [self.img_gray, img_equalized]
        titles = ["Исходное изображение", "После эквализации"]

        for i in range(2):
            # Отображение изображений
            ax[i, 0].imshow(images[i], cmap='gray', vmin=0, vmax=255)
            ax[i, 0].set_title(titles[i])
            ax[i, 0].axis('off')

            # Отображение графиков гистограммы и функции распределения
            hist, cdf = self.get_hist_and_cdf(images[i])
            ax[i, 1].bar(range(256), hist, color='steelblue', alpha=0.8)
            ax[i, 1].plot(cdf, color='crimson', linewidth=2, label='CDF (Функция распределения)')
            ax[i, 1].set_title(f"Гистограмма и CDF ({titles[i].lower()})")
            ax[i, 1].legend()

        plt.tight_layout()
        
        # Сохраняем результат в папку вместо вывода на экран в отдельном окне
        report_path = output_dir / f"report_{self.path.name}"
        plt.savefig(report_path)
        plt.close(fig) # Очищаем память
        print(f"Отчет успешно сохранен: {report_path.absolute()}")

if __name__ == "__main__":
    # Настраиваем пути относительно расположения самого скрипта
    base_dir = Path(__file__).parent
    input_dir = base_dir / "input_images"
    output_dir = base_dir / "output_reports"
    
    # Создаем нужные папки, если их нет
    input_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    
    # Ищем файлы изображений
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]

    if not image_paths:
        print(f"Поместите 3 изображения для анализа в папку: {input_dir.absolute()}")
    else:
        print("Начинаю пакетную обработку изображений...")
        for img_path in image_paths:
            try:
                analyzer = ImageAnalyzer(img_path)
                eq_img = analyzer.equalize()
                analyzer.visualize_comparison(eq_img, output_dir)
            except Exception as e:
                print(f"Ошибка при обработке файла {img_path.name}: {e}")
        
        print("Обработка полностью завершена! Проверьте папку output_reports.")