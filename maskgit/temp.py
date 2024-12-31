from PIL import Image, ImageEnhance
import numpy as np
# 打开图像
image = Image.open("input.png")

# 创建一个对比度调节器对象
enhancer = ImageEnhance.Contrast(image)

# 设置对比度因子（小于1降低对比度）
factor = 0.8  # 降低10%的对比度
enhanced_image = enhancer.enhance(factor)
enhancer_2 = ImageEnhance.Brightness(enhanced_image)

# 设置亮度因子（小于1降低亮度，大于1增加亮度）
factor = 0.8  # 降低亮度到原来的50%
enhanced_image_2 = enhancer_2.enhance(factor).convert('RGB')
# 保存调整后的图像
# enhanced_image_2.save("output.png")
def add_gaussian_noise(image, mean=0, std=10):
    # 将PIL图像转换为NumPy数组
    image_array = np.array(image)
    # 生成高斯噪声
    gaussian_noise = np.random.normal(mean, std, image_array.shape)
    # 将噪声添加到图像中
    noisy_image = image_array + gaussian_noise
    # 确保像素值在0到255之间
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    # 将NumPy数组转换回PIL图像
    return Image.fromarray(noisy_image)

noisy_image = add_gaussian_noise(enhanced_image_2)
# 保存图像
noisy_image.save('noisy_image.jpg')