from PIL import Image
import matplotlib.pyplot as plt


def overlay_images(original_image_path, predicted_image_path, output_image_path):
    """
    读取原图和预测图，将预测图像中的黑色部分设置为透明，并将其他部分的透明度设置为50%，然后生成叠加图像并保存。

    :param original_image_path: 原图路径
    :param predicted_image_path: 预测图路径
    :param output_image_path: 输出图像路径
    """
    original_image = Image.open(original_image_path).convert("RGBA")
    predicted_image = Image.open(predicted_image_path).convert("RGBA")

    # 检查图像是否读取成功
    if original_image is None or predicted_image is None:
        raise FileNotFoundError("图像路径错误或图像文件不存在")

    # 将预测图像调整为与原图相同的大小
    predicted_image = predicted_image.resize(original_image.size)

    # 将预测图像中的黑色部分设置为透明，并将其他部分的透明度设置为50%
    datas = predicted_image.getdata()
    new_data = []
    for item in datas:
        # 更改黑色（全零）为透明
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            new_data.append((0, 0, 0, 0))
        else:
            # 设置其他部分的透明度为50%
            new_data.append((item[0], item[1], item[2], int(255 * 0.8)))
    predicted_image.putdata(new_data)

    # 叠加图像
    combined_image = Image.alpha_composite(original_image, predicted_image)

    # 保存叠加后的图像
    combined_image.save(output_image_path)






