from PIL import Image, ImageDraw


def create_color_map_image(block_size=20, margin=5, image_width=100, output_file='./color_map.png'):

    """
        创建一个颜色映射图像，其中包含颜色块和对应的类别文字。

        参数:
            classes (list): 类别名称列表。
            colormap (list): 每个类别对应的颜色列表。
            block_size (int): 每个颜色块的大小（宽度和高度）。
            margin (int): 颜色块之间的间距。
            image_width (int): 图像的宽度。
            output_file (str): 保存图像的文件路径。

        返回:
            None
        """

    # 定义类别和颜色映射表
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
               'dog', 'horse', 'motorbike', 'person', 'potted plant',
               'sheep', 'sofa', 'train', 'tv/monitor']

    colormap = [
        [0, 0, 0],  # background: 黑色
        [128, 0, 0],  # aeroplane: 暗红色
        [0, 128, 0],  # bicycle: 暗绿色
        [128, 128, 0],  # bird: 暗黄色
        [0, 0, 128],  # boat: 暗蓝色
        [128, 0, 128],  # bottle: 紫色
        [0, 128, 128],  # bus: 蓝绿色
        [128, 128, 128],  # car: 灰色
        [64, 0, 0],  # cat: 暗棕色
        [192, 0, 0],  # chair: 暗红色
        [64, 128, 0],  # cow: 绿色
        [192, 128, 0],  # diningtable: 橙色
        [64, 0, 128],  # dog: 紫色
        [192, 0, 128],  # horse: 粉红色
        [64, 128, 128],  # motorbike: 青色
        [192, 128, 128],  # person: 浅灰色
        [0, 64, 0],  # potted plant: 暗绿色
        [128, 64, 0],  # sheep: 橙色
        [0, 192, 0],  # sofa: 暗绿色
        [128, 192, 0],  # train: 浅绿色
        [0, 64, 128]  # tv/monitor: 暗蓝色
    ]

    num_colors = len(colormap)
    image_height = (block_size + margin) * num_colors + margin

    # 创建带有透明背景的空白图像
    image = Image.new('RGBA', (image_width, image_height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 在图像上绘制颜色块和类别文字
    for i, (color, cls) in enumerate(zip(colormap, classes)):
        x = margin
        y = i * (block_size + margin) + margin
        draw.rectangle([x, y, x + block_size, y + block_size], fill=tuple(color) + (255,))
        draw.text((x + block_size + margin, y), cls, fill='black')

    # 显示图像或保存图像到文件
    # image.show()
    image.save(output_file)


