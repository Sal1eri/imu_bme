import math

import Augmentor
from pathlib import Path
from typing import List
import numpy as np
import shutil
import re

#指定形参类型
def AugmentImages(imagesDirectory: Path, segmentationsDirectory: Path, outputDirectory: Path,
                  count: int):
    #目录定位
    augmentor = Augmentor.Pipeline(source_directory=str(imagesDirectory.absolute()),
                                   output_directory=str(outputDirectory.absolute()))
    #保存格式与输入格式相同
    augmentor.set_save_format("auto")
    print(segmentationsDirectory.absolute())
    augmentor.ground_truth(str(segmentationsDirectory.absolute()))

    # Random transformations to apply 旋转图像
    augmentor.rotate(probability=1, max_left_rotation=20, max_right_rotation=20)
    augmentor.flip_left_right(probability=0.5)
    augmentor.flip_top_bottom(probability=0.5)
    augmentor.zoom_random(probability=0.5, percentage_area=0.7)
    augmentor.shear(probability=1, max_shear_left=20, max_shear_right=20)
    augmentor.random_distortion(probability=0.5, grid_width=5, grid_height=5, magnitude=3)
    augmentor.skew(probability=0.5, magnitude=0.3)

    # Resize images to 512x512 (in case some were cropped)
    # 参数 1 表示将所有的输入图像都进行调整大小操作。512 表示调整后的图像的宽度和高度都为 512 像素
    # 通过调用 resize() 方法，并提供所需的图像维度作为参数，可以将输入图像统一调整为相同的大小。
    augmentor.resize(1, 512, 512)

    # Execute the transformations
    # 从原始图像中随机选择并应用这些操作，生成指定数量的增强后图像样本。
    # 这个方法可用于生成训练集或测试集中的样本数据，以扩充原始数据集并增加样本的多样性。
    augmentor.sample(count)

    # Rearrange saved directory
    # 重新管理保存路径
    outputImagesPath = outputDirectory / "images"
    outputSegmentationsPath = outputDirectory / "segmentations"

    files = [path for path in outputDirectory.iterdir() if path.is_file()]
    segmentationFiles = [path for path in files if re.match("_groundtruth*", path.stem)]
    imageFiles = [x for x in files if x not in segmentationFiles]

    for segmentationFile in segmentationFiles:
        newFilename = re.sub(".*_", "", segmentationFile.name)
        segmentationFile.rename(outputSegmentationsPath / newFilename)

    for imageFile in imageFiles:
        newFilename = re.sub(".*_", "", imageFile.name)
        imageFile.rename(outputImagesPath / newFilename)


def SplitData(imagePaths: List[Path], segmentationPaths: List[Path], validationFraction: float,
              testingFraction: float,
              outputDirectory: Path):
    # Sort paths alphabetically
    imagePaths.sort(key=lambda x: x.stem)
    segmentationPaths.sort(key=lambda x: x.stem)

    # Only use paths that have matched names in image and segmentations directory
    imagePaths = np.asarray([path for path in imagePaths if
                             path.stem in [segPath.stem for segPath in segmentationPaths]])
    segmentationPaths = np.asarray([path for path in segmentationPaths if
                                    path.stem in [imagePath.stem for imagePath in imagePaths]])

    # Carry out the split!
    permutation = np.random.permutation(len(imagePaths))
    numValidation = math.ceil(len(imagePaths) * validationFraction)
    numTesting = math.ceil(len(imagePaths) * testingFraction)
    validationIndices = permutation[:numValidation]
    testingIndices = permutation[numValidation:(numValidation + numTesting)]
    trainingIndices = permutation[(numValidation + numTesting):]

    # Save the images
    _CopyToPath(imagePaths[trainingIndices], outputDirectory / "training" / "images")
    _CopyToPath(imagePaths[validationIndices], outputDirectory / "validation" / "images")
    _CopyToPath(imagePaths[testingIndices], outputDirectory / "testing" / "images")
    _CopyToPath(segmentationPaths[trainingIndices], outputDirectory / "training" / "segmentations")
    _CopyToPath(segmentationPaths[validationIndices],
                outputDirectory / "validation" / "segmentations")
    _CopyToPath(segmentationPaths[testingIndices], outputDirectory / "testing" / "segmentations")


def _CopyToPath(paths: List[Path], output: Path):
    for path in paths:
        newPath = output / path.name
        shutil.copy(path, newPath)
