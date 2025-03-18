from pathlib import Path
import numpy as np
import skimage.measure
import pandas as pd

#excel的书写函数
def AnalyzeAndExport(images: np.ndarray, path: Path):
    #ExcelWriter是一个上下文管理器，它可以帮助我们创建并管理Excel文件的写入过程。通过 pd.ExcelWriter(str(path.absolute()))
    # 创建一个Excel文件写入器，str(path.absolute()) 是路径对象 path 的绝对路径转换为字符串形式。
    #with方法可以及时释放资源
    with pd.ExcelWriter(str(path.absolute())) as writer:
        propertyNames = ['area', 'axis_major_length', 'axis_minor_length', 'centroid',
                         'eccentricity', 'equivalent_diameter_area', 'euler_number',
                         'extent', 'feret_diameter_max', 'orientation',
                         'perimeter', 'perimeter_crofton', 'solidity']
#np.max(images) 返回数组 images 中的最大值，images.shape[0] 返回数组的第一维大小，即图像的数量。
        size = (np.max(images)+1, images.shape[0])
        data = {propertyName: pd.DataFrame(np.ndarray(size, dtype=str)) for propertyName in propertyNames}

        # skimage.measure.regionprops() 是 skimage 库中的一个函数，
        # 它使用连通区域标记法（connected component labeling）来识别和测量图像中的区域属性。
        # 这个函数接收一个二值图像作为输入，并返回一个包含该图像中每个区域属性的列表。
        for t in range(images.shape[0]):
            regions = skimage.measure.regionprops(images[t])
            for propertyName in propertyNames:
                for region in regions:
                    value = getattr(region, propertyName)
                    label = region.label
                    data[propertyName].iloc[label, t] = str(value)
#写入excel
        for propertyName in propertyNames:
            data[propertyName].to_excel(writer, sheet_name=propertyName)