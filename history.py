import os
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
import datetime


class ImageSwitcher:
    def __init__(self, parent, image_folder='./user_results/history'):
        self.image = None
        self.photo = None
        self.root = tk.Toplevel()
        self.parent = parent
        self.root.title("History")
        self.root.geometry("800x500+320+150")
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if
                            f.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
        self.image_index = 0
        if not self.image_files:
            messagebox.showwarning("Warning", "No image files found in the specified folder.")
            self.root.destroy()
            self.parent.deiconify()
            return

        # Create a label to display the image
        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        self.time_label = tk.Label(self.root)
        self.time_label.pack()

        # 创建下拉选择框的选项
        self.options = ["FCN8x", "UNet", "DeepLabV3", "Unet3+", "Qnet", "PSPnet", "Uesnet50", "Unet2+"]
        self.selected_option = tk.StringVar(value=self.options[0])

        # 创建下拉选择框
        self.dropdown = tk.OptionMenu(self.root, self.selected_option, *self.options)
        self.dropdown.pack(pady=0)

        # 绑定下拉选择框的变化事件
        self.selected_option.trace('w', self.on_option_change)

        # 创建底部框架
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(side='bottom', pady=0, fill='y', expand=True)

        # Create a button to switch to the next image
        self.pre_button = tk.Button(bottom_frame, text="Pre", command=self.pre_image, width=20)
        self.pre_button.pack(side='left', padx=0)

        self.next_button = tk.Button(bottom_frame, text="Next", command=self.next_image, width=20)
        self.next_button.pack(side='left', padx=10)

        self.close_button = tk.Button(bottom_frame, text="Close", command=self.close, width=20)
        self.close_button.pack(side='right', padx=10)
        # Load the first image
        self.load_image()

    def on_option_change(self, *args):
        self.image_index = 0
        self.load_image()

    def load_image(self):
        while self.image_index < len(self.image_files):
            file_name = self.image_files[self.image_index]
            if self.selected_option.get() in file_name:
                image_path = os.path.join(self.image_folder, file_name)
                self.image = Image.open(image_path)
                self.image = self.image.resize((800, 400))
                self.photo = ImageTk.PhotoImage(self.image)
                self.image_label.config(image=self.photo)
                break
            else:
                # 如果没有找到符合条件的图片，检查下一张图片
                self.image_index += 1

        # 如果遍历完所有图片还没有找到，则返回
        if self.image_index >= len(self.image_files):
            return

        parts = file_name.split('_')
        # 获取模型名
        m_name = parts[1]

        # 获取文件的创建时间（时间戳）
        creation_time = os.path.getctime(image_path)
        # 将时间戳转换为 datetime 对象
        creation_time_dt = datetime.datetime.fromtimestamp(creation_time)

        # 格式化日期（不显示年份）
        creation_time_formatted = creation_time_dt.strftime('%b %d, %H:%M:%S')
        self.time_label.config(text=f"{creation_time_formatted}     {m_name}")

    def next_image(self):
        self.image_index = (self.image_index + 1) % len(self.image_files)
        self.load_image()

    def pre_image(self):
        self.image_index = (self.image_index - 1) % len(self.image_files)
        self.load_image()

    def show(self):
        self.root.mainloop()

    def close(self):
        self.root.destroy()
        self.parent.deiconify()
