import os
import sys
import argparse


def save_png_names_to_txt(folder_path, output_file='image_names.txt'):
    """
    读取文件夹中的所有PNG图片，并将图片名保存到TXT文件中

    参数:
        folder_path: 要读取的文件夹路径
        output_file: 输出TXT文件名，默认为'image_names.txt'
    """

    # 检查文件夹是否存在
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在！")
        return False

    # 获取文件夹中所有PNG图片
    png_files = []
    for file in os.listdir(folder_path):
        if file.lower().endswith('.png'):
            png_files.append(file.rsplit('.', 1)[0])

    if not png_files:
        print(f"警告：在文件夹 '{folder_path}' 中未找到PNG图片！")
        return False

    # 对图片名进行排序（可选）
    png_files.sort()

    # 写入TXT文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for png_name in png_files:
                f.write(png_name + '\n')

        print(f"成功！共找到 {len(png_files)} 个PNG图片")
        print(f"图片名已保存到: {os.path.abspath(output_file)}")
        return True

    except Exception as e:
        print(f"写入文件时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='读取文件夹中的PNG图片名并保存到TXT文件')
    path = '../../data/VOC2007/test/masks'
    parser.add_argument('folder', nargs='?', default='../../data/VOC2007/test/masks',
                        help='包含PNG图片的文件夹路径（默认为当前目录）')
    parser.add_argument('-o', '--output', default='../../data/VOC2007/test/image_names.txt',
                        help='输出TXT文件名（默认为image_names.txt）')

    args = parser.parse_args()

    # 执行主函数
    save_png_names_to_txt(args.folder, args.output)


if __name__ == "__main__":
    main()