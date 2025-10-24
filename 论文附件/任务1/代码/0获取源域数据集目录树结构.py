import os
from datetime import datetime

def generate_directory_tree(start_path, output_file="directory_tree.txt"):
    """
    生成指定路径的目录树结构并保存到文本文件
    
    参数:
        start_path (str): 要遍历的起始目录路径
        output_file (str): 输出文本文件的名称，默认为directory_tree.txt
    """
    # 检查起始路径是否存在
    if not os.path.exists(start_path):
        print(f"错误: 路径 '{start_path}' 不存在")
        return False
    
    # 确保起始路径是目录
    if not os.path.isdir(start_path):
        print(f"错误: '{start_path}' 不是目录")
        return False
    
    try:
        # 使用with语句安全打开文件
        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入标题和起始路径信息
            f.write(f"目录树结构: {start_path}\n")
            # 使用datetime获取并格式化当前时间
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"生成时间: {current_time}\n")
            f.write("=" * 50 + "\n\n")
            
            # 使用os.walk遍历目录树
            for root, dirs, files in os.walk(start_path):
                # 计算当前目录相对于起始路径的层级
                level = root.replace(start_path, '').count(os.sep)
                
                # 生成缩进字符串
                indent = ' ' * 4 * level
                
                # 写入当前目录名称
                f.write(f"{indent}{os.path.basename(root)}/\n")
                
                # 生成子级缩进
                sub_indent = ' ' * 4 * (level + 1)
                
                # 写入当前目录下的所有文件
                for file in files:
                    f.write(f"{sub_indent}{file}\n")
        
        print(f"目录树已成功生成到: {output_file}")
        return True
        
    except Exception as e:
        print(f"生成目录树时出错: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    # 指定要遍历的目录路径
    target_directory = r"源域数据集"
    
    # 指定输出文件名
    output_filename = "directory_tree.txt"
    
    # 生成目录树
    generate_directory_tree(target_directory, output_filename)