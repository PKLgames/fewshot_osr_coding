#!/bin/bash

# 设置变量
SOURCE_DIR="."  # 压缩文件所在目录，默认为当前目录
TARGET_DIR="../TAU-urban-acoustic-scenes-2022-mobile-development/audio"  # 统一存放wav文件的目标文件夹
EXTRACT_DIR="../temp_extract"  # 临时解压目录

echo "开始处理TAU-urban-acoustic-scenes-2022-mobile-development压缩文件..."

unzip -q -o ./TAU-urban-acoustic-scenes-2022-mobile-development.doc.zip -d ../
unzip -q -o ./TAU-urban-acoustic-scenes-2022-mobile-development.meta.zip -d ../

# 创建目标文件夹和临时目录
mkdir -p "$TARGET_DIR"
mkdir -p "$EXTRACT_DIR"

# 查找并排序所有相关的压缩文件
# 使用ls按数字顺序排序，确保按正确顺序解压
for zipfile in $(ls TAU-urban-acoustic-scenes-2022-mobile-development.audio.*.zip | sort -V); do
    echo "正在处理: $zipfile"
    
    # 解压到临时目录
    unzip -q -o "$SOURCE_DIR/$zipfile" -d "$EXTRACT_DIR"
    
    # 查找并移动所有wav文件到目标文件夹
    # 使用find命令递归查找所有子目录中的wav文件
    find "$EXTRACT_DIR" -name "*.wav" -type f -exec mv {} "$TARGET_DIR" \;
    
    # 清理临时解压目录，为下一个文件做准备
    rm -rf "$EXTRACT_DIR"/*
    
    echo "已完成: $zipfile"
done

# 删除临时目录
rm -rf "$EXTRACT_DIR"

echo "所有文件处理完成！"
echo "WAV文件已保存到: $TARGET_DIR"
echo "共找到 $(find "$TARGET_DIR" -name "*.wav" | wc -l) 个WAV文件"