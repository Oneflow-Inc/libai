import os
import json


def convert_txt2json(file_path):
    """Store the dataset in loose JSON format file, you can refer: 
    https://libai.readthedocs.io/en/latest/tutorials/basics/Preprocessing_Dataset.html
    """
    filename, ext = os.path.splitext(file_path)
    filename = filename.split('/')[-1]

    with open(file_path) as f:
        lines = f.readlines()
        print(len(lines))

    target_file = '/home/xiezipeng/libai/projects/MagicPrompt/' + filename +"_magicprompy.txt"
    with open(target_file, "w", encoding="utf-8") as f:
        for line in lines:
            line = '{' + "\"" + "text" + "\""  + ": " + "\"" + line.strip() + "\"" + "}" + "\n"
            f.write(line)
    os.rename(target_file, target_file[:-4] + '.json')

convert_txt2json("/home/xiezipeng/libai/xzp/test.txt")
convert_txt2json("/home/xiezipeng/libai/xzp/train.txt")
