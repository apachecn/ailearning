# -*- coding: utf-8 -*-

"""
数据格式转化
"""
import os
import emoji
from middleware.utils import get_catalog_files
from config.setting import Config

tag_dic = {"实体对象": "ORG",
           "正向观点": "Po_VIEW",
           "中性观点": "Mi_VIEW",
           "负向观点": "Ne_VIEW"}


# 转换成可训练的格式，最后以"END O"结尾
def from_ann2dic(r_ann_path, r_txt_path, w_path):
    q_dic = {}
    print("开始读取文件:%s" % r_ann_path)
    with open(r_ann_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line_arr = line.split()
            # print(">>> ", line_arr)
            cls = tag_dic[line_arr[1]]
            start_index = int(line_arr[2])
            end_index = int(line_arr[3])
            length = end_index - start_index
            for r in range(length):
                q_dic[start_index+r] = ("B-%s" % cls) if r == 0 else ("I-%s" % cls)

    # 存储坐标和对应的列名:  {23: 'B-Ne_VIEW', 24: 'I-Ne_VIEW', 46: 'B-ORG', 47: 'I-ORG'}
    print("q_dic: ", q_dic)

    print("开始读取文件内容: %s" % r_txt_path)
    with open(r_txt_path, "r", encoding="utf-8") as f:
        content_str = f.read()

    print("开始写入文本%s" % w_path)
    with open(w_path, "w", encoding="utf-8") as w:
        for i, strA in enumerate(content_str):
            # print(">>> %s-%s" % (i, strA))
            if strA == "\n":
                w.write("\n")
            else:
                if i in q_dic:
                    tag = q_dic[i]
                else:
                    tag = "O"  # 大写字母O
                w.write('%s %s\n' % (strA, tag))
        w.write('%s\n' % "END O")


# 生成train.txt、dev.txt、test.txt
# 除8，9-new.txt分别用于dev和test外,剩下的合并成train.txt
def create_train_data(data_root_dir, w_path):
    if os.path.exists(w_path):
        os.remove(w_path)
    for file in os.listdir(data_root_dir):
        path = os.path.join(data_root_dir, file)
        if file.endswith("8-new.txt"):
            # 重命名为dev.txt
            os.rename(path, os.path.join(data_root_dir, "dev.txt"))
            continue
        if file.endswith("9-new.txt"):
            # 重命名为test.txt
            os.rename(path, os.path.join(data_root_dir, "test.txt"))
            continue
        q_list = []
        print("开始读取文件:%s" % file)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip()
                if line == "END O":
                    break
                q_list.append(line)

        # 获取list 列表: ['美 O', '！ O', '气 O', '质 O', '特 O', '别 O', '好 O', '', '造 O', '型 O', '独 O', '特 O', '， O', '尺 B-ORG', '码 I-ORG', '偏 B-Ne_VIEW', '大 I-Ne_VIEW', '， O']
        # print("q_list: ", q_list)
        print("开始写入文本: %s" % w_path)
        with open(w_path, "a", encoding="utf-8") as f:
            for item in q_list:
                f.write('%s\n' % item)


def brat_1_format_origin(catalog):
    """
    格式化原始文件（去除表情符号的影响，brat占2个字符，但是python占1个字符）
    """
    with open('%s/origin/origin.txt' % path_root, "r", encoding="utf-8") as f:
        lines = f.readlines()
    with open('%s/tag_befer/befer.txt' % path_root, "w", encoding="utf-8") as f:
        # 转换原始文件
        for line in lines:
            text = emoji.demojize(line)
            f.write('%s' % text)
        # 创建标注的新文件
        with open('%s/tag_befer/befer.ann' % path_root, "w", encoding="utf-8") as f:
            pass

def brat_2_create_train_data(catalog):
    file_list = get_catalog_files("%s/tag_after" % catalog, status=-1, str1=".DS_Store")
    file_list = list(set([i.split("/")[-1].split(".")[0] for i in file_list]))
    print(file_list)
    for filename in file_list:
        r_ann_path = os.path.join(catalog, "tag_after/%s.ann" % filename)
        r_txt_path = os.path.join(catalog, "tag_after/%s.txt" % filename)
        w_path = os.path.join(catalog,  "new/%s-new.txt" % filename)
        print("filename", r_ann_path, r_txt_path, w_path)
        from_ann2dic(r_ann_path, r_txt_path, w_path)
    # 生成train.txt、dev.txt、test.txt
    create_train_data("%s/new" % catalog, "%s/new/train.txt" % catalog)


def main():
    catalog = Config.nlp_ner.path_root

    # brat_1_format_origin(catalog)
    brat_2_create_train_data(catalog)
