#import thư viện
import re
import time
import math
import random
import unicodedata
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#chuẩn hóa unicode, khoảng trắng
#đầu vào là các câu
def normalize_text(text): 
    text = unicodedata.normalize("NFC", str(text)) #để chuẩn hóa nhận các token dấu thanh là 1 không phải 2 hay 3.
    text = text.lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " URL ", text) #thấy URL thì thay bằng URL
    text = re.sub(r"\\b\\S+@\\S+\\.\\S+\\b", " EMAIL ", text) #tương tự
    text = re.sub(r"[^0-9a-zA-ZÀ-ỹ_\\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip() #text nhiều khoảng trắng thì chuyển về còn 1 khoảng trắng.
    return text
#trả về các câu đã chuẩn hóa Unicode, khoảng trắng, chữ thường

#segemt từ token
#đầu vào là các câu
def tokenize(text: str): 
    return normalize_text(text).split()
#trả về list token

#xử lý văn bản trùng
#đầu vào là series, nhập vào cột text để kiểm tra trùng lặp
def duplicate(data): 
    dup = data.duplicated(keep=False)
    if not dup.empty:
            print("có trùng:", dup.sum(), "dòng")
    else:
            print("không còn dòng trùng")
    return dup
# trả về series gồm giá trị True là có trùng, False là không trùng

#đầu vào là dataframe đó với cột cần xóa trùng lặp
def del_duplicate(df, column):
    df =  df.drop_duplicates(subset = column,keep="first") 
    #xét tập con là cột cần xóa duplicate, giữ lại những câu thấy lần đầu nhất, xóa những lần lặp lại sau của các câu này
    #đánh lại số dòng sau khi xóa
    df = df.reset_index(drop=True)
    return df
#trả về series đã xóa trùng lặp và đánh lại index

#xử lý cột rỗng
#đầu vào là dataframe
def emptycol(df):
    empty_idx = df[df.isna().any(axis = 1)].index #được một danh sách gồm các chỉ số của các dòng có ô rỗng
    if not empty_idx.empty:
        print("Có", len(empty_idx),"dòng có cột rỗng, cần xử lý.")
    else:
        print("không có dòng nào rỗng")
    return empty_idx

def del_emptycol(df, empty_idx):
    for i in empty_idx:
        df = df.drop(index = i)
    check = df[df.isna().any(axis = 1)].index
    print("Kiểm tra số dòng có rỗng:", len(check))
    return df
#trả về quy trình sử lý cột rỗng
