from preprocessing import normalize_text, tokenize, duplicate, del_duplicate, emptycol, del_emptycol
import lib 
from data_loader import load_data

df = load_data()
#chuẩn hóa
df["normalize"] = df["text"].map(normalize_text)
print(df["normalize"])

#tokenization
tokenized_corpus = [tokenize(s) for s in df["text"]]

word_freq = lib.Counter()
for sent in tokenized_corpus:
    word_freq.update(sent)
vocab = set(word_freq) # chỉ lấy các từ vựng không lấy số lần xuất hiện
print("Số câu:", len(tokenized_corpus))
print("Vocabulary size:", len(vocab))
print("10 từ phổ biến:", word_freq.most_common(10))
#tiếp đến người thứ 3 cần gán nhãn

#kiểm tra trùng lặp
print("Số dòng ban đầu:",len(df))
dup1 = duplicate(df["text"]) # được một Series có giá trị Bool
df = del_duplicate(df,"text")
#kiểm tra lại lần nữa còn trùng không
print("Kiểm tra trùng lặp sau khi đã xử lý lần 1.")
print(duplicate(df["text"]).sum())
print("Đánh lại số dòng thành công, số dòng sau khi loại bỏ trùng và rỗng:",len(df))

# có cột rỗng không
empty  = emptycol(df)
df = del_emptycol(df,empty)


