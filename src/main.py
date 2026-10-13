from preprocessing import normalize_text, tokenize, duplicate, del_duplicate, emptycol, del_emptycol
import lib 
from data_loader import load_data
from word_segmetation.scrdr_segment import segment_text
from pos_tagging import features, features2, train_test_valid, train_crf, predict

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

#tách từ
df["word_segmentation"] = df["text"].map(segment_text)
lib.display(df.head(10))


#huấn luyện CRF
#mô hình với cửa sổ là +-1
X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid(df,features)
print("Số câu train:", len(X_train), len(y_train))

crf = train_crf(X_train,y_train)
y_pred = predict(crf,X_test)

#mô hình có thêm đặc trưng cửa sổ là +-2
X_train2, y_train2, X_valid2, y_valid2, X_test2, y_test2 = train_test_valid(df,features2)
crf2 = train_crf(X_train2,y_train2)
y_pred2 = predict(crf2,X_test2)



