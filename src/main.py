from preprocessing import normalize_text, tokenize, duplicate, del_duplicate, emptycol, del_emptycol, preprocess_data
import lib 
from data_loader import load_data
from word_segmetation.segment import segment_text
from pos_tagging import features, features2, sentfeatures, train_test_valid, train_crf, predict, evaluate_pos
from evaluator import  calculate_metrics

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
print(df["upos"])
print(type(df.loc[0, "upos"]))
print(len(df.loc[0, "upos"]))

#tách từ
df["word_segmentation"] = df["text"].map(segment_text)
lib.display(df.head(10))


#huấn luyện CRF
#mô hình với cửa sổ là +-1
X_train, y_train, X_valid, y_valid, X_test, y_test = train_test_valid(df,features)
print("Số câu train:", len(X_train), len(y_train))

print("X:", len(X_train[0]))
print("y:", len(y_train[0]))

print("word_segmentation:", df["word_segmentation"].iloc[0])
print("upos:", df["upos"].iloc[0])

crf = train_crf(X_train,y_train)
y_pred = predict(crf,X_test)

#mô hình có thêm đặc trưng cửa sổ là +-2
X_train2, y_train2, X_valid2, y_valid2, X_test2, y_test2 = train_test_valid(df,features2)
crf2 = train_crf(X_train2,y_train2)
y_pred2 = predict(crf2,X_test2)

#kiểm tra xem
print("Cùng nhãn test:", y_test == y_test2)

#gọi hàm đánh giá để xem mô hình nào tốt hơn để lấy
s1 = evaluate_pos(y_test,y_pred)
s2 = evaluate_pos(y_test2,y_pred2)
#biểu diễn để dễ so sánh
#chuyển dictionary thành DataFrame
#xoay hàng cột
df1 = lib.pd.DataFrame(s1).T
df2 = lib.pd.DataFrame(s2).T
#đổi tên cột để phân biệt hai mô hình
df1 = df1.add_suffix("_window1")
df2 = df2.add_suffix("_window2")
# Ghép kết quả theo nhãn POS
comparison = lib.pd.concat([df1, df2], axis=1)
print(comparison.round(2))

#so sánh 
f1_compare = comparison[["f1_window1", "f1_window2"]].copy()
f1_compare["Chênh lệch f1"] = (f1_compare["f1_window2"] - f1_compare["f1_window1"])
print(f1_compare.round(2).to_string())

ac_compare = comparison[["accuracy_window1", "accuracy_window2"]].copy()
ac_compare["Chênh lệch accuracy"] = (ac_compare["accuracy_window2"] - ac_compare["accuracy_window1"])
print(ac_compare.round(2).to_string())

pre_compare = comparison[["precision_window1", "precision_window2"]].copy()
pre_compare["Chênh lệch precision"] = (pre_compare["precision_window2"] - pre_compare["precision_window1"])
print(pre_compare.round(2).to_string())

re_compare = comparison[["recall_window1", "recall_window2"]].copy()
re_compare["Chênh lệch recall"] = (re_compare["recall_window2"] - re_compare["recall_window1"])
print(re_compare.round(2).to_string())

#thử nghiệm kiểm tra với ngữ cảnh xa
different_sentences = []
for i, (x, true_tags, pred1, pred2) in enumerate(zip(X_test, y_test, y_pred, y_pred2)):
    if pred1 != pred2:
        different_sentences.append(i)
print("Số câu có dự đoán khác nhau:", len(different_sentences))
print("10 câu đầu tiên:", different_sentences[:10])

for i in different_sentences[:10]:
    print(f"\nCâu test thứ {i}:")

    for token_idx, (features, gold, p1, p2) in enumerate(
        zip(X_test[i], y_test[i], y_pred[i], y_pred2[i])
    ):
        if p1 != p2:
            print(f"Token thứ {token_idx}:")
            print("  Đặc trưng:", features)
            print("  Nhãn đúng:", gold)
            print("  Dự đoán cửa sổ +-1:", p1)
            print("  Dự đoán cửa sổ +-2:", p2)


#chạy tất cả trong giao diện:
def main(text, crf):
    # Bước 1: Tiền xử lý câu
    segmented = preprocess_data(text)

    # Bước 2: Chuyển câu thành đặc trưng cho CRF
    X = sentfeatures(segmented, features)

    # Bước 3: Dự đoán POS
    pos_result = predict(crf, [X])[0]

    # TODO Bước 4: Dependency parsing
    # Dùng hàm parser thực tế của dự án

    # TODO Bước 5: Triple extraction
    # Dùng hàm trích xuất thực tế của dự án

    return {
        "text": text,
        "tokens": segmented,
        "pos": pos_result,
    }

