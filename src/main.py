from preprocessing import normalize_text, tokenize, duplicate, del_duplicate, emptycol, del_emptycol, preprocess_data
import lib 
from data_loader import load_data
from word_segmetation.segment import segment_text
from pos_tagging import features, features2, sentfeatures, train_test_valid, train_crf, predict, evaluate_pos

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

lib.display(df1[["accuracy_window1","precision_window1","recall_window1","f1_window1"]])

# Sau khi huấn luyện xong
lib.joblib.dump(crf, "models/pos_crf.pkl")

# Khi chạy giao diện thì mở code này
#crf = lib.joblib.load("models/pos_crf.pkl")

#Xem các lỗ phổ biến:
errors = []

for sent_idx, (X_sent, y_true_sent, y_pred_sent) in enumerate(
    zip(X_test, y_test, y_pred)
):
    for token_idx, (feat, gold, pred) in enumerate(
        zip(X_sent, y_true_sent, y_pred_sent)
    ):
        if gold == pred:
            continue

        token = feat.get("token.lower", "")
        prev = feat.get("prev.lower", "")
        next_ = feat.get("next.lower", "")

        # Nhận diện dạng dữ liệu
        #có phải dạng số hoặc ngày tháng không
        is_number_like = bool(
            lib.re.fullmatch(r"[\d.,/%\-]+", token)
        )
        #là viết tắt có chữ cái viết hoa và nhiều hơn 1 ký tự
        is_viettat = (
            token.isupper() and len(token) > 1
        )
        is_multiword = " " in token

        # Gắn các nhóm có thể khảo sát
        categories = []
        if is_number_like:
            categories.append("Dạng số")
        if is_viettat:
            categories.append("Chữ viết tắt")
        if is_multiword:
            categories.append("Cụm nhiều từ")
        if token_idx == 0:
            categories.append("Đầu câu")
        if token_idx == len(X_sent) - 1:
            categories.append("Cuối câu")

        errors.append({
            "sentence_index": sent_idx,
            "token_index": token_idx,
            "token": token,
            "gold": gold,
            "predicted": pred,
            "prev": prev,
            "next": next_,
            "error_pair": f"{gold} -> {pred}",
            "categories": categories
        })
error_df = lib.pd.DataFrame(errors)
#thống kê các lỗi phổ biến

#Trường hợp những cặp nhãn bị nhầm nhiều nhất
print("Các cặp nhãn bị nhầm nhiều nhất:")
lib.display(error_df["error_pair"].value_counts().head(10))
#Trường hợp các từ bị nhầm nhiều nhất
print("Các token bị nhầm nhiều nhất:")
lib.display(error_df["token"].value_counts().head(10))
#trường hợp các nhóm lỗi được gắn nhãn theo hình thức/vị trí
category_counts = lib.Counter(
    category
    for categories in error_df["categories"]
    for category in categories
)
print("Số lỗi thuộc từng nhóm khảo sát:")
lib.display(lib.pd.DataFrame(
    category_counts.items(),
    columns=["Nhóm lỗi", "Số token lỗi"]
).sort_values("Số token lỗi", ascending=False))
print("Tổng số token dự đoán sai:", len(error_df))
lib.display(error_df)


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



