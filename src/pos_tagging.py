import lib
from data_loader import load_data
from evaluator import  calculate_metrics

#hàm kiểm tra từ là dấu câu hay không
#đầu vào là từ
def is_punct(word):
    if word == "":
        return False
    for char in word:
        category = lib.unicodedata.category(char) #phân loại ký tự
        if not category.startswith("P"): #xem có phải là dấu câu không
            return False
    return True
#đầu ra là bool, True thì là dấu câu, False thì không phải dấu câu.

#tìm đặc trưng ngữ cảnh và bề mặt
#hàm đặc trưng này ở phần ngữ cảnh cửa số hiện tại là +-1
#đầu vào là xét một câu với vị trí của từ trong câu
#cải tiến: dùng từ láy và từ chức năng trong tiếng việt
def features(sent, i, use_syllables = True, use_affix=True):
    token = sent[i] #lấy token
    low = token.lower()#chữ thường
    #tạo đict chứa đặc trưng
    feats = {
            "bias": 1.0, # hằng số cố định
            "token.lower": low, #chữ thường của token
            "is_upper": token.isupper(),#viết hoa nguyên token
            "is_title": token.istitle(),#token viết hoa chữ đầu
            "is_digit": sent[i].isdigit(), #có phải là số hay không
            "is_punct":is_punct(sent[i]), #có phải là dấu câu không
            "length": len(token), #độ dài token
        }
    if i > 0: #nếu không phải vị trí đầu câu
        #thêm vào dict key pre.lower
        feats["prev.lower"] = sent[i - 1].lower()
    else:
        #thêm vào dict key BOS
        feats["BOS"] = True #BOS - Beginnig Of Sentence tức là token này đứng đầu câu.
    if i < len(sent) - 1: # nếu vị trí này không phải cuối câu
        #thêm vào dict key next.lower
        feats["next.lower"] = sent[i + 1].lower()
    else: #nếu đúng là cuối câu
        #thêm vào dict key EOS
        feats["EOS"] = True # EOS-End OF Sentence, token đứng cuối câu.
    if use_syllables:
        syllable = low.split() #tìm các dạng từ láy có hai từ giống nhau: xanh xanh, đều đều, người người,...
        feats["syllables"] = syllable
        #them tiền tố hậu tố trong từ
    if use_affix:
        w = low.split()
        feats["prefix"] = (w[0] if w else "")
        feats["suffix"] = (w[-1] if w else "")
    return feats 
# trả về cái dict feature của một token.

#hàm bỏ lần lượt các feature để kiểm tra độ quan trọng của từng feature tới model
def make_features(remove_features=None):
    remove_features = list(remove_features or [])
    def configured_features( sent, i,use_syllables=True,use_affix = True):
        feats = features(sent, i,use_syllables,use_affix)
        for feature in remove_features:
            feats.pop(feature, None)
        return feats
    return configured_features
#đầu ra là dict features mới đã bỏ bớt features để test

#hàm đặc trưng thử với cửa sổ là +-2
def features2(sent, i,use_syllables = True, use_affix =True):
    token = sent[i] #lấy token
    feats = features(sent,i,use_syllables,use_affix)
    #feature về ngữ cảnh
    if i > 0: #nếu không phải vị trí đầu câu
        #thêm vào dict key pre.lower
        feats["prev2.lower"] = sent[i-2].lower()
    if i < len(sent) - 2: # nếu vị trí này không phải cuối câu
        #thêm vào dict key next.lower
        feats["next2.lower"] = sent[i+2].lower()
    return feats
#đầu ra là dict featurs có thêm ngữ cảnh khi tăng cửa sổ lên +-2

#áp dụng hàm đặc trưng 
def sentfeatures(sent, features_fn, use_syllables = True, use_affix = True): #bỏ vào danh sách tokens với hàm tạo feature cho token
    return [features_fn(sent, i,use_syllables, use_affix) for i in range(len(sent))] #mỗi từ có 1 dict feats, các dict này nằm trong 1 danh sách ứng với 1 câu
#chia tập train valid test
#do df có upos đọc thành kiểu string, chuẩn về danh sách để tránh đọc từng ký tự, không khớp với nhãn được
#đầu vào là chuỗi danh sách nhãn của một câu
def parse_labels(value):
    if isinstance(value, list): #value đó có phải list không
        return value
    if not isinstance(value, str): #value đó có phải chuỗi không nếu không phải luôn thì trả về danh sách rỗng
        return []
    return lib.re.findall(r"""['"]([^'"]*)['"]""", value) 
#tìm các đoạn nằm trong nháy đơn hoặc nháy kép ['"].. ['"]
# lấy phần nội dung bên trong không bao gồm dấu nháy ([^'"]*)
# trả về nội dung danh sách bên trong.

#dùng chung cả pipeline
def split_dataset(df):
    test_df = df.iloc[:2000].copy() # 2000 câu đầu
    valid_df = df.iloc[2000:4000].copy()
    train_df = df.iloc[4000:].copy()
    return test_df,valid_df,train_df

#đầu vào là dataframe
def train_test_valid(df, features_fn,use_syllables = True, use_affix = True):
    test_df = df.iloc[:2000].copy() # 2000 câu đầu
    valid_df = df.iloc[2000:4000].copy()
    train_df = df.iloc[4000:].copy()
    #không làm thay dổi dữ liệu gốc.

    X_train = [sentfeatures(words, features_fn,use_syllables, use_affix) for words in train_df["word_segmentation"]] # duyệt qua mỗi danh sashc, với mỗi danh sách tương ứng một câu, trong đó có các từ, danh sách features mỗi câu trong train_data
    y_train = [parse_labels(labels) for labels in train_df["upos"]] # danh sách label của từng câu tương ứng trong train data
    X_valid = [sentfeatures(words, features_fn,use_syllables, use_affix) for words in valid_df["word_segmentation"]]
    y_valid = [parse_labels(labels) for labels in valid_df["upos"]]
    X_test = [sentfeatures(words, features_fn,use_syllables, use_affix) for words in test_df["word_segmentation"]]# tương tự nhưng là test_Data
    y_test = [parse_labels(labels) for labels in test_df["upos"]]# ds nhãn test_data
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

#huấn luyện CRF
#đầu vào là phần train x,y gồm câu và nhãn
def train_crf(X_train,y_train):
    crf = lib.sklearn_crfsuite.CRF(
        algorithm="lbfgs", c1=0.1, c2=0.1, #gọi thuật toán huấn luyện crf với tham số regularization hạn chế mô hình học quá sát dữ liệu train
        max_iterations=100, #mô hình được phép lặp tối đa 100 lần khi học.
        all_possible_transitions=True # cho phép CRF xem xét tất cả các khả năng chuyển từ nhãn này sang nhãn khác.
    )
    crf.fit(X_train, y_train) #huấn luyện cho hai ds 
    return crf
#trả về mô hình crf đã được huấn luyện

#predict mô hình 
def predict(crf, X_test):
    y_pred = crf.predict(X_test)
    return y_pred

#hàm đánh giá mô hình
#đầu vào là danh sách gồm các danh sách nhãn gold và các danh sách nhãn dự đoán tương ứng câu.
def evaluate_pos(y_true, y_pred):
    #vậy tp là khi nhãn chuẩn là i dự đoán cũng được nhãn i
    #fn khi chuẩn nhãn i, dự đoán thành nhãn khác
    #fp là khi nhãn chuẩn khác, dự đoán là i
    #tn là khi chuẩn nhãn khác i, dự đoán cũng khác i
    # Chuyển danh sách nhãn của nhiều câu thành một danh sách
    y_true = [tag for sentence in y_true for tag in sentence]
    y_pred = [tag for sentence in y_pred for tag in sentence]

    #kiểm tra xem có cùng độ dài hay số lượng nhãn không
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Số nhãn chuẩn = {len(y_true)}, "
            f"số nhãn dự đoán = {len(y_pred)}"
        )

    #lấy tập hợp để dọn trùng lặp và có tập hợp các loại nhãn trong y_test với y_pred
    #hợp hai tập hợp các nhãn gold với nhãn dự đoán và sorted để có danh sách các loại nhãn theo thứ tự
    labels = sorted(set(y_true) | set(y_pred))
    cm = lib.confusion_matrix(y_true, y_pred, labels=labels)
    #được một mảng biểu diễn như ma trận hàng là nhãn chuẩn các cột là nhãn dự đoán, giá trị theo số lần dự đoán từ nhãn này thành nhãn kia.
    total = len(y_true)

    results = {}
    #
    for i, label in enumerate(labels):
        tp = int(cm[i, i]) #đường chéo
        fn = int(cm[i, :].sum() - tp) # lấy các giá trị nhãn dự đoán sai trừ đi giá trị nhãn đúng của mỗi hàng đó.
        fp = int(cm[:, i].sum() - tp) # lấy các giá trị nhãn chuẩn khác với nhãn dự đoán trừ đi giá trị nhãn đúng của mỗi cột đó.
        tn = int(total - tp - fn - fp) # lấy tổng số các nhãn trừ đi nhãn khớp chuẩn, nhãn không khớp chuẩn với chuẩn khác nhãn là được

        metrics = calculate_metrics(tp,tn, fp, fn)

        results[label] = {
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            **metrics # giải nén được các cặp key-value từ metrics vào dict này
        }
    return results
#trả về các thông số kết quả metrics

#phân tích lỗi kiểm tra các trương hợp lỗi nhiều nhất
#hàm kiểm tra lỗi phổ biến của pos
def error_analysis(X_test, y_test, y_pred):
    # Xem các lỗi phổ biến
    errors = []

    for sent_idx, (X_sent, y_true_sent, y_pred_sent) in enumerate(
        zip(X_test, y_test, y_pred)
    ):
        for token_idx, (feat1, gold1, pred1) in enumerate(
            zip(X_sent, y_true_sent, y_pred_sent)
        ):
            if gold1 == pred1:
                continue

            token = feat1.get("token.lower", "")
            prev = feat1.get("prev.lower", "")
            next_ = feat1.get("next.lower", "")

            # Nhận diện dạng dữ liệu
            # Có phải dạng số hoặc ngày tháng không
            is_number_like = bool(
                lib.re.fullmatch(r"[\d.,/%\-]+", token)
            )

            # Là viết tắt có chữ cái viết hoa và nhiều hơn 1 ký tự
            is_viettat = token.isupper() and len(token) > 1

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
                "gold": gold1,
                "predicted": pred1,
                "prev": prev,
                "next": next_,
                "error_pair": f"{gold1} -> {pred1}",
                "categories": categories
            })

    error_df = lib.pd.DataFrame(errors)

    if error_df.empty:
        print("Không có token nào được dự đoán sai.")
        return error_df

    # Các cặp nhãn bị nhầm nhiều nhất
    print("Các cặp nhãn bị nhầm nhiều nhất:")
    lib.display(error_df["error_pair"].value_counts().head(10))

    # Các token bị nhầm nhiều nhất
    print("Các token bị nhầm nhiều nhất:")
    lib.display(error_df["token"].value_counts().head(10))

    # Thống kê các nhóm lỗi
    category_counts = lib.Counter(
        category
        for categories in error_df["categories"]
        for category in categories
    )

    print("Số lỗi thuộc từng nhóm khảo sát:")
    lib.display(
        lib.pd.DataFrame(
            category_counts.items(),
            columns=["Nhóm lỗi", "Số token lỗi"]
        ).sort_values("Số token lỗi", ascending=False)
    )

    print("Tổng số token dự đoán sai:", len(error_df))
    lib.display(error_df)

    return error_df
