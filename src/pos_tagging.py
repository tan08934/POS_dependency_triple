import lib
from data_loader import load_data
from evaluator import calculate_prf1

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
def features(sent, i):
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
        #thêm vào dict key BOS
        feats["BOS"] = True #BOS - Beginnig Of Sentence tức là token này đứng đầu câu.
    if i < len(sent) - 1: # nếu vị trí này không phải cuối câu
        #thêm vào dict key next.lower
        feats["next.lower"] = sent[i + 1].lower()
    else: #nếu đúng là cuối câu
        #thêm vào dict key EOS
        feats["EOS"] = True # EOS-End OF Sentence, token đứng cuối câu.
    return feats 
# trả về cái dict feature của một token.

#hàm đặc trưng thử với cửa sổ là +-2
def features2(sent, i):
    token = sent[i] #lấy token
    feats = features(sent,i)
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
def sentfeatures(sent, features_fn): #bỏ vào danh sách tokens với hàm tạo feature cho token
    return [features_fn(sent, i) for i in range(len(sent))] #mỗi từ có 1 dict feats, các dict này nằm trong 1 danh sách ứng với 1 câu
#chia tập train valid test
#đầu vào là dataframe
def train_test_valid(df, features_fn):
    train_df, temp_df = lib.train_test_split(
        df, 
        test_size=0.2, 
        random_state=42) #chia 80-20
    valid_df, test_df = lib.train_test_split(
        temp_df, 
        test_size=0.5, 
        random_state=42) #chia 50 50 cho valid với test

    X_train = [sentfeatures(words, features_fn) for words in train_df["word_segmentation"]] # duyệt qua mỗi danh sashc, với mỗi danh sách tương ứng một câu, trong đó có các từ, danh sách features mỗi câu trong train_data
    y_train = [labels for labels in train_df["upos"]] # danh sách label của từng câu tương ứng trong train data
    X_valid = [sentfeatures(words, features_fn) for words in valid_df["word_segmentation"]]
    y_valid = [labels for labels in valid_df["upos"]]
    X_test = [sentfeatures(words, features_fn) for words in test_df["word_segmentation"]]# tương tự nhưng là test_Data
    y_test = [labels for labels in test_df["upos"]]# ds nhãn test_data
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

#hàm so sánh test/validation
