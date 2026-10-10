import lib 
from pathlib import Path

#Thư mục gốc của project
ROOT_DIR = Path(__file__).resolve().parent.parent

#đường dẫn
DATA_DIR = ROOT_DIR/"data"
#đọc file
def load_data():
    file_path = DATA_DIR/ "UDD-1.csv"
    df = lib.pd.read_csv(file_path, encoding = "utf-8")
    #lấy các cột:
    # text: văn bản
    # token
    # upros: từ loại từng token
    # head: từ mà token phụ thuộc vào
    # deprel: quan hệ cú pháp giữa token và head
    df = lib.pd.DataFrame(df, columns = ["text", "tokens", "upos", "head", "deprel"] ) 
    lib.display(df.columns.tolist())
    # tận 20000 câu
    df = df.head(3000) # lấy mẫu nhỏ hơn
    return df
