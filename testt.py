import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import re

def read_and_process(filename):
  """
  Đọc file txt, xử lý dữ liệu và trả về kết quả.
  Returns:
    str: Dữ liệu đã được xử lý và nối chuỗi với ký tự xuống dòng.
  """
  with open(filename, "r", encoding="utf-8") as f:
    data = f.read()
  # Tạo biến để lưu trữ kết quả
  processed_data = ""

  # Xử lý dữ liệu theo từng dòng
  for line in data.splitlines():
    # Loại bỏ ký tự đặc biệt
    line = re.sub(r"[^\w\s]", "", line)

    # Loại bỏ khoảng trắng thừa
    pattern = r"\s+"
    line = re.sub(pattern, " ", line)

    # Chuyển đổi sang chữ thường
    line = line.lower()

    # Thêm dữ liệu đã được xử lý vào biến kết quả
    processed_data += line + "\n"

  # Loại bỏ ký tự xuống dòng thừa ở cuối
  processed_data = processed_data[:-1]

  return processed_data

filename_train = "3.topic_detection_train.v1.0.txt"
filename_test = "3.topic_detection_test.v1.0.txt"

# Đọc và xử lý dữ liệu
processed_data_train = read_and_process(filename_train)
processed_data_test =read_and_process(filename_test)

# print(processed_data_train)
# print(processed_data_test)

# Tim kiếm tất cả các cặp nhãn và nội dung trong tập đã xử lý và đưa chúng vào tuple
pattern = r'(__label__\w+)\s((?:.(?!__label__))+)' 
matches_train = re.findall(pattern, processed_data_train, re.DOTALL)
matches_test = re.findall(pattern, processed_data_test, re.DOTALL)

# for match in matches_train:
#     print(match)

#Chuyển đổi danh sách các nhãn và nội dung thành DataFrame
train_df = pd.DataFrame(matches_train, columns=['label', 'content'])
test_df = pd.DataFrame(matches_test, columns=['label', 'content'])
pd.set_option('display.max_colwidth', None)

# In DataFrame để kiểm tra
test_df.head(6)


with open('vietnamese-stopwords.txt', 'r', encoding='utf-8') as f:
    vietnamese_stopwords = [line.strip() for line in f]

## Biểu diễn văn bản bằng TF-IDF
tfidf = TfidfVectorizer(stop_words=vietnamese_stopwords)
X_train_tfidf = tfidf.fit_transform(train_df['content'])
X_test_tfidf = tfidf.transform(test_df['content'])
X_test_tfidf


# Biểu diễn nhãn dưới dạng số
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(train_df['label'])
y_test = le.transform(test_df['label'])
y_test

# Huấn luyện mô hình Logistic Regression với TF-IDF
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)


# Đánh giá mô hình
y_pred = model.predict(X_test_tfidf)
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

for i in range(10):  # in ra 10 dự đoán đầu tiên
    index = i  # chỉ số của mẫu trong tập test
    predicted_label = le.inverse_transform([y_pred[index]])[0]
    true_label = test_df.loc[index, 'label']
    text = test_df.loc[index, 'content']
    print(f"content: {text[:100]}...")
    print(f"True Label: {true_label}, Predicted Label: {predicted_label}")
    print()