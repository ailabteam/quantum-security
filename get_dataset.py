import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("Bắt đầu quá trình chuẩn bị dữ liệu...")

# --- BƯỚC 1: CHUẨN BỊ DỮ LIỆU ---

# 1.1. Tải dữ liệu
# Chúng ta sẽ dùng file chứa tấn công DDoS từ bộ CIC-IDS2017.
# Link này trỏ thẳng đến file CSV đã được xử lý một phần.
url = "https://raw.githubusercontent.com/nprintz/CIC-IDS-2017/master/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
print(f"Đang tải dữ liệu từ {url}...")
try:
    df = pd.read_csv(url)
except Exception as e:
    print(f"Lỗi khi tải dữ liệu: {e}")
    exit()

print("Tải dữ liệu thành công!")

# 1.2. Làm sạch dữ liệu cơ bản
# Tên cột có thể có khoảng trắng ở đầu/cuối, cần loại bỏ.
df.columns = df.columns.str.strip()
print("Tên các cột sau khi làm sạch:", df.columns.tolist())

# Dữ liệu có thể chứa giá trị vô hạn (Infinity) hoặc NaN (Not a Number).
# Thay thế các giá trị vô hạn bằng NaN.
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Loại bỏ các dòng chứa bất kỳ giá trị NaN nào.
df.dropna(inplace=True)

# 1.3. Tạo bộ dữ liệu con (Subset) cho bài toán nhị phân
print(f"Số lượng dòng dữ liệu ban đầu: {len(df)}")
print("Phân bố các nhãn ban đầu:\n", df['Label'].value_counts())

# Lọc ra chỉ các dòng có nhãn 'BENIGN' và 'DDoS'
df_subset = df[df['Label'].isin(['BENIGN', 'DDoS'])].copy()

# 1.4. Lấy mẫu cân bằng (Balanced Sampling)
# Đây là bước cực kỳ quan trọng để QML chạy được và kết quả không bị lệch.
n_samples_per_class = 1000 # Bạn có thể thay đổi số này, nhưng hãy bắt đầu nhỏ.
random_state = 42 # Để đảm bảo kết quả có thể lặp lại

df_benign = df_subset[df_subset['Label'] == 'BENIGN'].sample(n=n_samples_per_class, random_state=random_state)
df_ddos = df_subset[df_subset['Label'] == 'DDoS'].sample(n=n_samples_per_class, random_state=random_state)

# Gộp lại thành một dataframe cân bằng
df_balanced = pd.concat([df_benign, df_ddos])

print(f"\nĐã tạo bộ dữ liệu con cân bằng với {len(df_balanced)} mẫu.")
print("Phân bố nhãn trong bộ dữ liệu mới:\n", df_balanced['Label'].value_counts())

# 1.5. Tách thành đặc trưng (X) và nhãn (y)
X = df_balanced.drop('Label', axis=1)
# Chuyển nhãn text thành số: BENIGN -> 0, DDoS -> 1
y = df_balanced['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

# 1.6. Phân chia dữ liệu thành tập Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_state, stratify=y
)
print(f"\nKích thước tập huấn luyện (train): {X_train.shape}")
print(f"Kích thước tập kiểm thử (test): {X_test.shape}")


# --- BƯỚC 2: TIỀN XỬ LÝ VÀ GIẢM CHIỀU ---

# 2.1. Chuẩn hóa dữ liệu (Scaling)
# Các thuật toán như PCA và SVM rất nhạy cảm với thang đo của dữ liệu.
scaler = StandardScaler()

# Quan trọng: Chỉ 'fit' scaler trên tập train để tránh rò rỉ thông tin từ tập test
X_train_scaled = scaler.fit_transform(X_train)
# Áp dụng scaler đã 'fit' đó cho tập test
X_test_scaled = scaler.transform(X_test)
print("\nĐã chuẩn hóa dữ liệu.")

# 2.2. Giảm chiều dữ liệu với PCA
# Giảm từ ~80 đặc trưng xuống còn 4 để phù hợp với số qubit của máy tính lượng tử.
n_components = 4
pca = PCA(n_components=n_components, random_state=random_state)

# Tương tự, chỉ 'fit' PCA trên tập train
X_train_pca = pca.fit_transform(X_train_scaled)
# Áp dụng PCA đã 'fit' cho tập test
X_test_pca = pca.transform(X_test_scaled)

print(f"Đã giảm chiều dữ liệu xuống còn {n_components} thành phần chính.")
print(f"Kích thước X_train sau PCA: {X_train_pca.shape}")
print(f"Kích thước X_test sau PCA: {X_test_pca.shape}")

# In ra lượng phương sai được giải thích bởi các thành phần chính
explained_variance = pca.explained_variance_ratio_.sum()
print(f"Tổng phương sai được giải thích bởi {n_components} thành phần: {explained_variance:.2%}")

print("\n--- HOÀN TẤT CHUẨN BỊ DỮ LIỆU ---")
print("Các biến đã sẵn sàng cho mô hình Machine Learning:")
print("X_train_pca, y_train, X_test_pca, y_test")
