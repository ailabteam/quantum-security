import numpy as np
import os
import time
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# --- PENNYLANE IMPORTS ---
import pennylane as qml
from pennylane import numpy as pnp # Sử dụng numpy của PennyLane để hỗ trợ auto-differentiation

print("Bắt đầu quá trình huấn luyện và đánh giá...")
print("Sử dụng backend: PennyLane")

# --- 1. TẢI DỮ LIỆU ĐÃ XỬ LÝ ---
data_dir = 'processed_data'
print(f"Tải dữ liệu từ thư mục '{data_dir}'...")

try:
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file dữ liệu trong '{data_dir}'.")
    print("Hãy chạy script process_data.py trước để tạo dữ liệu.")
    exit()

print("Tải dữ liệu thành công.")
print(f"Kích thước X_train: {X_train.shape}")
print(f"Kích thước X_test: {X_test.shape}")


# --- 2. HUẤN LUYỆN BASELINE: SVM CỔ ĐIỂN (Giữ nguyên) ---
print("\n--- Bắt đầu huấn luyện SVM Cổ điển (Baseline) ---")
classical_svm = SVC(kernel='rbf', random_state=42)
start_time = time.time()
classical_svm.fit(X_train, y_train)
end_time = time.time()
classical_training_time = end_time - start_time
print(f"Huấn luyện SVM cổ điển hoàn tất trong {classical_training_time:.4f} giây.")
y_pred_classical = classical_svm.predict(X_test)
accuracy_classical = accuracy_score(y_test, y_pred_classical)

print("\nKết quả của SVM Cổ điển:")
print(f"Độ chính xác (Accuracy): {accuracy_classical:.4f}")
print("Báo cáo phân loại (Classification Report):")
print(classification_report(y_test, y_pred_classical, target_names=['BENIGN (0)', 'DDoS (1)']))


# --- 3. HUẤN LUYỆN MÔ HÌNH HYBRID: QSVM VỚI PENNYLANE ---
print("\n--- Bắt đầu huấn luyện QSVM (với PennyLane) ---")

# 3.1. Thiết lập môi trường lượng tử
n_qubits = X_train.shape[1]  # Số qubit bằng số chiều dữ liệu
dev = qml.device("default.qubit", wires=n_qubits)

# 3.2. Định nghĩa Quantum Feature Map
# Đây là một mạch mã hóa góc (Angle Embedding) đơn giản, tương tự như các feature map khác.
def feature_map_circuit(x):
    qml.AngleEmbedding(x, wires=range(n_qubits))
    # Thêm một lớp entanglement để tạo các mối tương quan lượng tử
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])

# 3.3. Định nghĩa hàm tính Kernel
# Kernel giữa hai vector x1 và x2 là |<ψ(x2)|ψ(x1)>|^2
# Điều này được tính bằng cách áp dụng mạch cho x1, sau đó áp dụng mạch ngược (adjoint) cho x2,
# và đo xác suất của trạng thái |00...0>.
@qml.qnode(dev)
def kernel_circuit(x1, x2):
    feature_map_circuit(x1)
    qml.adjoint(feature_map_circuit)(x2)
    return qml.probs(wires=range(n_qubits))

def quantum_kernel(A, B):
    """Tính toán ma trận kernel giữa hai bộ dữ liệu A và B."""
    kernel_matrix = np.zeros((A.shape[0], B.shape[0]))
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            # Lấy xác suất của trạng thái |0...0>
            prob_0 = kernel_circuit(A[i], B[j])[0]
            kernel_matrix[i, j] = prob_0
    return kernel_matrix

# 3.4. Tính toán trước ma trận Kernel
# QUAN TRỌNG: Bước này là bước tốn nhiều thời gian nhất.
print("Bắt đầu tính toán ma trận kernel lượng tử cho tập train... (Có thể mất rất nhiều thời gian)")
start_time_kernel = time.time()
# Tính kernel matrix cho tập train
kernel_train = quantum_kernel(X_train, X_train)
end_time_kernel = time.time()
print(f"Tính xong kernel cho tập train trong {end_time_kernel - start_time_kernel:.2f} giây.")

# 3.5. Huấn luyện SVM với Kernel đã tính toán
# Sử dụng kernel='precomputed' để báo cho SVC rằng chúng ta cung cấp ma trận kernel đã tính sẵn.
pennylane_qsvm = SVC(kernel='precomputed', random_state=42)

start_time_train = time.time()
pennylane_qsvm.fit(kernel_train, y_train)
end_time_train = time.time()
quantum_training_time = (end_time_kernel - start_time_kernel) + (end_time_train - start_time_train)

print(f"Huấn luyện QSVM (PennyLane) hoàn tất.")

# 3.6. Đánh giá mô hình
print("Bắt đầu tính toán ma trận kernel cho tập test...")
start_time_pred = time.time()
# Để dự đoán, cần tính kernel giữa tập test và tập train
kernel_test = quantum_kernel(X_test, X_train)
end_time_pred = time.time()
print(f"Tính xong kernel cho tập test trong {end_time_pred - start_time_pred:.2f} giây.")

y_pred_quantum = pennylane_qsvm.predict(kernel_test)
accuracy_quantum = accuracy_score(y_test, y_pred_quantum)

print("\nKết quả của QSVM (PennyLane):")
print(f"Độ chính xác (Accuracy): {accuracy_quantum:.4f}")
print("Báo cáo phân loại (Classification Report):")
print(classification_report(y_test, y_pred_quantum, target_names=['BENIGN (0)', 'DDoS (1)']))


# --- 4. TỔNG KẾT VÀ SO SÁNH ---
print("\n--- TỔNG KẾT KẾT QUẢ ---")
print(f"{'Mô hình':<25} | {'Accuracy':<10} | {'Thời gian tổng (s)':<25}")
print("-" * 70)
print(f"{'SVM Cổ điển (RBF)':<25} | {accuracy_classical:<10.4f} | {classical_training_time:<25.4f}")
print(f"{'QSVM (PennyLane)':<25} | {accuracy_quantum:<10.4f} | {quantum_training_time:<25.4f}")
