import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from data_generator import prepare_training_data

def train_svm_model(X_train, y_train, X_test, y_test):
    """
    Huấn luyện mô hình SVM với hyperparameter tuning
    """
    print("Đang huấn luyện mô hình SVM...")
    
    # Định nghĩa grid search parameters
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'class_weight': ['balanced', None]
    }
    
    # Tạo SVM model
    svm = SVC(random_state=42)
    
    # Grid search với cross-validation
    grid_search = GridSearchCV(
        svm, 
        param_grid, 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Huấn luyện model
    grid_search.fit(X_train, y_train)
    
    # Lấy model tốt nhất
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Đánh giá trên tập test
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return best_model, grid_search

def plot_confusion_matrix(y_test, y_pred, save_path='models/confusion_matrix.png'):
    """
    Vẽ confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['small', 'medium', 'large'],
                yticklabels=['small', 'medium', 'large'])
    plt.title('Confusion Matrix - SVM Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_learning_curve(model, X_train, y_train, save_path='models/learning_curve.png'):
    """
    Vẽ learning curve
    """
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, val_mean, label='Cross-validation score', color='red', marker='s')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='red')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve - SVM Model')
    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Hàm chính để huấn luyện SVM model
    """
    print("=== HUẤN LUYỆN MÔ HÌNH SVM ===")
    
    # Tạo thư mục models nếu chưa có
    import os
    os.makedirs('models', exist_ok=True)
    
    # Đọc dữ liệu
    try:
        request_data = pd.read_csv('data/request_data.csv')
        print(f"Đã đọc {len(request_data)} mẫu dữ liệu từ data/request_data.csv")
    except FileNotFoundError:
        print("Không tìm thấy file dữ liệu. Đang tạo dữ liệu mẫu...")
        from data_generator import generate_sample_data
        request_data, _ = generate_sample_data(2000)
        request_data.to_csv('data/request_data.csv', index=False)
    
    # Chuẩn bị dữ liệu
    X, y = prepare_training_data(request_data)
    
    # Chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Kích thước tập huấn luyện: {X_train.shape}")
    print(f"Kích thước tập kiểm tra: {X_test.shape}")
    
    # Huấn luyện model
    best_model, grid_search = train_svm_model(X_train, y_train, X_test, y_test)
    
    # Dự đoán trên tập test
    y_pred = best_model.predict(X_test)
    
    # Vẽ biểu đồ
    plot_confusion_matrix(y_test, y_pred)
    plot_learning_curve(best_model, X_train, y_train)
    
    # Lưu model
    joblib.dump(best_model, 'models/svm_model.joblib')
    joblib.dump(grid_search, 'models/svm_grid_search.joblib')
    
    print("\n=== KẾT QUẢ ===")
    print("✅ Mô hình SVM đã được huấn luyện thành công!")
    print("📁 Model đã được lưu vào: models/svm_model.joblib")
    print("📁 Grid search results đã được lưu vào: models/svm_grid_search.joblib")
    print("📊 Biểu đồ đã được lưu vào thư mục models/")
    
    # Hiển thị thông tin model
    print(f"\nThông tin model:")
    print(f"- Kernel: {best_model.kernel}")
    print(f"- C parameter: {best_model.C}")
    print(f"- Gamma: {best_model.gamma}")
    print(f"- Support vectors: {best_model.n_support_}")

if __name__ == "__main__":
    main() 