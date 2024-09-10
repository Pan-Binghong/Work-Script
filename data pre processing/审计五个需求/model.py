import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.utils import resample
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path):
    return pd.read_excel(file_path)


def preprocess_data(data):
    # Dropping columns containing Chinese characters and not relevant for the model
    columns_to_drop = ['证券代码', '审计意见类型', '审计师', '公司全称', '所在行业', '财务报告审计意见', 'Unnamed: 0',
                       '年份']
    data_cleaned = data.drop(columns=columns_to_drop)
    # Convert date columns to datetime and extract year and month
    data_cleaned['统计截止日期年'] = pd.to_datetime(data_cleaned['统计截止日期']).dt.year
    data_cleaned['统计截止日期月'] = pd.to_datetime(data_cleaned['统计截止日期']).dt.month
    data_cleaned['审计日期年'] = pd.to_datetime(data_cleaned['审计日期']).dt.year
    data_cleaned['审计日期月'] = pd.to_datetime(data_cleaned['审计日期']).dt.month
    data_cleaned = data_cleaned.drop(columns=['统计截止日期', '审计日期'])
    return data_cleaned


def balance_data(data):
    # Balancing the classes
    class_counts = data['是否纳入'].value_counts()
    data_majority = data[data['是否纳入'] == 0]
    data_minority = data[data['是否纳入'] == 1]
    data_minority_upsampled = resample(data_minority, replace=True, n_samples=class_counts[0], random_state=123)
    data_balanced = pd.concat([data_majority, data_minority_upsampled])
    data_balanced = data_balanced.sample(frac=1, random_state=123).reset_index(drop=True)
    return data_balanced


def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, class_report


def get_user_input():
    print("\n请根据提示逐一输入特征值：")
    features_names = ["审计费用合计", "上期审计意见", "客户重要性", "事务所行业专长", "四大审计", "更换审计师",
                      "业务复杂度", "总资产收益率", "资产负债率", "经营亏损", "经营现金流比例", "销售收入增长率",
                      "统计截止日期年", "统计截止日期月", "审计日期年", "审计日期月"]
    features_list = []
    for feature in features_names:
        value = input(f"请输入 {feature} :")
        try:
            features_list.append(float(value))
        except ValueError:
            print("输入无效，请输入一个数字。")
            return get_user_input()  # 重新开始输入流程，如果输入错误
    return features_list
def predict_result(model, imputer, input_features):
    # Reshape the input to match the model's expected format
    input_features = [input_features]
    input_imputed = imputer.transform(input_features)  # Impute missing values if necessary
    prediction = model.predict(input_imputed)
    return "纳入" if prediction[0] == 1 else "未纳入"


# Example usage in the main function
def main(file_path):
    data = load_data(file_path)
    data_cleaned = preprocess_data(data)
    data_balanced = balance_data(data_cleaned)

    X = data_balanced.drop('是否纳入', axis=1)
    y = data_balanced['是否纳入']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Imputing missing values
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Model training with best parameter found earlier
    optimized_model = LogisticRegression(C=0.01, penalty='l2', max_iter=1000, random_state=42)
    optimized_model.fit(X_train_imputed, y_train)

    # Model evaluation
    accuracy, conf_matrix, class_report = evaluate_model(optimized_model, X_test_imputed, y_test)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", conf_matrix)
    print("Classification Report:\n", class_report)

    # User input prediction
    user_features = get_user_input()
    prediction = predict_result(optimized_model, imputer, user_features)
    print("预测结果:", prediction)



if __name__ == '__main__':
    main(r"merged_data_with_additional_info_filled_code.xlsx")