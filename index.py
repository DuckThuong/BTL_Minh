import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import os

train_data = pd.read_csv('10F_train.csv')
test_data = pd.read_csv('10F_test.csv')
sample_submission = pd.read_csv('10F_sample_submission.csv')

train_data = train_data.fillna(train_data.mean())  
test_data = test_data.fillna(test_data.mean())

label_encoder = LabelEncoder()
train_data['Geography'] = label_encoder.fit_transform(train_data['Geography'])
train_data['Gender'] = label_encoder.fit_transform(train_data['Gender'])
test_data['Geography'] = label_encoder.transform(test_data['Geography'])
test_data['Gender'] = label_encoder.transform(test_data['Gender'])

X_train = train_data.drop(columns=['Exited', 'Surname', 'ID'])
y_train = train_data['Exited']
X_test = test_data.drop(columns=['Surname', 'ID'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train_scaled, y_train)

y_public_pred = knn_classifier.predict(X_test_scaled)

public_results = []
for i in range(len(y_public_pred)):
    public_results.append((test_data['ID'][i], y_public_pred[i]))

public_submission_df = pd.DataFrame(public_results, columns=['ID', 'Exited'])
public_submission_df.to_csv(r'C:\Users\ADMIN\Desktop\BTL_Minh\task.csv', index=False)

# for i in range(3):  # In ra 3 kết quả đầu tiên
#     print(f"ID: {test_data['ID'][i]}, Predicted Exited: {y_public_pred[i]}")

# if os.path.exists(r'C:\Users\ADMIN\Desktop\BTL_Minh\task.csv'):
#     print("File đã được ghi thành công.")
# else:
#     print("Có lỗi xảy ra khi ghi file.")

print(test_data['Gender kkkkk :'].head())

