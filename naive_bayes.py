import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from tabulate import tabulate  # Tambahkan import untuk mencetak tabel dengan rapi

# 1. Baca dataset
try:
    data = pd.read_csv('student_lifestyle_dataset.csv')
    print("Dataset Loaded Successfully!")
except FileNotFoundError:
    print("Dataset file not found. Please check the file path.")
    exit()

# 2. Tampilkan informasi dataset
print("\nDataset Info:")
info = {
    "Column": data.columns,
    "Non-Null Count": data.count(),
    "Dtype": data.dtypes
}
info_df = pd.DataFrame(info)
print(tabulate(info_df, headers='keys', tablefmt='fancy_grid', showindex=False))

print("\nDataset Statistics:")
print(data.describe())  # Menghapus tabulate untuk statistik

print("\nChecking for missing values...")
if data.isnull().sum().any():
    print(tabulate(data.isnull().sum().reset_index(name="Missing Count"), headers=["Column", "Missing Count"], tablefmt='fancy_grid'))
    print("Handling missing values...")
    data.fillna(data.median(), inplace=True)
    print("Missing values handled.")
else:
    print("No missing values detected.")

# 3. Encoding kolom kategorikal
print("\nEncoding categorical columns...")
data['Stress_Level'] = data['Stress_Level'].astype('category').cat.codes
print("Encoding complete.")

# 4. Pisahkan fitur dan target
target_column = 'Stress_Level'
features = data.drop(columns=[target_column])
target = data[target_column]

# 5. Bagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
print("\nData split into training and testing sets.")

# 6. Inisialisasi dan latih model Naive Bayes
print("\nTraining Naive Bayes model...")
model = GaussianNB()
model.fit(X_train, y_train)
print("Model training complete.")

# 8. Prediksi menggunakan data uji
predictions = model.predict(X_test)

# 9. Evaluasi model
accuracy = accuracy_score(y_test, predictions)
classification_rep = classification_report(y_test, predictions, output_dict=True)

# Tampilkan hasil evaluasi dalam bentuk tabel
accuracy_table = pd.DataFrame(
    {"Metric": ["Accuracy"], "Value": [accuracy]}
)
classification_table = pd.DataFrame(classification_rep).transpose()

print("\nModel Evaluation (Accuracy):")
print(tabulate(accuracy_table, headers='keys', tablefmt='fancy_grid', showindex=False))

print("\nModel Evaluation (Classification Report):")
print(tabulate(classification_table, headers='keys', tablefmt='fancy_grid'))
