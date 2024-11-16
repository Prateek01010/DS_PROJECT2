import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
dataset = pd.read_csv('dataset_phishing.csv')

# Prepare features (X) and target labels (y)
X = dataset.drop(['url', 'status'], axis=1)  # Drop 'url' and 'status' columns
y = dataset['status'].apply(lambda x: 1 if x == 'phishing' else 0)  # Convert status to binary labels

# Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Generate confusion matrix and classification report
confusion = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions, output_dict=True)

# Print classification report for reference
print(classification_report(y_test, predictions))

### Plot Confusion Matrix ###
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Legitimate', 'Phishing'], yticklabels=['Legitimate', 'Phishing'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

### Plot Classification Report Metrics ###
labels = ['Legitimate', 'Phishing']
precision = [report['0']['precision'], report['1']['precision']]
recall = [report['0']['recall'], report['1']['recall']]
f1_score = [report['0']['f1-score'], report['1']['f1-score']]

x = np.arange(len(labels))  # label locations
width = 0.25  # bar width

fig, ax = plt.subplots(figsize=(10, 6))
bar1 = ax.bar(x - width, precision, width, label='Precision')
bar2 = ax.bar(x, recall, width, label='Recall')
bar3 = ax.bar(x + width, f1_score, width, label='F1 Score')

# Add labels and title
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# Add value labels on top of bars
for bar in bar1 + bar2 + bar3:
    height = bar.get_height()
    ax.annotate(f'{height:.2f}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # vertical offset
                textcoords="offset points",
                ha='center', va='bottom')

plt.show()
