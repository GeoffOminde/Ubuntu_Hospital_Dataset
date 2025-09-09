# 🌟 Ubuntu-Powered Python Data Analysis: “I Am Because We Are” in Code 🌍🏥

## 🎯 Lesson Goal
By the end of this lesson, you’ll be able to **load, explore, clean, and visualize hospital data using Python** — all while embracing the Ubuntu philosophy:  
*"I am because we are."*  

In data, no row is an island. Every patient, every record, every number — they matter because they are part of a community. Let’s honor that together.

---

## 🧑‍⚕️ What You’ll Need
- **Python** (Jupyter Notebook or Google Colab recommended)  
- **Libraries:** `pandas`, `matplotlib`, `seaborn`, `numpy`, `scikit-learn`  
- A fun, beginner-friendly attitude!  

💡 **Ubuntu Tip:** Just like in a village, we help each other learn. If you get stuck, ask a friend, search online, or revisit — learning is a shared journey.

---

## 📦 Step 1: Create Our Ubuntu Hospital Dataset 🏥
We’ll create a fictional hospital dataset inspired by Ubuntu values — where every patient’s story contributes to the health of the whole community.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

data = {
    'Patient_ID': range(1, 101),
    'Age': np.random.randint(1, 90, size=100),
    'Gender': np.random.choice(['Male', 'Female'], size=100),
    'Condition': np.random.choice(['Flu', 'Diabetes', 'Hypertension', 'Asthma', 'Healthy'], size=100, p=[0.3, 0.2, 0.2, 0.2, 0.1]),
    'Treatment_Duration_Days': np.random.randint(1, 15, size=100),
    'Satisfaction_Score': np.random.randint(1, 11, size=100),
    'Follow_Up_Needed': np.random.choice([True, False], size=100, p=[0.4, 0.6])
}

df = pd.DataFrame(data)
print("🏥 Welcome to Ubuntu General Hospital!")
print(df.head(10))
💬 Ubuntu Reflection: Each row is a person. Their age, condition, satisfaction — these aren’t just numbers. They represent lived experiences. Handle them with care.

🔍 Step 2: Explore the Data — Know Your Community
python
Copy code
print("=== Community Snapshot ===")
print(f"Total Patients: {len(df)}")
print(f"Average Age: {df['Age'].mean():.1f} years")
print(f"Most Common Condition: {df['Condition'].mode()[0]}")
print(f"Average Satisfaction: {df['Satisfaction_Score'].mean():.2f}/10")

# Gender distribution
print(df['Gender'].value_counts())

# Visualize Conditions
sns.countplot(data=df, x='Condition', palette='viridis')
plt.title("🩺 Community Health Conditions — We Rise By Lifting All")
plt.xlabel("Condition")
plt.ylabel("Number of Patients")
plt.show()
💬 Ubuntu Reflection: Visualization gives voice to the community’s needs. Who needs more care?

🧹 Step 3: Clean the Data — Healing the Gaps Together
python
Copy code
# Introduce missing satisfaction scores
df.loc[np.random.choice(df.index, size=10), 'Satisfaction_Score'] = np.nan
avg_satisfaction = df['Satisfaction_Score'].mean()
df['Satisfaction_Score'].fillna(avg_satisfaction, inplace=True)
💬 Ubuntu Reflection: Instead of deleting incomplete records, we uplift them using the wisdom of the whole.

📈 Step 4: Analyze & Visualize — Wisdom Through Sharing
python
Copy code
# Treatment Duration by Condition
sns.boxplot(data=df, x='Condition', y='Treatment_Duration_Days', palette='Set2')
plt.title("⏳ Treatment Duration by Condition — Understanding to Serve Better")
plt.show()

# Satisfaction vs Treatment Duration
sns.scatterplot(data=df, x='Treatment_Duration_Days', y='Satisfaction_Score', hue='Condition', palette='deep', s=100)
plt.title("😊 Satisfaction vs Treatment Duration")
plt.show()
💬 Ubuntu Reflection: Data reveals patterns — but only when we ask compassionate questions.

🤖 Step 5: Simple Prediction — Serving the Future Together
python
Copy code
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df_encoded = pd.get_dummies(df, columns=['Condition', 'Gender'], drop_first=True)
X = df_encoded.drop(['Follow_Up_Needed', 'Patient_ID'], axis=1)
y = df_encoded['Follow_Up_Needed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Follow-Up', 'Follow-Up'], yticklabels=['No Follow-Up', 'Follow-Up'])
plt.title("🔍 Confusion Matrix — Where Can We Improve Our Care?")
plt.show()
💬 Ubuntu Reflection: Even machines must learn with humility. Misclassifications teach us to listen better.

🌈 Step 6: Celebrate & Reflect — Ubuntu Closing Circle
python
Copy code
print("🎉 CONGRATULATIONS! You’ve completed the Ubuntu Data Journey.")
print("You honored stories, healed gaps, and predicted needs with compassion.")
🌿 Ubuntu Principles in Your Code:
Treated missing data with community averages

Visualized conditions to understand collective burdens

Predicted needs proactively

Reflected on meaning, not just metrics

💡 Next Steps:
Try real Kaggle datasets (Heart Disease UCI, Diabetes Health Indicators)

Ask: “Who is not represented in this data?”

Share your notebook — learning grows when shared

📬 Share your Ubuntu Data Notebook with #UbuntuDataScience — let’s build a global village of compassionate analysts!
Happy coding, data healer. 🌿🐍📊