import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import seaborn as sns
from seaborn import axes_style

salary_data = pd.read_csv('Salary_Data.csv')
pd.set_option('display.max_columns', None)
print(salary_data.head())
salary_data = pd.DataFrame(salary_data)


#Count of genders
plt.figure(figsize=(12, 10))
gender_counts = salary_data['Gender'].value_counts().reset_index()
gender_counts.columns = ['Gender', 'Count']

ax = sns.barplot(x='Gender', y='Count', data=gender_counts)
ax.set_xlabel("")
plt.title('Count of Gender')
plt.savefig("CountOfGender", dpi= 300, bbox_inches='tight')
plt.show()

#Average age of genders
mean_ages = salary_data.groupby('Gender')['Age'].mean()
(mean_ages)


#Unified and stored education levels
def UnifyEducationLevel(s):
    if pd.isna(s):
        return s
    s = s.lower()
    if 'bachelor' in s:
        return 'Bachelor'
    elif 'master' in s:
        return 'Master'
    elif 'phd' in s:
        return 'PhD'
    elif 'high school':
        return 'High School'
    return s

salary_data['Education Level'] = salary_data['Education Level'].apply(UnifyEducationLevel)
print(salary_data['Education Level'].value_counts())
education_counts = salary_data['Education Level'].value_counts()


#Graph Education count per degree
plt.figure(figsize=(12, 10))
ax = sns.barplot(x=education_counts.index, y=education_counts.values)
ax.set_xlabel("")
plt.title('Count by Education Level')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.savefig("EducationCount", dpi= 300, bbox_inches='tight')
plt.show()


#Histogram of ages
plt.figure(figsize=(12, 10))
ax = sns.histplot(x='Age', bins = 10, data = salary_data)
ax.set_xlabel("")
plt.title("Age")
plt.savefig("AgeSpan", dpi= 300, bbox_inches='tight')
plt.show()

#Top 20 frequent jobs
top20Jobs = salary_data['Job Title'].value_counts().head(20)

top20_df = top20Jobs.reset_index()
top20_df.columns = ['Job Title', 'Count']

plt.figure(figsize=(12, 10))

ax = sns.barplot(
    x='Job Title',
    y='Count',
    data=top20_df.sort_values('Count', ascending=False),
)

plt.xticks(rotation=45, ha='right')

plt.title('Top 20 Job Titles by Frequency', fontsize=16)
plt.xlabel('Job Title', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig("20MostFrequentJobs", dpi= 300, bbox_inches='tight')
plt.show()


#Frequency of salary
plt.figure(figsize=(12, 10))
ax = sns.histplot(x="Salary",bins = 10, data = salary_data)
ax.set_xlabel("")
plt.title("Salary")
plt.savefig("SalaryFrequency", dpi= 300, bbox_inches='tight')
plt.show()


#Average Years of Experience
plt.figure(figsize=(12, 10))
ax = sns.histplot(x="Years of Experience", bins=10, data=salary_data)
ax.set_xlabel("")
plt.title('YOE')
plt.savefig("YearsOfExperience", dpi= 300, bbox_inches='tight')
plt.show()


#Gender and Salary
custom_palette = {
    "Male": "#8ecae6",
    "Female": "#a8dadc",
    "Other": "#e9c46a",
}

plt.figure(figsize=(12, 10))
ax = sns.boxplot(x = "Gender",
            y = "Salary",
            data = salary_data,
            palette = custom_palette,
            legend = False)

plt.title("Gender & Salary")
ax.set_xlabel("")
plt.savefig("Gender&Salary", dpi= 300, bbox_inches='tight')
plt.show()

custom_palette = {
    "High School": "#8ecae6",
    "Bachelor": "#a8dadc",
    "Master": "#e9c46a",
    "PhD": "#c77dff"
}


#Education and Salary
plt.figure(figsize=(12, 10))
ax = sns.boxplot(x = "Education Level",
                 y = "Salary",
                 hue = "Education Level",
                 data = salary_data,
                 palette = custom_palette,
                 order = ["High School", "Bachelor","Master", "PhD"],
                 legend= False)


plt.savefig("Education&Salary", dpi= 300, bbox_inches='tight')
ax.set_xlabel("")
plt.show()


#Boxplot for salary and years of experience
salary_data['AgeGroup'] = pd.cut(salary_data['Age'], bins=37)

# Create a categorical variable for years of experience
# You need to create this category column first
salary_data['YOE_Group'] = pd.cut(salary_data['Years of Experience'], bins=37)

# Create age categories
salary_data['AgeCategory'] = pd.cut(
    salary_data['Age'],
    bins=[0, 30, 40, 50, float('inf')],
    labels=['Young', 'Early-career', 'Mid-career', 'Senior']
)

# Define custom palette for the hue
custom_palette = {
    "Young": "#8ecae6",
    "Early-career": "#a8dadc",
    "Mid-career": "#e9c46a",
    "Senior": "#c77dff"
}

plt.figure(figsize=(12, 10))
# Use AgeCategory as the hue, but YOE_Group on the x-axis
ax = sns.boxplot(x="YOE_Group",
                y="Salary",
                hue="AgeCategory",
                data=salary_data,
                palette=custom_palette)

# Format x-axis labels
labels = [f'{int(interval.left)}-{int(interval.right)}' for interval in salary_data['YOE_Group'].cat.categories]
ax.set_xticklabels(labels, rotation=45)

plt.title("Years of Experience & Salary by Career Stage")
plt.legend(title="Career Stage")
plt.tight_layout()
#plt.savefig("YOE_Salary_by_CareerStage", dpi=300, bbox_inches='tight')
plt.show()


#Years, Salary & Age
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
xfactor = "Age"

for i, yoe in enumerate([1, 3, 5, 7, 9, 11]):
    g = sns.regplot(x=xfactor, y="Salary", ax=axes[i//2, i%2], data=salary_data[salary_data['Years of Experience']==yoe])
    g.set(title=f'YoE = {yoe}')

plt.tight_layout()
plt.savefig("YearsAgeSalary", dpi= 300, bbox_inches='tight')
plt.show()


#Gender & Education participants
custom_palette = {
    "High School": "#8ecae6",
    "Bachelor": "#a8dadc",
    "Master": "#e9c46a",
    "PhD": "#c77dff"
}

plt.figure(figsize = (12, 10))
ax = sns.countplot(x = "Gender",
                   hue = "Education Level",
                   data = salary_data,
                   palette =  custom_palette)
ax.set_title("Participants of Gender & Education")
plt.savefig("Gender&EducationParticipants", dpi= 300, bbox_inches='tight')
plt.tight_layout
plt.show()


#Top jobs for each gender
male_pop_jobs = salary_data[salary_data["Gender"] == "Male"]["Job Title"].value_counts()[:10].index
female_pop_jobs = salary_data[salary_data["Gender"] == "Female"]["Job Title"].value_counts()[:10].index
# Combine to get unique job titles from both top 10 lists
t = set(list(male_pop_jobs) + list(female_pop_jobs))

# Calculate male count for each job for sorting
job_count_diff = []
for e in t:
    job_data = salary_data[salary_data["Job Title"] == e]
    dc = job_data[job_data["Gender"] == "Male"].shape[0]
    job_count_diff.append([e, dc])

# Sort by male count
job_count_diff.sort(key=lambda x: x[1])

# Define custom palette with soft colors
custom_palette = {
    "Male": "#8ecae6",  # Soft blue
    "Female": "#a8dadc",  # Pale teal
    "Other": "#e9c46a",  # Soft gold/amber
}

# Create figure and plot
plt.figure(figsize=(14, 10))

# Filter to include only the selected job titles
both_pop_jobs_data = salary_data[salary_data['Job Title'].isin(t)]

# Create the countplot with custom palette
g = sns.countplot(y='Job Title',
                  hue='Gender',
                  data=both_pop_jobs_data,
                  order=[e[0] for e in job_count_diff],
                  palette=custom_palette)

# Add title and labels
g.set(title='Number of Participants by Gender and Job Title')
plt.xlabel('Count')
plt.ylabel('Job Title')
plt.legend(title='Gender')

# Adjust layout and save
plt.tight_layout()
plt.savefig("Gender_Job_Distribution", dpi=300, bbox_inches='tight')
plt.show()


