import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
"""NOTE:
To run this program successfully try to comment out plt.show() 
because of plot graph should not interrupting each other
"""

"""Load Excel File"""
XLSXFile = pd.ExcelFile('TakenMind-Python-Analytics-Problem-case-study-1-1.xlsx')
Existing_Employees = pd.read_excel(XLSXFile, 'Existing employees')
Left_Employees = pd.read_excel(XLSXFile, 'Employees who have left')

"""Save File as csv file"""
Existing_Employees.to_csv('Existing-Employees.csv')
Left_Employees.to_csv('Employees-Who-Have-Left.csv')

"""Data Cleaning and formatting"""
ExCsv = pd.read_csv('Existing-Employees.csv', nrows=1).columns
ExCsv = pd.read_csv('Existing-Employees.csv', usecols=ExCsv[1:])
LeftCsv = pd.read_csv('Employees-Who-Have-Left.csv', nrows=1).columns
LeftCsv = pd.read_csv('Employees-Who-Have-Left.csv', usecols=LeftCsv[1:])
ExCsv['left'] = 0
LeftCsv['left'] = 1
cat_data1 = ExCsv[['salary']].copy()
cat_data2 = LeftCsv[['salary']].copy()
ExCsv['salary_level'] = cat_data1
LeftCsv['salary_level'] = cat_data2

"""Salary level Data conversion for Existing & Employees who have left company X"""
salary_level = {'low': 1, 'medium': 2, 'high': 3}
ExCsv['salary_level'] = ExCsv["salary_level"].apply(lambda x: salary_level[x])
LeftCsv['salary_level'] = LeftCsv["salary_level"].apply(lambda x: salary_level[x])

"""Employees Data combined together as one Dataset"""
df1 = pd.DataFrame(ExCsv)
df2 = pd.DataFrame(LeftCsv)
frames = [df1, df2]
Combined_EmployeesDataset = pd.concat(frames, ignore_index=False)
Combined_EmployeesDataset.to_csv('Combined-Employees-Dataset.csv', encoding='utf-8')

Dataset = pd.read_csv('Combined-Employees-Dataset.csv', nrows=1).columns #Get rid of Unnamed Columns in csv files
Dataset = pd.read_csv('Combined-Employees-Dataset.csv', usecols=Dataset[1:])

print Dataset.head(29)

"""Correlations Analysis"""
sb.set(style="white")

"""Correlation Matrix Computation"""
corr = Dataset.corr()

"""Mask generate for upper triangular matrix"""
Maskgen = np.zeros_like(corr, dtype=np.bool)
Maskgen[np.triu_indices_from(Maskgen)] = True

"""Matplotlib figure size setup"""
f, axis = plt.subplots(figsize=(5, 4))

"""Custom diverging colormap generate"""
custom_map = sb.diverging_palette(10, 220, as_cmap=True)

"""Draw the heatmap with the mask and correct aspect ratio"""
sb.heatmap(corr, mask=Maskgen, cmap=custom_map, vmax=0.5, square=True, xticklabels=True, yticklabels=True, linewidths=0.5, cbar_kws={"shrink": 0.5}, ax=axis)
plt.show()


"""Features Analysis"""
sb.set(style="white")
f, axis = plt.subplots(figsize=(5, 4))
sb.barplot(x=Dataset.satisfaction_level, y=Dataset.left, orient="h", ax=axis)
plt.show()

"""Histogram Plot"""
sb.set(style="darkgrid")
g = sb.FacetGrid(Dataset, row="dept", col="left", margin_titles=True)
bins = np.linspace(0, 1, 13)
g.map(plt.hist, "satisfaction_level", color="steelblue", bins=bins, lw=0)
plt.show()


"""Analysis of the leaving employees"""
sb.set(style="white", palette="muted", color_codes=True)
f, axes = plt.subplots(3, 3, figsize=(9, 7))
sb.despine(left=True)

"""Employees that have left company X"""
leavers = Dataset.loc[Dataset['left'] == 1]

"""Histogram plot"""
sb.distplot(leavers['satisfaction_level'], kde=False, color="b", ax=axes[0, 0])
sb.distplot(leavers['salary_level'], kde=False, color="b", ax=axes[0, 1])
sb.distplot(leavers['average_montly_hours'], kde=False, color="b", ax=axes[0, 2])
sb.distplot(leavers['number_project'], kde=False, color="b", ax=axes[1, 0])
sb.distplot(leavers['last_evaluation'], kde=False, color="b", ax=axes[1, 1])
sb.distplot(leavers['time_spend_company'], kde=False, bins=5, color="b", ax=axes[1, 2])
sb.distplot(leavers['Work_accident'], bins=10, kde=False, color="b", ax=axes[2, 1])
sb.distplot(leavers['promotion_last_5years'], bins=10, kde=False, color="b", ax=axes[2, 0])
plt.show()


"""All key employees in company X"""
key_employees = Dataset.loc[Dataset['last_evaluation'] > 0.7].loc[Dataset['time_spend_company'] >= 3]
key_employees.to_csv('All-Key-Employees.csv')
key_employees.describe().to_csv('All-Key-Employees-Summary.csv')

"""Key employees that has been lost by company X"""
lost_key_employees = key_employees.loc[Dataset['left'] == 1]
lost_key_employees.to_csv('Lost-Key-Employees.csv')
lost_key_employees.describe().to_csv('Lost-Key-Employees-Summary.csv')

print("Number of key employees: ", len(key_employees))
print("Number of lost key employees: ", len(lost_key_employees))
print("Percentage of lost key employees: ", round((float(len(lost_key_employees))/float(len(key_employees))*100), 2), "%")


""" Find out why performing employees leave company X"""
"""Fetch out employees with a good last evaluation in company X"""
leaving_performers = leavers.loc[leavers['last_evaluation'] > 0.7]

sb.set(style="white")

"""Correlation matrix computation"""
corr = leaving_performers.corr()

"""Mask generate for the upper triangle matrix"""
genmask = np.zeros_like(corr, dtype=np.bool)
genmask[np.triu_indices_from(genmask)] = True

""" matplotlib figure size set"""
f, ax = plt.subplots(figsize=(5, 4))

"""Custom diverging colormap generate"""
colormap = sb.diverging_palette(10, 220, as_cmap=True)

"""Draw the heatmap with the mask and correct aspect ratio"""
sb.heatmap(corr, mask=genmask, cmap=colormap, vmax=0.5, square=True, xticklabels=True, yticklabels=True, linewidths=0.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()


"""Satisfied employees leaving company X"""
"""filter out people with a good last evaluation"""
satisfied_employees = Dataset.loc[Dataset['satisfaction_level'] > 0.7]

sb.set(style="white")

"""Correlation matrix computation"""
corr = satisfied_employees.corr()

"""Mask generate for the upper triangle matrix"""
gemask = np.zeros_like(corr, dtype=np.bool)
gemask[np.triu_indices_from(gemask)] = True

""" matplotlib figure size set"""
f, ax = plt.subplots(figsize=(5, 4))

"""Custom diverging colormap generate"""
colormap = sb.diverging_palette(10, 220, as_cmap=True)

"""Draw the heatmap with the mask and correct aspect ratio"""
sb.heatmap(corr, mask=gemask, cmap=colormap, vmax=.5, square=True, xticklabels=True, yticklabels=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()




