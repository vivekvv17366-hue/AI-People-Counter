import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('people_count.csv')

print('\npeople counting data:')
print(df)

df['Time']=pd.to_datetime(df['Time'])


print('\nBasic statistics')
print(df.describe())

max_people=df['Current People'].max()

print(f'\nMaximum occupacy {max_people}')






plt.figure(figsize=(10,5))

plt.plot(df['Time'],df['Current People'],marker='o')

plt.xlabel('Time')
plt.ylabel('Number of People')
plt.title('People Occupacy Over Time')

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()