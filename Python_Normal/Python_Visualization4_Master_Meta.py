import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# https://heartbeat.fritz.ai/introduction-to-matplotlib-data-visualization-in-python-d9143287ae39

#Getting Started
import matplotlib.pyplot as plt

# If you’re using the Jupyter notebook we can easily display plots using: %matplotlib inline.
# However, if you’re using Matplotlib from within a Python script, you have to add plt.show() method inside the file to be able display your plot.


import numpy as np

#linspace(1+2i,10+10i,8)
#1+2i와 10+10i 사이에 균일한 간격의 점 8개의 복소수로 구성된 벡터를 만듭니다.
x = np.linspace(0,10,100)
print(x)

y = x**2
plt.plot(x,y)
plt.title('First Plot')
plt.xlabel('x Label')
plt.ylabel('y label')
plt.show()


#Multi Plot
#subplot(m,n,p)는 현재 Figure를 m×n 그리드로 나누고, p로 지정된 위치에 좌표축을 만듭니다.
plt.subplot(1,3,1)
plt.plot(x,y, 'red')

plt.subplot(1,3,2)
plt.plot(y,x, 'green')

plt.subplot(1,3,3)
plt.plot(y,x, 'green')

plt.show()

#2. Object oriented Interface: This is the best way to create plots.
# axes1과 같은 경우 0.1, 0.1, 0.8, 0.8이라고 작성이 되어있는데 ,
# 첫 0.1은 이미지의 x축의 위치, 두번째 0.1은 이미지의 y축의 시작위치, 세번째 0.8은 이미지의 가로길이,
# 네번째 0.8은 높이를 의미한다.0~1까지 값을 입력할 수 있으며 모두 상대적인 길이이다.
fig = plt.figure()
ax = fig.add_axes([0.1, 0.2, 0.8, 0.9])
ax.plot(x, y, 'purple')
plt.show()

#fig.add_axes함수
#이 함수는 각각의 위치를 지정하는 함수다.
#axes1과 같은 경우 0.1, 0.1, 0.8, 0.8이라고 작성이 되어있는데 , 첫 0.1은 이미지의 x축의 위치, 두번째 0.1은 이미지의 y축의 시작위치, 세번째 0.8은 이미지의 가로길이, 네번째 0.8은 높이를 의미한다.
# 0~1까지 값을 입력할 수 있으며 모두 상대적인 길이이다.
# fig = plt.figure()
# ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])
# ax1.plot(x, y, 'purple')
# ax2.plot(x, y, 'red')
# plt.show()

# fig, axes = plt.subplots(nrows=3, ncols=3)
# plt.show()

# tight_layout 여백 조정
# fig, axes = plt.subplots(nrows=3, ncols=3)
# plt.tight_layout()
# plt.show()

# fig, ax = plt.subplots(nrows=3, ncols=3)
# ax[0,1].plot(x,y)
# ax[1,2].plot(y,x)
# plt.tight_layout()
# plt.show()

#dpi 고해상도
# fig = plt.figure(figsize=(8,2), dpi=100)
# ax = fig.add_axes([0,0,1,1])
# ax.plot(x,y)
# plt.show()

# fig.savefig('my_sample.png')

# import matplotlib.image as mpimg
# plt.imshow(mpimg.imread('my_sample.png'))

# fig = plt.figure(figsize=(8,6), dpi=60)
# ax = fig.add_axes([0,0,1,1])
# ax.plot(x, x**2)
# ax.plot(x, x**3, 'red')
# plt.show()

# fig = plt.figure(figsize=(8,6), dpi=60)
# ax = fig.add_axes([0,0,1,1])
# ax.plot(x, x**2, label='X Square Plot')
# ax.plot(x, x**3, 'red', label='X Club Plot')
# ax.legend()
# plt.show()

# fig = plt.figure(figsize=(8,6), dpi=60)
# ax = fig.add_axes([0,0,1,1])
# ax.plot(x, y, color='purple', linewidth=3, linestyle='--', marker='o', markersize=3)
# plt.show()

# fig = plt.figure(figsize=(8,6), dpi=60)
# ax = fig.add_axes([0,0,1,1])

# # lw: linestyle,  ls: lines style
# ax.plot(x, y, color='purple', lw=3, ls='--')
# ax.set_xlim([0, 1])
# ax.set_ylim([0, 5])
# plt.show()

# x = np.random.randn(10000)
# plt.hist(x)
# plt.show()

import matplotlib.pyplot as plt
import datetime
import numpy as np

# x = np.array([datetime.datetime(2018,9,28, i,0) for i in range(24)])
# y = np.random.randint(100, size=x.shape)
# plt.plot(x,y)
# plt.show()

# fig, ax = plt.subplots()
# x = np.linspace(-1,1,50)
# y = np.random.randn(50)
# ax.scatter(x, y)
# plt.show()

# print(np.random.rand(10,4))
# my_df = pd.DataFrame(np.random.rand(10,4), columns=['a', 'b', 'c', 'd']);
# my_df.plot.bar();
# plt.show();

my_data = pd.read_csv('./nations.csv')
my_data['gdp_percap'].fillna(my_data['gdp_percap'].median(), inplace=True)
print(my_data.head())
print(my_data.groupby(['country']).mean())
#
avg_gdp_percap = my_data.groupby(['country']).mean()['gdp_percap']
top_five_countries = avg_gdp_percap.sort_values(ascending=False).head()
china = my_data[my_data['country'] == 'Macao SAR, China']
print(china.describe())

# plt.plot(china['year'], china['gdp_percap'])
# plt.xlabel('Year')
# plt.ylabel('GDP')
# plt.title('GDP Per capita of Macao SAR, China')
# plt.show()

# china.plot.bar(x='year', y='gdp_percap')
# plt.show()

# plt.subplot(311)
# plt.title('GDP Per Capita')
# plt.plot(china['year'], china['gdp_percap'])
# plt.subplot(312)
# plt.title('GDP in Billions')
# plt.plot(china['year'], (china['population']*china['gdp_percap']/10**9))
# plt.subplot(313)
# plt.title('Population in Millions')
# plt.plot(china['year'], china['population']/10**6)
# plt.tight_layout()
# plt.show()


# plt.subplot(3,1,1)
# plt.title('GDP Per Capita')
# plt.bar(china['year'], china['gdp_percap'], color='r')
#
# plt.subplot(312)
# plt.title('GDP in Billions')
# plt.bar(china['year'], (china['population']*china['gdp_percap']/10**9), color='g')
#
# plt.subplot(313)
# plt.title('Population in Millions')
# plt.bar(china['year'], china['population']/10**6, color='b')
# plt.tight_layout()
# plt.show()



# plt.plot(china['year'], (china['population']/china['population'].iloc[0]*100))
# china_gdp = china['population'] * china['gdp_percap']
# plt.plot(china['year'], china_gdp/china_gdp.iloc[0]/100)
# plt.plot(china['year'], china['gdp_percap']/china['gdp_percap'].iloc[0]/100)
# plt.title('Gdp and Population growth in China(first year = 100)')
# plt.legend(['Population', 'GDP', 'GDP Per Capita'], loc=4)
# plt.show()

# plt.bar(china['year'], (china['population']/china['population'].iloc[0]*100), color='r')
# china_gdp = china['population'] * china['gdp_percap']
# plt.bar(china['year'], china_gdp/china_gdp.iloc[0]/100)
# plt.bar(china['year'], china['gdp_percap']/china['gdp_percap'].iloc[0]/100, color='b')
# plt.title('Gdp and Population growth in China(first year = 100)')
# plt.legend(['Population', 'GDP', 'GDP Per Capita'], loc=4)
# plt.show()

# qt = my_data[my_data['country'] == 'Qatar']
#
# plt.bar(qt['year'], qt['gdp_percap'])
# plt.bar(china['year'], china['gdp_percap'])
# plt.legend(['Population', 'GDP'])
# plt.show()