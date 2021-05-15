g = lambda x:x**2
print(g(8))

f = lambda x,y:x+y
print(f(2,2))

def inc(n):
    return lambda x: x + n

f = inc(2)
g = inc(4)
print(f(12))
print(g(12))
print(inc(2)(12))

a = [1,2,3,4]
b = [17,12,11,10]
print(list(map(lambda x, y:x+y, a,b)))

foo = [2, 18, 9, 22, 17, 24, 8, 12, 27]
print('filter')
print(list(filter(lambda x: x%3==0, foo)))

from functools import reduce
t = [47, 11, 42, 13]
result = reduce(lambda x, y : x + y, t)
print('reduce')
print(result)

a = [1, 6, 2, 5, 2, 7, 2, 8, 9, 11, 5, 26]
result = list(map(lambda x : x**2, a))  # 제곱시키기
print(result)
result2 = list(map(lambda x : str(x) if x % 2 == 0 else x, a))   # 짝수인 것은 string 타입으로 cast 아니면 단순히 반환
print(result2)

b = [12, 16, 24, 5, 20, 27, 12, 8, 9, 110, 51, 26]
result3 = list(map(lambda x, y : x + y, a, b))  # 리스트 자료형 두 개 받아서 연산
print(result3)


def square(x):
    return lambda : x*x

listOfLambdas = [square(i) for i in [1,2,3,4,5]]
for f in listOfLambdas:
    print(f())

listOfLambdas = [lambda i=i: i*i for i in range(1, 6)]
for f in listOfLambdas:
   print(f())


full_name = lambda first, last: f'Full name: {first.title()} {last.title()}'
print(full_name('guido', 'van rossum'))