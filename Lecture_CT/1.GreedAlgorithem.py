n = 1260
count = 0
# 큰 단위의 화폐부터 차례대로 확인하기
array = [500, 100, 50, 10]

for coin in array:

    # 해당 화폐로 거슬러 줄 수 있는 동전의 개수 세기
    # 나누기 연산 후 소수점 이하의 수를 버리고, 정수 부분의 수만 구함
    count += n // coin

    # 화페로 나눈 나머지 n에 대입
    n %= coin

    print(n,coin, n // coin, count, n)

print(count)