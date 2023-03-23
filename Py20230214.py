# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:08:01 2023

@author: 김충환
"""

# 한줄 주석

'''
여러줄 주석

실행 :
    F5 : 한줄 실행, F9 : 현재 라인 실행
'''

a=10
print(a)


print('hello')
print("hello")

print(10,20,30,40)

print('abc'*3)
print("abc"*3)


print("학번:"+100)
print("학번:",100)
print("학번:"+"100")
print("학번:"+str(100))

print("'안녕하세요'라고 말했습니다.")
print('"안녕하세요"라고 말했습니다.')
print("\"안녕하세요\"라고 말했습니다.")
print('\'안녕하세요\'라고 말했습니다.')
#\n : new line
print("동해물과 백두산이 마르고 닳도록\n하느님이 보우하사")
#\ 라인 연결, 다음라인도 연결된 문장.
# \ 사용 시 공백이 없어야 함.
print("동해물과 백두산이 마르고 닳도록 \
      하느님아 보우하사")

print("동해물과 백두산이 마르고 닳도록",
      "하느님이 보우하사")

print("""동해물과 백두산이 마르고 닳도록
      하느님이 보우하사
      무궁화 삼천리 화려강산""")

#10 20 30 출력
print(10,end="\t")
print(20,end="\t")
print(30,end="\n")

#10, 20, 30, 출력
print(10,end=",")
print(20,end=",")
print(30)


#문자열 : 문자들의 모임, 문자 여러 개, 문자의 배열로 인식
print("안녕하세요"[0])
print("안녕하세요"[2])
#문자열의 범위를 지정하여 출력하기
#문자열[첫번째 인덱스 : 마지막인덱스+1: 증감값]
print("안녕하세요"[0:2])
print("안녕하세요"[:2])
#처음부터 4번인덱스까지 2칸씩
print("안녕하세요"[:5:2])
print("안녕하세요"[2:]) #2번 인덱스 이후
print("안녕하세요"[::2]) 
print("안녕하세요"[::-1]) #뒤부터


#len() : 문자열의 길이
print(len("안녕하세여"))

#자료형 : 변수선언하지 않고 사용가능함.
#변수의 자료형은 값으로 결정됨

n = 10
type(n)
n=10.5
type(n)
n='안녕'
type(n)


#연산자
#산술 연산자 : +,-,*,/,%,//,**

5+7
5*7
5/7
5%7
5//7 # 정수형 몫의 값
5**2 # 제곱
5^2 #비트연산자 (XOR)


# 3741초는 몇시간 몇분 몇초인지 출력
sec = 3741
h = sec // 3600
m = (sec % 3600) // 60
s = (sec % 3600) % 60
print (h,"시간",m,"분",s,"초")


#대입 연산자 : =, +=,-=,*= ...
a=10
a += 10
a
a-= 5
a
a*=2
a


#문자열에서 사용가능한 대입연산자 : =,+=,*=
s="abc"
s += "d"
s
s *= 3
s


#자연수를 입력받아 +100을 한 값을 출력하기
num = int(input("자연수를 입력하세요"))
num += 100
print (num)

#형변환 함수
#int() : 정수형으로 변환
#float() : 실수형으로 변환
#str() : 문자열형으로 변환
print("num+100="+str(num))


#2,8,16 진수를 10진수로 변환

print(int("11",2))
print(int("11",8))
print(int("11",16))

#10진수를 2,8,6진수로 변환
print(10,"의 2진수:",bin(10))
print(10,"의 8진수:",oct(10))
print(10,"의 16진수:",hex(10))


# 형식문자를 이용하여 출력하기 :
# %d(10진수 정수)
# %f(실수)
# %s(문자열)

print("%d * %d = %d" %(2,3,6))
print("%f * %f = %f" %(2,3,6))
print("%.2f * %.2f = %.2f" %(2,3,6))

# %x,%X : 16진수 표현
print("%X" % (255))
print("%x" % (255))

print("안녕 %s!, 나도 %s " % ("홍길동","김삿갓"))

#format 함수를 이용한 출력
#{0:5d} : 첫번째값을 정수형 5자리로 출력
#{1:5d} : 두번째값을 정수형 5자리로 출력
#{2:5d} : 세번째값을 정수형 5자리로 출력

print("{0:5d}{1:5d}{2:5d}".format(100,200,300))
print("{1:5d}{2:5d}{0:5d}".format(100,200,300))

#직접 변수이름으로 출력
a=100
b=200
print(f"{a},{b}")


## 조건문 : if 문
# 들여쓰기 해야함
score = 65
if score >= 90 :
    print("A학점")
    print("합격입니다.")
else :
    if score >= 80 :
        print("B학점")
        print("합격입니다.")
    else :
        if score >= 70 :
            print("C학점")
            print("합격입니다.")
        else:
            if score >= 60 :
                print("D학점")
                
            else :
                print("F학점")


#if elif 구문
score = 65
if score >= 90 :
    print("A학점")
    print("합격입니다.")
elif  score >= 80 :
    print("B학점")
    print("합격입니다.")
elif score >= 70 :
    print("C학점")
    print("합격입니다.")
elif score >= 60 :
    print("D학점")               
else :
    print("F학점")

# 정수가 60이상이면 PASS 60미만이면 FAIL출력

inp=int(77)

if inp >=60 :
    print("PASS")
else:
    print("FAIL")


#간단한 조건식
#TRUE if 조건식 else FALSE
score = 65
print(score,"점수는", 'PASS' if score >=60 else 'FAIL')


#반복문
#1부터 100까지의 합 구하기
num = 100
hap = 0
# range(1,num+1, 증감값) : 1 ~ num 까지의 숫자들
for i in range(1,num+1) :
    hap += i
print("1부터 %d 까지의 합:%d" % (num,hap))





#1~45사이의 숫자 6개 출력하기
#난수 생성
import random
rnum = random.randrange(1,46)
print(rnum)
for i in range(1,7 ) :
    print(random.randrange(1,46), end=",")

'''
문자열 함수
'''

cnt = 0
#len(a) : a 문자열의 길이5
for i in range(len(a)):
    if a[i] == '1' :
        cnt += 1
        
print(a,"에서 1 문자의 갯수 :", cnt)
print(a,"에서 1 문자의 갯수 :",a.count('1'))



'''
collection  : 여러개의 데이터를 저장 할 수 있는 객체
list        : 배열의 형태, 인덱스 사용가능,[]로 표시함
tuple       : 상수화된 리스트. 변경불가 리스트 ()로 표시
set         : 중복불가, 집합 {}로 표시함
dictionary  : 자바의 Map, (key, value) 쌍인 객체들 {}로 표시함
'''


a = [0,0,0,0]
b = []
print(a, len(a)) #lan(a) : 리스트 요소의 갯수
print(b,len(b))

#a 리스트의 길이만큼 숫자를 입력받아, a에 저장하고, 입력받은 수의
# 전체 합계를 출력하기
hap=0
for i in range(len(a)):
    a[i] = int(input(str(i+1)+'번째 숫자 입력 : '))
    hap += a[i]
print(a,"요소의 합 : ", hap)
print(a,"요소의 합 : ", sum(a))







#Q1
height = int(input("삼각형의 높이를 입력하세요: "))

for i in range(1, height+1):
    print(" "*(height-i) + "*"*(2*i-1))


#Q2
year = int(input("년도를 입력하세요: "))

if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
    print(f"{year}년은 윤년입니다.")
else:
    print(f"{year}년은 평년입니다.")


#Q3
sum = 0
num = 1

while True:
    sum += num
    if sum > 1000:
        break
    num += 2

print(num)


#Q4
for c in range(-20, 51):
    f = (9 / 5) * c + 32
    print(f"섭씨온도: {c}, 화씨온도: {f}")


#Q5
money = int(input("금액을 입력하세요: "))

# 동전 종류와 금액
coins = [("500원동전", 500), ("100원동전", 100), ("50원동전", 50), ("10원동전", 10), ("1원동전", 1)]

for coin, value in coins:
    # 현재 동전으로 바꿔줄 수 있는 개수
    count = money // value
    # 남은 금액 계산
    money = money - count * value
    # 동전 개수 출력
    print(f"{coin} : {count}개")


#Q6
for i in range(2, 10):
    for j in range(2, 10):
        print("{0}X{1}={2:2d}".format(i, j, i*j), end=" ")
    print()

for i in range(2, 10):
    for j in range(2, 10):
        print("{0}X{1}={2:2d}".format(j, i, i*j), end="\t")
    print()






