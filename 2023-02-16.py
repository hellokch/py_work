# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 09:17:06 2023

@author: 김충환
"""
##예외 처리 : 예측가능한 오류 발생시 정상처리
#try except 문장

idx = "파이썬".index("일")
idx = "파이썬".find("일")
idx

#예외처리하기
try :
    idx="파이썬".index("이") 
    print(idx)
except :
    print("파이썬글자에는 '일'이라는 글자가 없습니다.")



##다중 예외처리
num1 = input("숫자형 데이터1 입력:")
num2 = input("숫자형 데이터2 입력:")
try:
    n1=int(num1)
    n2=int(num2)
    print(n1+n2)
    print(n1/n2)
except ValueError as e:
    print ("숫자로 변환 불가")
    print (e)
except ZeroDivisionError as e:
    print ("두번째 숫자는 0안됨")
    print (e)
finally :
    print ("프로그램 종료")


#나이를 입력받아 19세 미만이면 미성년, 19세 이상이면 성인 출력하기
#입력된 데이터가 숫자가 아니면 숫자만 입력하세요 메세지 출력하기

try: #오류 발생 시 except 블럭으로 이동
    age = int(input("나이를 입력하세요: "))
    if age < 19:
        print("미성년")
    else:
        print("성인")
except ValueError:
    print("숫자만 입력하세요.")

#else 블럭 : 오류발생이 안된 경우 실행되는 블럭
try :
    age = int(input("나이를 입력하세요: "))
except :
    print("숫자만 입력하세요.")
else : #정상적인 경우 실행 블럭
    if age < 19:
        print("미성년")
    elif age >= 19:
        print("성인")

#raise : 예외 강제 발생
try :
    print(1)
    raise ValueError
    print(2)
except ValueError:
    print("ValueError 강제 발생")

#pass 예약어 : 블럭 내부에 실행될 문장이 없는 경우
n=9
if n>10 :
    pass
else :
    print("n의 값은 10 이하입니다.")


try:
    age = int(input("나이를 입력하세요 :"))
    if age < 19:
        print("미성년")
    else :
        print("성년")
except ValueError:
    pass

def dumy():
    pass


##
# 클래스 : 사용자 정의 자료형
#       구조체 + 함수 => 변수+함수의 모임.
# 상속 : 다중상속 가능. 여러개의 부모클래스가 존재.
# self : 자기참조변수. 인스턴스 함수의 매개변수로 설정해야함 (java에서 this)
# 생성자 : def __init__(self)
###

#기본생성자 제공 클래스 : 생성자를 구현하지 않음
class Car:
    color=""
    speed=0
    def upSpeed(self,value):
        self.speed += value
    def downSpeed(self,value):
        self.speed -= value

car1 = Car()
car1.color = "빨강"
car1.speed = 10

car2 = Car() #객체화
car2.color = "파랑"
car2.speed = 20
car2.upSpeed(30)
print("자동차1의 색상 :%s, 현재 속도 : %dkm" % (car1.color,car1.speed))
print("자동차2의 색상 :%s, 현재 속도 : %dkm" % (car2.color,car2.speed))



class Car:
    color=""
    speed=0
    def __init__(self,v1,v2=0):
        self.color = v1
        self.speed = v2
        
    def upSpeed(self,value):
        self.speed += value
    def downSpeed(self,value):
        self.speed -= value

car1 = Car("빨강", 10) #객체화. 생성자 호출
car2 = Car("파랑", 20) 
car3 = Car("노랑")
car2.upSpeed(30)
print("자동차1의 색상 :%s, 현재 속도 : %dkm" % (car1.color,car1.speed))
print("자동차2의 색상 :%s, 현재 속도 : %dkm" % (car2.color,car2.speed))
print("자동차3의 색상 :%s, 현재 속도 : %dkm" % (car3.color,car3.speed))



#멤버변수 : 클래스 내부에 설정
#인스턴스변수 : 객체별로 할당된 변수. self.변수명
#클래스변수 : 객체에 공통 변수      클래스명.변수명

class Car:
    color="" #색상
    speed=0 #속도
    num = 0 #자동차 번호
    count=0 # 자동차 객체 갯수
    def __init__(self,v1="",v2=0): #생성자
        self.color = v1 #인스턴스변수
        self.speed = v2 #인스턴스변수
        Car.count += 1#클래스변수
        self.num = Car.count #인스턴스변수        
    def printMessage(self) :
        print ("색상:%s, 속도:%dkm, 번호%d, 생산번호:%d" % (self.color,self.speed,self.num,Car.count))


car0=Car()
car1=Car("빨강",10)
car1.printMessage()
car2=Car("파랑")
car1.printMessage()
car2.printMessage()


# 문제 : Card 클래스 구현하기
#   멤버변수 : kind(카드종류), number(카드숫자),
#             no(카드번호), count(현재까지 생성된 카드 갯수)
#   멤버함수 : printCard()  kind:heart, number:1,no:1,count:1

class Card:
    count = 0
    kind = ''
    number = 0
    def __init__(self, kind= 'spade', number = 1):
        self.kind = kind
        self.number = number
        Card.count += 1
        self.no = Card.count
        
    def printCard(self):
        print("kind:{}, number:{}, no:{}, count:{}".format(self.kind, self.number, self.no, Card.count))


card1 = Card()
card1.printCard()  
card2 = Card("heart")
card2.printCard()
card3 = Card("spade", 10)
card3.printCard()


#상속 : 기존의 클래스를 이용하여 새로운 클래스 생성 다중상속이 가능
#class 클래스명 (부모클래스1, 부모클래스2, ....)

class Car:
    speed = 0
    door = 3
    def upSpeed(self,v):
        self.speed += v
        print("현재 속도(부모클래스):%d" % self.speed)


#class Sedan extends Car{}
class Sedan(Car): #Car클래스를 상속.
    pass

class Truck(Car):
    def upSpeed(self,v):
        self.speed += v
        if self.speed > 150:
            self.speed = 150
        print("현재 속도(자손클래스):%d" % self.speed)

car1 = Car()
car1.upSpeed(200)
sedan1 = Sedan()
sedan1.upSpeed(200)
truck1 = Truck()
truck1.upSpeed(200)

'''
  함수 : def 예약어로 함수 정의
         return 값 : 함수를 종료하고 값을 전달
         매개변수 : 함수를 호출할때 필요한 인자값 정의
              가변매개변수 : 매개변수의 갯수를 지정안함. 0개이상. * p 표현
              기본값설정 : (n1=0,n2=0) : 0,1,2개의 매개변수 가능 
  예외처리 : try, except, finally, else, raise
  클래스 : 멤버변수,멤버함수, 생성자.
           인스턴스변수 : self.변수명. 객체별로 할당되는 변수
           클래스변수   : 클래스명.변수명. 해당 클래스의 모든객체들의 공통변수
    self : 자기참조변수. 인스턴스함수에 첫번째 매개변수로 설정.
   생성자 : __init__(self,...) : 객체생성에 관여하는 함수
           클래스내부에 생성자가 없으면 기본생성자를 제공.        
   상속 : class 클래스명 (부모클래스명1,부모클래스명2,..) :
          다중상속가능           
          오버라이딩 : 부모클래스의 함수를 자손클래스가 재정의
'''

#클래스에서 사용되는 특별한 함수
# 매직매소드 (magic method)라고 불림
#__xxxx__형태인 함수.
class Line :
    length = 0
    def __init__(self,length):
        self.length=length
    def __repr__(self):
        return "선길이:" + str(self.length)
    def __add__(self,other):
        print("+연산자 사용:" ,end="")
        return self.length + other.length
    def __lt__(self,other):
        print("< 연산자 호출:")
        return self.length < other.length
    def __gt__(self,other):
        print("> 연산자 호출:")
        return self.length > other.length
    def __eq__(self,other):
        print("== 연산자 호출:")
        return self.length == other.length
    
line1 = Line(200)
line2 = Line(100)
print("line1 = ", line1)
print("line2 = ", line2)
print("두 선의 합 = ", line1 + line2)
print("두 선의 합 = ", line1.__add__(line2))

if line1 < line2 :
    print("line2 선이 더 깁니다.")
elif line1 == line2:
    print("line1과 line2 선의 길이는 같습니다.")
elif line1 > line2 :
    print("line1 선이 더 깁니다.")
    
'''
클래스에서 사용되는 연산자에 사용되는 특수 함수
+   __add__(self, other)
–	__sub__(self, other)
*	__mul__(self, other)
/	__truediv__(self, other)
//	__floordiv__(self, other)
%	__mod__(self, other)
**	__pow__(self, other)
&	__and__(self, other)
|	__or__(self, other)
^	__xor__(self, other)
<	__lt__(self, other)
>	__gt__(self, other)
<=	__le__(self, other)
>=	__ge__(self, other)
==	__eq__(self, other)
!=	__ne__(self, other)
   
생성자 : __init__(self,...) : 클래스 객체 생성시 요구되는 매개변수에 맞도록 매개변수 구현
출력   : __repr__(self) : 클래스의 객체를 출력할때 문자열로 리턴.   
'''   

#추상 함수 : 자손클래스에서 강제로 오버라이딩 해야하는 함수
#           함수의 구현부에 raise NotImplementedError()를 기술해야함.


class Parent:
    def method(self) : 
        raise NotImplementedError
class Child(Parent):
    #pass
    def method(self):
        print("자손클래스에서 오버라이딩 함")

ch= Child()
ch.method()




#모듈 : import 모듈명
#mod1.py, mod2.py 함수 호출하기
import mod1
import mod2
print("mod1 모듈 add()=", mod1.add(40,30))
print("mod1 모듈 sub()=", mod1.sub(40,30))
print("mod2 모듈 add()=", mod2.add(40,30))
print("mod2 모듈 sub()=", mod2.sub(40,30))

import mod1 as m1
import mod2 as m2
print("mod1 모듈 add()=", m1.add(40,30))
print("mod1 모듈 sub()=", m1.sub(40,30))
print("mod2 모듈 add()=", m2.add(40,30))
print("mod2 모듈 sub()=", m2.sub(40,30))


from subdir import mod1 as m1
from subdir import mod2 as m2
print("mod1 모듈 add()=", m1.add(40,30))
print("mod1 모듈 sub()=", m1.sub(40,30))
print("mod2 모듈 add()=", m2.add(40,30))
print("mod2 모듈 sub()=", m2.sub(40,30))





### 문자열의 형태 변경 변경하기
data = '''
   park 800915-1234567
   kim 890125-2345678
   choi 850125-a123456
'''
#1. 정규식 없이 주민번호 뒷자리 감추기
print(data)
result = [] #

for line in data.split("\n"):
    word_result = []
    for word in line.split(" "):
        if len(word) == 14 and word[:6].isdigit() and \
            word[7:].isdigit():
                word = word[:6]+"-"+"*******"
        word_result.append(word)
    result.append(" ".join(word_result))

print("\n".join(result))
    



#Q1
nation_dict = {}

# 나라와 수도 등록
while True:
    nation = input("등록할 나라와 수도를 입력하세요(예: 한국 서울): ")
    if nation == "종료":
        break
    else:
        nation_list = nation.split()
        nation_dict[nation_list[0]] = nation_list[1]

# 등록된 나라와 수도 출력
print("등록된 나라와 수도:")
for nation, capital in nation_dict.items():
    print(nation, capital)

# 나라 입력받아 해당 나라의 수도 출력 또는 등록
while True:
    nation_input = input("나라 이름을 입력하세요(종료 입력 시 종료): ")
    if nation_input == "종료":
        break
    elif nation_input in nation_dict:
        print(nation_input, nation_dict[nation_input])
    else:
        capital_input = input("등록되지 않은 나라입니다. 수도를 입력하세요: ")
        nation_dict[nation_input] = capital_input





def fibo(n):
    result = [0, 1]
    for i in range(n - 2):
        result.append(result[-1] + result[-2])
    return result

n = int(input("피보나치 수열의 요소 갯수를 입력하세요(3이상의 값): "))
print(f"fibo({n})={fibo(n)}")



num = int(input("자연수를 입력하세요"))
if((lambda x : True if x % 2 == 1 else False)(num)) :
    print(num, "숫자는 홀수입니다.")
else:
    print(num,"숫자는 짝수 입니다.")
















