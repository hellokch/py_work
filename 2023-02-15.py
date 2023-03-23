# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""





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

b=[]
#a 리스트의 길이만큼 숫자를 입력받아
#b 에 저장하고, 입력받은 수의 합계 출력하기
for i in range(len(a)):
    b.append(int(input(str(i+1)+'번째 숫자 입력:')))
print(b,"요소의 합:", sum(b)) # : b 요소의 합
#%s : 문자열 표현의 형식 문자. list 값도 %s로 사용 가능
print("b= %s" % b)
    

a
a.append(1)
a
#a 리스트 정렬
a.sort()
a
#a 리스트의 마지막 요소 출력하기
len(a)
a[4]
a[len(a)-1]
a[-1]
a[-2]
a
a.pop() #제거
a

#index(값) 함수 : 요소의 인덱스 위치 리턴
a.reverse()
a
#a.find(400)
#insert(인덱스, 값) : 리스트 중간에 요소를 추가
a.index(11)
a.index(400)

a
a.insert(1,222)
a

a.remove(222)
a
b
a.extend(b)
a
a.count(11)


#문자열을 분리하여 리스트로 저장하기
dd="2022/11/25"
c = dd.split("/")
print(c)

#문제 : ss 문자열의 모든 숫자들의 합을 출력하기
ss = "10,20,50,60,30,40,50,60,30"

sslist = ss.split(",")
sslist
sum(sslist)
hap =0
for n in sslist :
    hap += int(n)
print(sslist,"의 요소의 합:",hap)


#map 함수 : 리스트의 요소에 적용함수를 설정
print(sslist,"의 요소의 합:", sum(list(map(int,sslist))))
#map(함수이름, 리스트) : 리스트의 요소들마다 함수 적용
#map(int,sslist) : sslist의 모든 요소들이 int 함수 적용하여 자료형이 정수형으로 변환
#list함수 : 리스트로 변환
mlist = list(map(int,sslist))
mlist
sum(mlist)

###dictionary : {key1:value1,key2:value2,} java map 같은 자료형
score_dic = {"lee":100,"hong":70,"kim":90}

score_dic['hong']
score_dic['hong'] = 75
print(score_dic)
#p의 점수 80점으로 추가하기
score_dic['park']=80
print(score_dic)

#park 정보 제거하기
del score_dic['park']
print(score_dic)

#키 값만 조회
print(score_dic.keys())
type(score_dic.keys())
print(list(score_dic.keys()))
#값들만 조회
print(score_dic.values())
print(list(score_dic.values()))

#키와 값들의 쌍인 값으로 조회
print(score_dic.items())
print(list(score_dic.items()))

#dictionary 객체들을 반복문으로 조회

for n in score_dic :
    #n : key값
    print(n,"=",score_dic[n])
print()
for n in score_dic.keys() :
    #n : key값
    print(n,"=",score_dic[n])


for n,s in score_dic.items() :
    #n키값 s벨류값
    print(n,"=",s)


#[(lee,100),...]
for v in score_dic.items():
    print(v[0],"=",v[1])
    print(v)
#v : 튜플객체 (k,v)쌍인 객체
for s in score_dic.values():
    print(s)


'''
문제 : 1. 궁합음식의 키를 입력받아 해당되는 음식을 출력하기
         등록안된 경우 오류 발생. => 등록여부 판단필요
       2.종료 입력시 등록된 내용 출력하기
         등록된음식 :
              떡볶이 : 어묵
              짜장면 : 단무지
       3. 등록이 안된경우 
          등록여부를 입력받아, 등록하는 경우 궁합음식을 입력받기  
          등록하시겠습니까(y)? 
             y입력 : foods객체에 추가.
                     궁합음식 입력받아서 foods에 추가함
             y가아닌경우 :
                     음식을 다시 입력하기    
              
foods = {"떡볶이":"어묵", "짜장면":"단무지"}
'''
foods = {"떡볶이":"어묵","짜장면":"단무지","라면":"김치","맥주":"치킨"}

while True:
    food = input("음식의 이름을 입력하세요 (종료는 'exit'): ")
    if food == "exit":
        print("등록된 음식: ")
        for key, value in foods.items():
            print(f"{key} : {value}")
        break
        
    if food in foods:
        print(f"{food}의 궁합음식은 {foods[food]} 입니다.")
    else:
        answer = input("등록되지 않은 음식입니다. 등록하시겠습니까? (y/n): ")
        if answer == 'y':
            new_food = input(f"{food}의 궁합음식을 입력하세요: ")
            foods[food] = new_food
            print(f"{food}이(가) 등록되었습니다.")
        else:
            print("다시 입력해주세요.")


print("등록된 음식:")
for f in foods.items():
    print(f[0],":",f[1])
for f in foods:
    print(f,":",foods[f])


#튜플 : 상수화된(변경불가)리스트.( )
tp1 = (10,20,30)
print(tp1)

for t in tp1:
    print(t)

#인덱스 사용 가능
print(tp1[0],tp1[1],tp1[2])
tp1.append(40) #변경안됨 오류날거임
tp1[1]=100 #변경 안됨

#튜플객체를 변경하기 위해서는 리스트로 변경하면 됨
list1 = list(tp1)
list1.append(40)
list1[0]=100
list1
#tuple(): 튜플객체로 변경
tp1=tuple(list1)
tp1

#tp1의 요소의 갯수와 변수의 갯수가 동일하면 사용가능
a,b,c,d = tp1
print(a,b,c,d)

#tp1의 요소의 갯수 구하기
print(len(tp1))
#list의 요소의 갯수 구하기
print(len(list1))
#tp1의 요소의 합 구하기
print(sum(tp1))
#list의 요소의 합 구하기
print(sum(list1))

#2,3번째 요소만 출력하기
print(tp1[1],tp1[2])
print(tp1[1:3])
print(tp1[:3])
print(tp1[::3])


tp1 #(100,20,30,40)
#tp1의 요소를 역순으로 출력하기
#tp1.reverse() #역순으로 객체를 수정, 듀플에서 불가
list1.reverse() #역순으로 객체를 수정
list1
#(40,30,20,100)
#1 tp1 요소를 역순으로 출력
for i in range(len(tp1)-1,-1,-1):
    print(tp1[i],end=",")

#2
print(tp1[::-1]) #역순 출력
tp1
print(list1[::-1])


### set : 중복불가. 집합을 표현하는 객체 {}
set1 = {30,10,20,10}
print(set1)
#10 요소는 한개만 출력됨. 순서 지정안됨
#print(set1[0]) # 인덱스 사용 불가

#집합 구현하기
set1={1,2,3,4,5,6}
set2={1,2,3,4,5,1,2,3,4,5}
print(set1)
print(set2)
set3= {5,6,7,8}

#교집합
print("set1 & set2",set1 & set2)
print("set1 & set3",set1 & set3)
print("set1 & set3",set1.intersection(set3)) #교집합 함수


#합집합
print("set1 | set2",set1 | set2)
print("set1 | set3",set1 | set3)
print("set1 합 set3",set1.union(set3))


## comprehension(컴프리헨션) 방식으로 Collection 객체 생성
#규칙성이 있는 데이터를 Collection 객체의 요소로 저장하는 방식
#numbers 리스트 : 1~10 까지의 데이터 저장
#1값을 대입
numbers=[1,2,3,4,5,6,7,8,9,10]
print(numbers)
#2 반복문 이용
numbers=[]
for n in range(1,11):
    numbers.append(n)
print(numbers)

#3. 컴프리헨션 이용
numbers=[x for x in range(1,11)]
print(numbers)



#두개의 리스트 데이터를 각각 한개씩 튜플로 생성하고, 튜플을 리스트로 생성하기
clist=['black','white']
slist=['S','M','L']

#1 반복문 이용
dlist = []
for c in clist :
    for s in slist :
        dlist.append((c,s))
print(dlist)

dist = [(c,s) for c in clist for s in slist]
print(dlist)
#[(black,S),(black,M),(black,L),(white,S),(white,M),(white,L)]


#1~10사이의 짝수 제곱값을 가진 set 객체 생성하기
my_set = {x**2 for x in range(1, 11) if x % 2 == 0}
print(my_set)



#dictionary 데이터 생성하기
products = {"냉장고":220, "건조기":140, "TV":130, "세탁기":150 , "컴퓨터":200}
#200미만의 제품만 prodict1 객체 저장하기
product1={}
for k in products : 
    if products[k] < 200:
        product1[k]=products[k]
print(product1)


product1={}
for k in products.keys():
    if products[k] <200:
        product1[k] = products[k]
print(product1)    


product1={}
for k,v in products.items():
    if v < 200:
        product1[k]=v
print(product1)

product1={k:v for k, v in products.items() if v < 200}
print(product1)
product1={k:products[k] for k in products if products[k] < 200}
print(product1)

'''
Colection : 데이터의 모임
    리스트(list)         : 배열. 순서유지. 첨자(인덱스) 사용가능.[]
    튜플(tuple)          : 상수화된 리스트 , 변경불가리스트.()
    딕셔너리(dictionary) : (key,value) 쌍 인객체{}
                items() : (key,value)쌍 인 객체를 리스트 형태로 리턴
                keys()  : (key)들만 리스트 형태로 리턴
                values(): value 들만 리스트형태로 리턴
    세트(set) : 중복불가 순서X 첨자(인덱스) 사용불가 집합표현객체 {}
            &, intersection() : 교집합
            |, union() : 합집합.
            
    컴프리헨션(comprehension) : 패턴(규칙) 이 있는 데이터를 생성하는 방법
'''
#######################
#함수와 람다
# 함수 : def 예약어사용
#######################

def func1():
    print("func2() 함수 호출됨")
    return 10 #함수 종료 후 값을 리턴
def func2(num):
    print("func2() 함수 호출됨",num)
     #리턴값이 없는 함수
a=func1()
print(a)
b=func2(100)
print(b)
func2('abc')

#전역변수 : 모든 함수에서 접근이 가능한 변수
#지역변수 : 함수 내부에서만 접근이 가능한 변수

def func3():
    c=300 #지역변수
    print("func3() 함수 호출됨",a,b,c)
   

def func4():
    a=110 #지역변수
    b=220 #지역변수
    print("func4() 함수 호출됨",a,b)
#함수 내부에서 전역 변수값 수정하기
def func5():
    global a,b #a,b변수는 전역변수를 사용함
    a=110
    b=220
    print("func5() 함수 호출됨",a,b)
a=100
b=200

func3()
#print("main:",a,b,c) #c 변수는 func3() 함수에서만 사용 가능
print("main",a,b)
func4()
print("main",a,b)
func5()
print("main",a,b)


#매개변수
def add1(v1,v2):
    return v1+v2
def sub1(v1,v2):
    return v1-v2

hap = add1(10,20)
sub = sub1(10,20)
print(hap)
print(sub)

hap= add1(10.5,20.1)
sub= sub1(10.5,20.1)
print(hap)
print(sub)

hap = add1("python","3.9") #문자끼리+
print(hap)
#hap = add1("python","3.9","7") #add1 함수의 매개변수 갯수가 틀


#가변 매개변수 : 매개변수의 갯수를 정하지 않은 경우
def multiparam(* p):
    result = 0
    for i in p:
        result += i
    return result

print(multiparam())
print(multiparam(10))
print(multiparam(10,20))
print(multiparam(10,20,30))
print(multiparam(1.5,2.5,3))
#print(multiparam("1.5","2.5","3")) 기본적으로 함수가 int이기 때문에 오류가 난다.

#매개변수에 기본값 설정
def hap1(num1=0, num2=1):
    return num1+num2

print(hap1())
print(hap1(10))
print(hap1(10,20))
print(hap1(0,20))
#print(hap1(10,20,30)) # 함수의 매개변수가 3개이기때문에 오류가 난다!
#TypeError: hap1() takes from 0 to 2 positional arguments but 3 were given



#리턴값이 두 개인 경우 : 리스트로 리턴
def multiReturn(v1,v2) :
    list1=[]
    list1.append(v1+v2)
    list1.append(v1-v2)
    return list1

list1=multiReturn(200,100)
print(list1)

#람다식을 이용한 함수 위의 함수를 람다식으로 전환
hap2=lambda num1,num2:num1+num2

print(hap2(10))
print(hap2(10,20))
print(hap2(10.5,20.5))
#기본값 매개변수
hap3 = lambda num1=0,num2=1:num1+num2
print(hap3(10))
print(hap3(10,20))
print(hap3(10.5,20.5))

#문제:리스트의 평균을 구해주는 함수 getMean 구현하기
def getMean(l):
    return sum(l)/len(l) if len(l)>0 else 0
#true 문 if 조건문 else false문

list1 = [1,2,3,4,5,6]
print(getMean(list1))
print(getMean([]))

getMean2= lambda l:sum(l)/len(l) if len(l)>0 else 0

list1 = [1,2,3,4,5,6]
print(getMean2(list1))
print(getMean2([]))


#mylist1 보다 각각의 요소가 10이 더 많은 요소를 가진 mylist2 생성
mylist1=[1,2,3,4,5]

#1 반복문
mylist2 = []
for i in mylist1:
    mylist2.append(i+10)
print(mylist2)
#2 컴프리헨션
mylist2 = []
mylist2 = [i+10 for i in mylist1]
print(mylist2)
#3 map방식
#map(함수, 리스트) : 리스트의 각 요소에 함수 적용
mylist2 = []
def add10(n):
    return n+10
mylist2 = list(map(add10, mylist1))
print(mylist2)

#4 map 람다 방식
mylist2 = []
mylist2 = list(map(lambda x: x+10, mylist1))
print(mylist2)

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


#mystr 문자열에 파이썬 문자의 위치를 strpos 리스트에 저장하기
mystr = "파이썬 공부중입니다. 파이썬을 열심히 공부합시다"
#1
strpos =[]
index = 0
while True :
    index = mystr.find("파이썬", index)
    if index <0:
        break
    strpos.append(index)
    index += 1
print(strpos)

strpos =[]
index = 0
while True :
    try :
        index = mystr.index("파이썬", index)
        if index <0:
            break
        strpos.append(index)
        index += 1
    except :
        break

print(strpos)





#Q1
'''
1. 화면에서 한개의 문자를 입력받아 대문자인 경우는 소문자로, 
   소문자인 경우는 대문자로 숫자인 경우는 20을 더한 값을 출력하기
[결과]
한개의 문자를 입력하세요 : 1
1 + 20 = 21

한개의 문자를 입력하세요 : a
a 문자의 대문자는 A

한개의 문자를 입력하세요 : A
A 문자의 소문자는 a
'''
input_char = input("한개의 문자를 입력하세요: ")

if input_char.isupper():
    print(f"{input_char} 문자의 대문자는 {input_char.lower()}")
elif input_char.islower():
    print(f"{input_char} 문자의 소문자는 {input_char.upper()}")
elif input_char.isdigit():
    result = int(input_char) + 20
    print(f"{input_char} + 20 = {result}")
else:
    print("입력된 문자가 올바르지 않습니다.")



#Q2
'''
2 (1)+(1+2)+(1+2+3)+... (1+2+3+...10)=220 출력하기

[결과]
(1)+(1+2)+(1+2+3)+(1+2+3+4)+(1+2+3+4+5)+(1+2+3+4+5+6)+(1+2+3+4+5+6+7)+(1+2+3+4+5+6+7+8)+(1+2+3+4+5+6+7+8+9)+(1+2+3+4+5+6+7+8+9+10)=220
'''

max_num = 10
outnum = 0
output = ''
i=1
for i in range(1,max_num+1) :
    for j in range(1,i+1):
        if(j==1) :
            output += '('
        outnum += j
        if(j!=i) : 
            output += str(j)+'+'
        if(j==i) :
            output += str(j)
            output += ')'
    if(i != max_num):
        output += "+"
output += '=' + str(outnum)
print(output)

#Q3
'''
3. 화면에서 자연수를 입력받아 각각의 자리수의 합을 구하기.

[결과]
자연수를 입력하세요 : 12345
자리수 합 : 15
'''
in_num = int(input("자연수를 입력해주세요 \n"))
s = int(len(str(in_num))) #자릿수 
outnum = 0
for i in range(s-1,-1,-1):
    
    outnum += in_num // (10**i)
    in_num = in_num % (10**i)
print('각 자리 숫자의 합은 '+str(outnum) + '입니다')

##############
num = input("자연수를 입력하세요 : ")
sum = 0

for i in num:
    print(i)
    sum += int(i)

print("자리수 합 :", sum)

#Q4
'''
4. aa,bb 리스트를 생성하고,
aa 리스트는 0부터 짝수 100개를 저장하고
bb 리스트는 aa 배열의 역순으로 값을 저장하기.
aa[0] ~ aa[9], bb[99]~bb[90] 값을 출력하기

[결과]
aa[ 0]= 0,aa[ 1]= 2,aa[ 2]= 4,aa[ 3]= 6,aa[ 4]= 8,aa[ 5]=10,aa[ 6]=12,aa[ 7]=14,aa[ 8]=16,aa[ 9]=18,
bb[99]= 0,bb[98]= 2,bb[97]= 4,bb[96]= 6,bb[95]= 8,bb[94]=10,bb[93]=12,bb[92]=14,bb[91]=16,bb[90]=18,
'''
aa = list(range(0, 200, 2))[:100]
bb = aa[::-1]

for i in range(10):
    print(f"aa[{i}]={aa[i]},", end="")
print()
for j in range(10):
    print(f"bb[{99-j}]={bb[99-j]},", end="")


#Q5

height = int(input("모래시계의 높이를 홀수로 입력하세요 : "))
for i in range(height // 2, -1, -1):
    print(" " * (height // 2 - i), end="")
    print("*" * (2 * i + 1))
for i in range(1, height // 2 + 1):
    print(" " * (height // 2 - i), end="")
    print("*" * (2 * i + 1))


















