# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 09:24:33 2023

@author: 김충환
"""

import sqlite3
conn = sqlite3.connect("mydb")
cur = conn.cursor()

cur.execute("select * from member")
memlist = cur.fetchall();
for m in memlist :
    print(m[0],m[1],m[2])
conn.close()

conn = sqlite3.connect("mydb")
cur = conn.cursor()
cur.execute("select * from member")
while True :
    row = cur.fetchone()
    if row == None :
        break
    print(row)
conn.close()




data = [
    ('test7','테스트7','test7@aaa.bbb'),
    ('test8','테스트8','test8@aaa.bbb'),
    ('test9','테스트9','test9@aaa.bbb'),
    ('test10','테스트10','test10@aaa.bbb')
]

conn = sqlite3.connect("mydb")
cur = conn.cursor()
cur.executemany("INSERT INTO member(id, name, email) VALUES (?, ?, ?)", data)
conn.commit()
conn.close()



#db내용 수정하기
conn = sqlite3.connect("mydb")
cur = conn.cursor()

param = []
param.append("hongkd@aaa.bbb")
param.append("test7")
cur.execute("update member set email = ? where id = ?", param)
conn.commit()
conn.close()


#이름이 테스트10 회원정보 삭제하기
conn = sqlite3.connect("mydb")
cur = conn.cursor()
param = []
param.append("테스트10")
cur.execute("delete from member where name=?", param)
conn.commit()
conn.close()


#오라클 데이터베이스에 접속하기
#오라클 모듈을 설정 해야함
#pip install cx_Oracle
#pip install : 외부 모듈을 설정 명령어. console에서 실행


import cx_Oracle

conn = cx_Oracle.connect('kic','1111','localhost/xe')
cur = conn.cursor()
cur.execute("select * from board")
st_list = cur.fetchall()
for st in st_list :
    print(st)
conn.close()


conn = cx_Oracle.connect('kic','1111','localhost/xe')
cur = conn.cursor()
cur.execute("select * from student")
st_list = cur.fetchall()
for st in st_list :
    print(st)
conn.close()

'''

'''

conn = cx_Oracle.connect('kic','1111','localhost/xe')
cur = conn.cursor()

sql = "insert into student (STUDNO,NAME,GRADE,ID,JUMIN) values (:STUDNO,:NAME,:GRADE,:ID,:JUMIN)"
cur.execute(sql, studno = 5555, name = '파이썬', grade=5, id='test1', jumin='9001011234567')

#dictionary 객체
param = {"studno" : 5575, "name" :'파이썬3', "grade":5, "id":'test3', "jumin":'9001011234567'}
cur.execute(sql,param)

conn.commit()
conn.close()


conn = cx_Oracle.connect('kic','1111','localhost/xe')
cur = conn.cursor()

cur.execute("select * from student where grade =%d" %(5))
st_list = cur.fetchall()
for st in st_list :
    print(st)
conn.close()


'''
pandat : 표 형태의 데이터 저장 모듈
Series : 1차원 형태의 데이터
DataFrame : 2차원 형태 (행,열) 의 데이터.
            Series 데이터의 모임
'''
import pandas as pd

#딕셔너리 데이터를 Series 데이터로
dict_data = {'a':1,'b':2,'c':3}
sr = pd.Series(dict_data)
print(sr)
print(sr.index)
print(sr.values)
#튜플 데이터를 시리즈 데이터로
tuple_data = ("홍길동", '1991-01-25', '남', True)
sr = pd.Series(tuple_data,index=["이름", "생년월일", "성별", "학생여부"])
print(sr)
print(sr.index)
print(sr.values)

#한개의 값만 조회
sr[0] #순서로 조회
sr["이름"] #인덱스로 조회
sr.이름 #인덱스로 조회
sr[1] #순서로 조회.
sr["생년월일"] #인덱스로 조회
sr.생년월일#인덱스
#여러개의 값 조회
sr[[0,1]] #선택조회
sr[['이름','생년월일']] #인덱스 조회
#sr['이름', '생년월일'] 이거는 오류남
#여러개의 값 조회. 범위를 지정하여 조회
sr[0:2] #순서로 조회. 마지막값 앞까지
sr['이름':'성별'] #인덱스 조회. 마지막값까지



dict_data = {'c0':[1,2,3],'c1' : [4,5,6], 'c2':[7,8,9],'c3':[10,11,12],'c4':[13,14,15]}
df = pd.DataFrame(dict_data)
print(df)
print("컬럼명 : ", df.columns)
print("인덱스명 : ",df.index)
#한개 조회
df["c0"]
#여러개 조회
df[["c0","c1"]]
type(df[["c0","c1"]])


df = pd.DataFrame([[15,'남','서울중'],[17,'여','서울여고'],
                   [17,'남','서울고']],
                  index=['홍길동','성춘향','이몽룡'],
                  columns=['나이','성별','학교'])

print(df)
print(df.index)
print(df.columns)

#인덱스명 변경하기
df.index=["학생1","학생2","학생3"]
print(df)

df.columns=["age", "gender", "school"]
print(df)




#rename : 컬럼명, 인덱스명의 일부만 변경하기
#inplace = True : 객체 자체 변경 #default는 false임
df.rename(columns = {"age":"나이"})
print(df)

df = df.rename(index={"학생":"홍길동"})
print(df)

exam_data={'수학':[90,80,70],
           '영어':[98,88,95],
           '음악':[85,95,100],
           '체육':[100,90,90]}


#exam_data를 이용하여 인덱스는 홍길동, 이몽룡, 김삿갓 인 DATAFRAME객체 생성하기
df = pd.DataFrame(exam_data, index=['홍길동', '이몽룡', '김삿갓'])
print(df)


#mean() : 평균
print(df.mean())
print(type(df.mean()))

print("수학평균",df.mean()["수학"])
print("수학평균",df["수학"].mean())
print(df["수학"])

#수학총점

print("수학평균",df.sum()["수학"])
print("수학평균",df["수학"].sum())
print(df["수학"].sum())


#median() : 중간값
'''
    중간값 : 데이터를 정렬하여 가운데 값
    수학 90,80,70
    영어 98,95,88
        데이터의 갯수 홀수 : 가운데 값.
        데이터의 갯수 짝수 : 가운데 두 값의 평균
'''

print(df.median())

print(df.median()["수학"])
print(df["수학"].median(),)


#홍길동 데이터 조회하기
df.수학
df["수학"]

#인덱스명으로 조회하기 => 행을 값 조회. .loc 사용
#loc[인덱스명] : 인덱스에 해당하는 행을 조회
#iloc[순서] : 순서 해당하는 행을 조회
df.loc["홍길동"] #홍길동 행 (index) 조회
df.iloc[0] #첫번째 행 순서 조회
type(df.loc["홍길동"])
df.loc["홍길동"].mean()
df.loc["홍길동"].median()

df.std() #표준편차
df.var() #분산



df.describe()
type(df.describe())

df.describe()["수학"]
df["수학"].describe()

df.info()
df["수학"].info()

df.head()

df.tail

#김사깟의 총점,평균,중간값, 표준편차를 조회하기
df.loc["김삿갓"]
print("총점:", df.loc["김삿갓"].sum())
print("평균:", df.loc["김삿갓"].mean())
print("중앙값:", df.loc["김삿갓"].median())
print("표준편차:", df.loc["김삿갓"].std())
df.loc["김삿갓"].describe()





#데이터프레임 복사하기
df2 = df #얕은복사, df.df2는 동일한 객체임
df2.info()
df

#df 데이터의 홍길동 인덱스를 홍길순으로 변경하기
df.rename(index ={"홍길동" : "홍길순"},inplace=True)
df
df2


#깊은 복사 : 두개 객체가 다른 객체
df3 = df[:] #깊은 복사. [:] 범위지정. 전체영역
#df3 데이터의 홍길순 인덱스를 홍길홍으로 변경하기
df.rename(index={"홍길순":"홍길동"}, inplace=True)
df
df3


df4 = df.copy()
df4

#drop() : 행,열 제거하기
#axis = 0 : 행을 의미함
#axis = 1 : 열을 의미함
#행 제거
df3.drop(["홍길동"], axis=0, inplace = True)
df3
#열 제거
df3.drop(["체육"],axis =1, inplace = True)
df3
#열 제거
del df3["음악"]
df3

del df4["음악"], df4["체육"]
df4
df4 = df.copy()
df4
df4.drop(["음악","체육"],axis=1,inplace=True)
df4




#df 데이터의 수학, 영어 컬럼 조회하기
df[["수학","영어"]]
df["수학":"영어"] ##얘는 오류남
#df 데이터의 수학 컬럼 조회하기
df["수학"] #series 객체
df[["수학"]] #dataframe 객체
type(df["수학"]) #series 객체
type(df[["수학"]]) #dataframe객체



#df 데이터의 이몽룡(row, index) 학생 점수 조회하기
df.loc["이몽룡"] #인덱스 이름
df
df.iloc[1] #순서 조회
#df 데이터의 이몽룡, 김삿갓 학생 점수 조회하기
df.loc[["이몽룡", "김삿갓"]] #인덱스 이름
df.iloc[[1,2]] #순서 조회
df.iloc[1:2] #순서 조회




#범위로 조회하기
df.loc["이몽룡":"김삿갓"] #이몽룡(index) 부터 김삿갓까지
df.loc["이몽룡":] #이몽룡부터 끝까지
df.loc[:"이몽룡"] #처음부터 이몽룡 까지
df.loc[:] #처음부터 끝까지
df.loc[::2] #처음부터 끝까지 2칸씩
df.loc[::-1] #처음부터 끝까지 역순으로 조회

df.iloc[1:3] #1번부터 2번까지
df.iloc[1] #1번부터 끝까지
df.iloc[:2] #처음부터 1번까지
df.iloc[:] #처음부터 끝까지
df.iloc[::2] #처음부터 끝까지 2칸씩
df.iloc[::-1] #처음부터 끝까지 역순으로




#이몽룡의 수학, 영어 점수 조회하기
df.iloc["이몽룡"][["수학","영어"]] #씨리즈 객체
df.iloc["이몽룡",["수학","영어"]]
df.iloc[["이몽룡"],["수학","영어"]] #DataFrame 객체
df.iloc[["이몽룡"]][["수학","영어"]] #DataFrame 객체

df[:] #컬럼만 조회.
df.loc[::-1]  #df.loc[행의 범위, 열의 범위]



#jeju1.csv 파일을 판다스 모듈을 이용하여 읽기
import pandas as pd
#read_csv : jeju1.csv 파일을 읽어 DataFrame 객체로 생성
df=pd.read_csv("data/jeju1.csv")
type(df)
df.info()
df.head()
df.tail()
df

df.장소
df["장소"]
df[["장소"]]
type(df[["장소"]])

#LON, LAT만 조회하기
df[["LON","LAT"]]


df.index
#set_index : 장소 컬럼을 인덱스로 변경하기
df.set_index("장소", inplace = True)
df
df.info()

df.loc["돔베돈"]
df.loc[["돔베돈"]]
df.index
#인덱스 값을 여행지 컬럼으로 추가하기
#컬럼추가 : dataframe[새 컬럼명] = 값
#컬럼수정 : dataframe[기존 컬럼명] = 값
df["여행지"] = df.index
df
df.info()

#현재의 index값을 컬럼으로 변경하기
df.reset_index(inplace=True)
df.info()
df
df.index

#장소 컬럼 제거하기
df.drop("장소", axis = 1, inplace=True)
df
#df데이터의 내용을 csv파일로 저장하기
#to_csv("파일이름", index=False)
#index=False : 인덱스는 파일에 저장 안함 기본값 True

df.to_csv("data/df_jeju.csv", index=False)
df.to_csv("data/df_jeju.csv", index=True)



'''
read_excel("파일이름", "시트이름", "인덱스 컬럼")
sheet_name = None : sheet 이름 지정 안함
                    모든 sheet 읽음
'''

#한개의 sheet 읽기
df=pd.read_excel("data/sales_2015.xlsx", "january_2015",index_col=None)
type(df)

'''
{"sheet이름 :" dataframe 데이터, ....}

'''
dfall = pd.DataFrame()
dfall
for name,data in df.items():
    print("sheet 이름:",name)
    print("data의 자료형 :", type(data))
    df.concat(data, sort=False)

type(dfall)
data.info()
dfall








#Q1 rect 클래스 구현

'''
아래 내용이  실행 되도록  Rect 클래스 구현하기
    가로,세로를 멤버변수로.
    넓이(area),둘레(length)를 구하는 멤버 함수를 가진다
    클래스의 객체를 print 시 :  (가로,세로),넓이:xxx,둘레:xxx가 출력
[결과]
(10,20), 넓이:200,둘레:60
(10,10), 넓이:100,둘레:40
200 면적이 더 큰 사각형 입니다.
—--------------------------------------------------------
     rect1 = Rect(10,20)
     rect2 = Rect(10,10)
     print(rect1)
     print(rect2)
     if rect1 > rect2 :
         print(rect1.area(),"면적이 더 큰 사각형 입니다.")
     elif  rect1 < rect2 :  
         print(rect2.area(),"더 큰 사각형 입니다.")
     elif rect1 == rect2 :
         print(rect1.area(),"=",rect2.area(),"같은 크기의 사각형 입니다.")
         
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

class Rect:
    def __init__(self,wid,hei):
        self.wid = wid
        self.hei = hei
        
    def area(self):
        area = self.wid * self.hei / 2        
        return int(area)
    def __str__(self):
        return f'(가로 :{self.wid}, 높이 :{self.hei}),넓이{self.area()}'
    
    def __lt__(self,other):
        print("< 연산자 호출:")
        return self.area() < other.area()
    def __gt__(self,other):
        print("> 연산자 호출:")
        return self.area() > other.area()
    def __eq__(self,other):
        print("== 연산자 호출:")
        return self.area() == other.area()

rect = Rect(10,20)
print (rect.area())

rect1 = Rect(10,20)
rect2 = Rect(10,10)
print(rect1)
print(rect2)
if rect1 > rect2 :
    print(rect1.area(),"면적이 더 큰 사각형 입니다.")
elif  rect1 < rect2 :  
    print(rect2.area(),"더 큰 사각형 입니다.")
elif rect1 == rect2 :
    print(rect1.area(),"=",rect2.area(),"같은 크기의 사각형 입니다.")


#Q2
'''
아래 내용이 실행 되도록, Calculator 클래스를 상속받은 
   UpgradeCalculator  클래스 구현하기
   
   Calculator  클래스
     value 멤버변수
     add 멤버함수 => 현재 value의 값에 매개변수로 받는 값을 더하기
   UpgradeCalculator 클래스
     minus 멤버함수 => 현재 value의 값에 매개변수로 받는 값을 빼기

—---------------------------------------------------------------
cal = UpgradeCalculator()
cal.add(10)
cal.minus(7)

print(cal.value) # 10에서 7을 뺀 3을 출력

'''
class Calculator:
    def __init__(self, value=0):
        self.value = value
    def add (self, add):
        self.value += add
        
class UpgradeCalculator(Calculator):
    def minus(self, sub):
        self.value -= sub

cal = UpgradeCalculator()
cal.add(10)
cal.minus(7)

print(cal.value) # 10에서 7을 뺀 3을 출력


'''
2번에서 구현한 Calculator 클래스를 이용하여 
   MaxLimitCalculator 클래스 구현하기
MaxLimitCalculator 클래스에서 value 값은 절대 100 이상의 값을 가질수 없다.


—-----------------------------
cal = MaxLimitCalculator()
cal.add(50) # 50 더하기
cal.add(60) # 60 더하기
print(cal.value) # 100 출력


'''

class MaxLimitCalculator(Calculator):
    def add (self, add):
        self.value += add
        if self.value > 100:
            self.value = 100

cal = MaxLimitCalculator()
cal.add(50) # 50 더하기
cal.add(60) # 60 더하기
print(cal.value) # 100 출력

















