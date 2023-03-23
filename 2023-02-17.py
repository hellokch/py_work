# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:46:57 2023

@author: 김충환
"""

#2. 정규식을 이용하여 주민번호 뒷자리 감추기
import re #정규식을 위한 모듈 regular expression

'''
re.compile(정규식 패턴): 패턴 객체 생성
pat : 패턴객체. 정규식 형태로 정의된 형태를 지정하는 객체
d{6,7} d는 숫자
(\\d{6,7})[-]d{7} : 형태 지정.패턴.
    =>앞의 6또는 7 자리 숫자, - , 7자리 숫자 인 형태 패턴
() : 그룹 ----> g<1>, g<2> .....패턴 안에()을 묶음
\d{6,7} : \d(숫자 데이터) {6,7}(6,7개의 자릿수) 숫자 6또는 7개
[-] : -문자
\d{7} : 숫자 7개
'''

pat = re.compile("(\\d{6,7})[-]\\d{7}")

data = '''
   park 800915-1234567
   kim 890125-2345678
   choi 850125-a123456
'''

print(pat.sub("\g<1>-******", data))


#정규식을 이용하여 데이터 찾기
str1 = "The quick brown fox jomps over the laze dog Te Thhhhhhe,THE"

str_list = str1.split()
#공백을 기준으로 분리
print(str_list)


#대소문자 구분 함.(디폴트)
pattern = re.compile("Th*e") 
# h* 는 h가 0개 이상
#패턴 객체 설정

count = 0
for word in str_list:
    if pattern.search(word):
        count += 1
print("결과 1=>%s:%d" %("갯수",count))


#re.I : 대소문자 구분없이 검색
pattern = re.compile("Th*e",re.I) 

count = 0
for word in str_list:
    if pattern.search(word):
        count += 1
print("결과 2=>%s:%d" %("갯수",count))



#결과 3
pattern = re.compile("Th*e",re.I) 

print("결과3=>", end="")

for word in str_list:
    if pattern.search(word):
        print(word,end=",")
print()


print("결과4=>",re.findall(pattern, str1))

print("결과5=>",pattern.sub('aaa', str1))




#문제
#str2 문자열에서 온도의 평균 출력하기
str2 = "서울:25도,부산:23도,대구:27도,광주:26도,대전:25도,세종:27도"

pattern = re.compile("\\d{2}")
tlist=re.findall(pattern, str2)
print(tlist)
tlist = list(map(int, tlist))
print(sum(tlist)/len(tlist))

'''
  정규식에서 사용되는 기호
  1. () : 그룹
  2. \g<n> : n번째 그룹
  3. [] : 문자
     [a] : a 문자
     [a-z] : 소문자
     [A-Za-z] : 영문자(대소문자)
     [0-9A-Za-z] : 영문자+숫자
  4. {n} :n개 갯수
     ca{2}t : a 문자가 2개
      caat : true
      ct   : false
      cat  : false
     {n,m} :n개이상 m개이하 갯수
     ca{2,5}t : a 문자가 2개이상 5개 이하
      ct   : false
      cat  : false
      caat : true
      caaaaaaaaat : false
  5. \d : 숫자. [0-9]동일
  6. ?  : 0개또는 1개.
    ca?t : a문자는 없거나 1개    
    ct : true
    cat : true
    caat : false
  7. * : 0개이상  
    ca*t : a문자는 0개 이상
    ct : true
    cat : true
    caat : true
  8. + : 1개이상  
    ca+t : a문자는 1개 이상
    ct : false
    cat : true
    caat : true
  9. \s : 공백
     \s* : 공백문자 0개이상  
     \s+ : 공백문자 1개이상  
'''
'''
파일 읽기
open("파일명", 파일모드,[인코딩])
인코딩 : 파일의 저장방식. 기본값:cp949형식
파일모드
    r : 일기
    w : 쓰기. 기존파일의 내용을 무시. 새로운 내용으로 추가
    a : 쓰기. 기존의 파일의 내용 추가
    t : text 모드. (기본값)
    b : 이진모드. binary모드. 이미지, 동영상.....
'''


infp = open\
("C:\\Users\\KIM\\Py_Workspace\\2023-02-17.py","rt",encoding="UTF-8")
while True:
    instr = infp.readline()#한줄씩 읽기
    if instr == None or instr == '':
        break
    print(instr,end="") #화면에 출
infp.close()


#파일 쓰기 : 콘솔에서 내용을 입력받아 파일로 저장하기
#현재 폴더의 data.txt파일에 저장
outfp = open("data.txt","w",encoding="UTF-8")
while True:
    outstr = input("내용입력 => ")
    if outstr =='':
        break
    outfp.writelines(outstr+"\n")
outfp.close()


#위에꺼 열기
intxt = open("data.txt","rt",encoding="UTF-8")
while True:
    instr = intxt.readline()
    if instr == None or instr == '':
        break
    print(instr,end="") 
intxt.close()

#파일 한번에 읽어오기
intxt = open("data.txt","r",encoding="UTF-8")
print(intxt.read())
intxt.close()


intxt = open("data.txt","r",encoding="UTF-8")
print(intxt.readlines()) 
intxt.close()

'''
  readline() : 한줄씩 읽기
  read()     : 버퍼의 크기만큼 한번 읽기
  readlines() : 한줄씩 한번에 읽어서 줄별로 리스트로 리턴
'''

#이미지 파일을 읽어 복사하기
#apple.gif 파일을 읽어서 apple2.gif파일로 복사하기


infp = open("엄지로아콘.jpg","rb")#원본파일. 읽기위한파일
outfp = open("apple2.gif","wb") #복사본 파일. 쓰기위한 파일

while True:
    indata = infp.read() #설정된 버퍼의 크기만큼 읽기
    if not indata : #파일의 끝. EOF(End of File)
        break
    outfp.write(indata) #복사본 파일에 데이터 쓰기.
infp.close()
outfp.close()





# score.txt 파일을 읽기 모드로 열기
with open("score.txt", "r",encoding="UTF-8") as f:
    lines = f.readlines()  # 파일 내용을 리스트로 읽어오기

scores = []  # 점수를 저장할 빈 리스트

# 파일에서 읽어온 각 줄에서 이름과 점수를 추출하여 점수 리스트에 추가
for line in lines:
    name, score_str = line.strip().split(",")  # 쉼표로 분리
    score = int(score_str)  # 문자열을 정수로 변환
    scores.append(score)

# 점수의 총점과 평균 구하기
total = sum(scores)
average = total / len(scores)
print(scores)
print(lines)


# 결과 출력
print("총점:", total)
print("평균:", average)


import os
#현재 작업 폴더 위치 조회
print(os.getcwd())
#작업폴더의 위치 변경
os.chdir("c:/User/KIM")
os.chdir("c:/User/KIM/Py_Workspace")


import os.path

file = os.getcwd()
file = os.getcwd()+"\\aaa"
file

if os.path.isfile(file) :
    print(file,"은 파일입니다.")
elif os.path.isdir(file) :
    print(file,"은 폴더입니다")
else:
    print("파일이 없읍니다.")

if os.path.exists(file):
    print(file,'은존재합니다.')
else :
    print("은 없습니다.")




'''
    정규식 : 문자열의 형태를 지정할 수 있는 방법.
            import re 모듈 사용
            패턴 = re.compile(정규식 패턴) : 패턴 객체 생성
            리스트 = re.findall(패턴, 문자열) :
                문자열에서 패턴에 해당하는 문자열의 목록 리턴
            패턴.search(문자) : 패턴에 맞는 문자?
            패턴.sub(치환할문자, 대상문자열) : 치환
    파일 : open(파일명, 모드, [encoding])
        os.getcwd() : 작업폴더 조회
        os.chdir() : 작업폴더 변경
        os.path.isfile(file) : 파일?
        os.path.isdir(file) : 폴더?
    
'''


import os
print(os.listdir())
file="data.txt"
os.path.exists(file) #존재함?
#문제 : 작업파일의 하위파일 목록출력하기
#파일인경우 : 파일의 크기 os.path.getsize(파일명)
#폴더인경우 : 하위파일의 갯수
#작업폴더의 하위파일 갯수
len(os.listdir())


#현재 작업 폴더
cwd = os.getcwd();
cwd
for f in os.listdir():
    if os.path.isfile(f):
        print(f, ":파일, 크기:",os.path.getsize(f))
    elif os.path.isdir(f):
        print(f, ":폴더, 하위파일의 갯수:",len(os.listdir()))



os.mkdir("temp") #폴더 생성
os.rmdir("temp") #폴더 제거


###엑셀파일 읽기
import openpyxl

'''
    xlsx : openpyxl 모듈 사용
    xls  : xlrd 모듈 읽기
            xlwd 모듈로 쓰기
'''

filename = "data/sales_2015.xlsx"
#엑셀파일 전체
book = openpyxl.load_workbook(filename)


sheet = book.worksheets[0]
data=[]

for row in sheet.rows:
    line =[]
    print(row)
    #enumerate(row) : 목록에서
    #               l : 인덱스
    #               d : 데이터. 셀의 값.
    for l, d in enumerate(row):
        line.append(d.value) #셀의 내용을 line 추가
    #print(line) 
    data.append(line)
print(data)


###xls 형식의 엑셀파일 읽기
import xlrd
infile = "data/ssec1804.xls"
#workbook : 엑셀파일 전체 데이터
workbook = xlrd.open_workbook(infile)
#workbook.nsheets : sheet의 갯수
print("sheet 의 갯수", workbook.nsheets)

for worksheet in workbook.sheets():
    #worksheet : 한개의 sheet 데이터
    print("worksheet 이름:",worksheet.name)
    print("행의 수:", worksheet.nrows)
    print("컬럼의 수:", worksheet.ncols)
    
    for row_index in range(worksheet.nrows):
        for column_index in range(worksheet.ncols):
            print(worksheet.cell_value(row_index,column_index),",",end="")
            print()





### sqlite : 파이썬 내부에 존재하는 데이터 베이스
#

import sqlite3


# executescript : 여러개의 sql 문장을 실행.
#                 각각의 문장들은 ; 로 구분됨
'''
    drop table if exists items;
    item_id integer primary key :
        item_id 컬럼이 숫자형 기본키. 값이 자동 증가
    name text unique : 문자형 데이터, 중복불가
    
    create table items (item_id integer primary key,
                        name text unique,
                        price integer);
    => insert 구문 실행
    => item_id 컬럼을 제외 : 값이 자동 증가됨
    insert into items (name,price) values ('Apple',800);                  
    insert into items (name,price) values ('Orange',500);
    insert into items (name,price) values ('Banana',300);
'''

dbpath = "test.sqlite" #database파일 이름
conn = sqlite3.connect(dbpath)
cur = conn.cursor() #sql구문을 실행할 수 있는 객체
cur.executescript(
    '''
    drop table if exists items;
    create table items (item_id integer primary key,
                        name text unique,
                        price integer);
    insert into items (name,price) values ('Apple',800);                  
    insert into items (name,price) values ('Orange',500);
    insert into items (name,price) values ('Banana',300);
''')
conn.commit()
#execute : sql 명령문 실행
cur.execute("select * from items")
#fetchall() : select 결과를 전부 리스트 전달
item_list = cur.fetchall()
print(item_list)




'''
문제 : mydb sqlite 데이터베이스 생성
mydb에 member 테이블 생성하기
id char(4) primary key,
name char(15),
email char(20)
인 컬럼을 가진다
'''
#insert into member(id,name,email) vlaues ('kimsk','김삿갓','kim@aaa.bbb')

import sqlite3

# 데이터베이스 생성
conn = sqlite3.connect('mydb')

# 커서 생성
cur = conn.cursor()

# member 테이블 생성
cur.execute('''CREATE TABLE member (
            id char(4) primary key, 
            name char(15), 
            email char(20))''')

# 데이터 삽입
cur.execute("INSERT INTO member(id,name,email) VALUES ('kimsk','김삿갓','kim@aaa.bbb')")
# 조회
cur.execute('SELECT * FROM member')
items_list = cur.fetchall()
# 출력
for row in items_list:
    print(row)
    
conn.commit()





#mydb.sqlite3

while True :
    d1 = input("ID : ")#사용자 아이디
    if d1 == '':
        break
    d2 = input("name : ")
    d3 = input("email : ")
    sql = "insert into member(id,name,email) values ('"+d1+"','"+d2+"','"+d3+"')"
    
    print (sql)
    cur.execute(sql)
    conn.commit()

#mapping 방식
param = []
sql = "insert into member (id,name,email) values (?,?,?)"
param.append("kic4")
param.append("ddddd")
param.append("kic4@mmm.com")
cur.execute(sql, param)
conn.commit()

print(items_list)

conn.close()



#Q1
'''
화면에서 주민등록번호를 000000-0000000 형태로 입력받는다.
   주민등록번호 뒷자리의  첫 번째 숫자는 성별을 나타낸다. 
   주민등록번호에서 성별을 나타내는 숫자를 조회하여
   성별을 나타내는 숫자가 1,3 이면 남자로 2,4면 여자로 출력한다. 
   그외는 내국인아님으로 출력한다.
   -이 없는 경우는 '주민번호 입력오류' 출력하기
'''

import re

while True:
    jumin = input("주민번호를 입력 해 주세요 ex)000000-0000000 : ")
    if jumin == "":
        break
    if re.match(r"\d{6}-\d{7}", jumin):
        gen = int(jumin.split("-")[1][0])
        if gen == 1 or gen == 3:
            print("남자입니다.")
        elif gen == 2 or gen == 4:
            print("여성입니다.")
        else:
            print("내국인이 아닙니다.")
    else:
        print("잘못된 주민번호 형식입니다.")



'''
소문자와 숫자로 이루어진 문자를 암호화 하고 복호화 하는 프로그램 작성하기
  원래 문자 : a b c d e f g h i j k l m n o p q r s t u v w x y z 
  암호 문자 : ` ~ ! @ # $ % ^ & * ( ) - _ + = | [ ] { } ; : , . /
  원래 숫자 : 0 1 2 3 4 5 6 7 8 9 
  암호 숫자 : q w e r t y u i o p
[결과]
문자를 입력하세요 : abc123
암호화
`~!wer
복호화
abc123

'''


import re


origin_t = "a b c d e f g h i j k l m n o p q r s t u v w x y z"
pass_t = "` ~ ! @ # $ % ^ & * ( ) - _ + = | [ ] { } ; : , . /"
origin_n = "0 1 2 3 4 5 6 7 8 9"
pass_n = "q w e r t y u i o p"
o_t_l = origin_t.split(" ")
p_t_l = pass_t.split(" ")
o_n_l = origin_n.split(" ")
p_n_l = pass_n.split(" ")


while True :
    in_str = input("문자를 입력하세요 : \n")
    if in_str == "" :
        break
    str_count = len(in_str)
    aa =[]
    bb = []
    for i in range(0,str_count) :
        temp = in_str[i]
        if temp in o_t_l :
            aa.append(o_t_l.index(in_str[i]))
            bb.append(p_t_l[aa[i]])
        if temp in o_n_l :
            aa.append(o_n_l.index(in_str[i]))
            bb.append(p_n_l[aa[i]])
        
    print("암호화\n","".join(bb))
    print("복호화\n",in_str)

'''
16진수를 입력하면 16진수 인지 아닌지 판단하여
16진수가 맞으면 10진수로 변경하기.
16진수가 아닌 경우 16진수 아님을 출력하기
'''
def is_hex(s):
    hex_digits = set("0123456789ABCDEF")
    for char in s:
        if char.upper() not in hex_digits:
            return False
    return True

while True:
    s = input("16진수를 입력하세요: ")
    if s == "":
        break
    if is_hex(s):
        n = int(s, 16)
        print("10진수:", n)
    else:
        print("16진수 아님")