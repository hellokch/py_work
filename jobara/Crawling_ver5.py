# pip install selenium
# pip install beautufulsoup4
import requests
import time
import pandas as pd
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tqdm import tqdm
from konlpy.tag import Okt

stopwords = ['그리고', '하지만', '그런데', '따라서', '그래서', '또한', '이러한',
             '바로', '이런', '결국','합니다','했습니다','입니다', '그것', '그때', 
             '그들', '그것들', '그리고', '그만큼', '그러나', '그래서', '그런데',
             '그렇게', '그리하여', '그것으로', '그중에서', '그대로', '그리고도', 
             '그것만으로', '그래도', '그것들의', '그럼에도 불구하고', '그것과 같은',
             '그와 같은',  '그는', '그때에', '그러한', '그저', '그녀를', '그녀의', 
             '그들의', '그런', '그에', '그럼으로써','하다', '되다', '있다', '않다',
             "있는", "그", "할", "이", "것", "하다", "그것", "이런", "하는", 
            "그런", "같은", "있다", "많은", "어떤", "이러한", "저", "때문에", 
            "에서", "그리고", "또는", "그러나", "그래서", "그렇게", "그렇지만", 
            "그때", "그들", "이것은", "그럼", "어떻게", "그러면", "이러한", 
            "그것이", "이것을", "그것은", "저는", "저의", "이러한", "저는", "저의",
            "그런데", "그러한", "그럴", "이렇게", "이상", "모든", "매우", "아주", 
            "정말", "너무", "그냥", "무엇인가", "무엇", "무슨", "어디", "누가", 
            "누구", "언제", "언젠가", "어느", "어떤", "이런", "저런", "하나", "둘",
            "셋", "넷", "다섯", "어쩌면", "어쨌든", '서','사항','나','하','년',
            "어떻게든", "어떤지", "어쩐지", "어쨌든", "어떤가요", "뭐", "어떡해", 
            "그게", "이게", "그거", "이거", "저거", "그저께", "어제", "오늘", 
            "내일", "모레", "이번", "그번", "저번", "내년", "작년", "그래서", 
            "그러니까", "그렇다면", "이러한", "같이", "함께", "열심히", "잘", 
            "매우", "진짜", "어쨌든", "그러면서", "물론", "뿐", "지와", "하셨나요",
            "최",'을','에','의','를','한','주','이','적','해','십','으로','하여',
            '시오','과','구체','대해','및','와','위해','하고','바랍니다','했던',
            '서술','은','후','된','등','가','수','중','하시오','하기','인','는',
            '가지','여','위','주세요','해주세요','이를','점','들','하게','최대',
            '이나','입','만의','때','니까','있다면','고','해주십시오','자','원',
            '로','지','대한','어떠한','에게','귀하','하십시오','이며','대하',
            '까지','성','큰','내','화','왜','하세요','하였는지','도','관','시',
            '하는지','예','상세','따라','간략히','있는지','끼친','했는지','단락',
            '항목','대','부분','못','삼','거나','되','남는','있어','이고','byte',
            '최대한','들어','으로서','기','에서의','하며','이었으며','함','인가요',
            '하고자','있습니까','임','줄','하신','더','하면서','히','기를','될',
            '하지','제','세','현','갖게','보다','있다고','시키기','게','과의','간',
            '내외','있을지','입니다','연','여러분','개','기와','있으면','또한','외',
            '계','형','하였던','되기','만','싶은지','하는데','키','워드','적용',
            '동','소','하십니까','되지','력','인해','했을','과는','한다면','정'
            '간략하게','Bytes','이었던','부항','께서','꼭','이하','합니다','전',
            '두','간의','자세하게','문항','된다고','교','바이트','있게','글자',
            '해보고','부문','여러','자세히','왔는지','택','시켜','않았을','귀하는',
            '온','My','안','야','되고','이라','라고','모두','먼저','단','경',
            '냈던','제외','출','약','회','받은','다','비해','적어주세요','군',
            '타','면','간단히','울','에도','불구','바를','사','합','있으며','A',
            '하였고','하시고','이라도','있도록','쓰시오','지어','아니라','하기에',
            '순','업','보십시오','까지의','bytes','당','이내','내의','지키기',
            '오','어','데','또','준','으로써','있었는지','해주시고','라는',
            '있을','에는','아닌','되어야','했거나','씩','신','한다고','식','네',
            '이어야만', '나타낼','했습니까','양','싶으며','찾아서','강','하셨는지',
            '으로의','항','목별','상의','밖에','만큼','미만','What','바','됩니다',
            '간단하게','요약','이었거나','되었던','같습니다','구','어떠했는지','요',
            '뒤','영','직스','으로부터','you','스','하면','란','B','해야만',
            '이라는','어떠합니까','에서도','아니면','띄어쓰기','바라며','나를',
            '후의','번','하였을','되었을','지를','했으며','있겠는지를','주로','님',
            '아는','거쳤으며','있는지와','하시는지','적어주십시오','별로','ex','엔',
            '주시길','BYTE','이었나요','좋습니다','어떠하였는지','음','하셨던',
            '했는지를','좀','써','싶습니까','틀','라','이었고','했는지에','해주시',
            '때문','보세요','써주십시오','하는지와','있을지를','니','즈','라면',
            '였으며','ㅇ','있','a','하시어','하시오이내','싶으신','적으시오','트',
            '입다','아','으로서의',]


def page_calc(tot_num):
    if tot_num % 20 == 0:
        max_page = tot_num // 20
    else:
        max_page = tot_num // 20 + 1
    return max_page

def init_data_url(start_page, end_page):
    pre_url = 'https://www.jobkorea.co.kr'
    options = Options()
    options.headless = True  
    driver = webdriver.Chrome('/chromedriver/chromedriver.exe',\
                              options=options)
    names = []
    fields = []
    urls = []
    idxs = []
    grades = []
    
    for i in tqdm(range(start_page, end_page+1)):
        pagenum = i
        
        search_url =\
            f'{pre_url}/starter/'\
            f'PassAssay?Page={pagenum}&Pass_An_Stat=1'
        driver.get(search_url)
        time.sleep(1)
        response = requests.get(driver.current_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for name in soup.select('.titTx'):
            names.append(name.text.strip())
        for field in soup.select('.field'):
            if field.text.strip() not in ['인턴', '신입']:
                fields.append(field.text.strip())
        for url in soup.find_all('p', {'class': 'tit'}):
            full_url = pre_url + url.find('a')['href']
            url_without_params = full_url.split('?')[0]
            urls.append(url_without_params)
            idx = full_url.split('View/')[1].split('?')[0]
            idxs.append(idx)
        for grade in soup.select('.grade'): #전문가 별점 추가
            grade_text = grade.text.strip()
            grades.append(grade_text.strip())

    data = {'name': names, 'field': fields,'url': urls, 'grades': grades }
    idxs = pd.to_numeric(idxs)
    df = pd.DataFrame(data, index = idxs)
    
    driver.quit()
    filename = f'init_{start_page}to{end_page}'
    df.to_csv(f'data/{filename}.csv', index=True)
    print(filename + ' 저장완료')
    return filename



def read_context(file_name):
    in_df = pd.read_csv(f'data/{file_name}', index_col=0)
    
    name = []
    field = []
    board_num = []
    grades = []
    A = []
    P = []
    
    for url in tqdm(in_df['url']):
        try:
            response = requests.get(url)
            response.raise_for_status()
        except (requests.exceptions.HTTPError,\
                requests.exceptions.RequestException):
            print(f'{url}에서 오류 발생. 다음 URL로 넘어갑니다.')
            continue

        soup = BeautifulSoup(response.text, 'html.parser')

        que_div_txs = soup.find_all('span', 'tx')[6:]
        ans_div_txs = soup.find_all('div', 'tx')[:len(que_div_txs)]

        if ans_div_txs is None:
            print(f'{url}에서 ans_div_txs을 찾을 수 없습니다. 넘어갑니다.')
            continue

        company_name = in_df.loc[in_df['url'] == url, 'name'].iloc[0]
        industry_field = in_df.loc[in_df['url'] == url, 'field'].iloc[0]
        grade = in_df.loc[in_df['url'] == url, 'grades'].iloc[0]
        board_number = \
            in_df.loc[in_df['url'] == url, 'url'].iloc[0].split('View/')[1]
        answer_list = []
        points = 0
        for ans_div_tx in ans_div_txs:
            for b_tag in ans_div_tx.find_all('b'):
                if 'good' in b_tag['class']:
                    points += 1
                elif 'bad' in b_tag['class']:
                    points -= 1

            for tag in ans_div_tx('span'):
                tag.decompose()
            for tag in ans_div_tx('p'):
                tag.decompose()
            answer = ans_div_tx.get_text(strip=True).replace(",", "")\
                .replace('\n', '').replace('\r', '')
            answer_list.append(answer)
            
        name.append(company_name)
        field.append(industry_field)
        board_num.append(board_number)
        grades.append(grade)
        
        A.append(answer_list)
        P.append(points)

    out_df = pd.DataFrame({"게시판번호": board_num, "기업명": name,\
                           "산업분류": field,"등급" : grades ,"점수": P, "답변": A})
    out_df.to_csv(f'data/context_{file_name}', index=False, encoding='utf-8')
    print(f'save)complete : context_{file_name}')

def text_cleaning(text):

    nnumber = re.compile('[0-9]+자')
    text = nnumber.sub("", text)
    nhangul = re.compile('[^ ㄱ-ㅣ 가-힣 a-zA-Z]+')
    text = nhangul.sub("", text)
    sub1 = nhangul.sub("",text)
    text = sub1.replace(" 이내","")
    result = ' '.join(text)
    return result

def tokenize(text):
    words = okt.morphs(text, stem=True) 
    words = [word for word in words if word not in stopwords] 
    return words




### 필요한 과정만 주석 풀고 실행하시면 됩니다.
## 처음에 해당 폴더 내 ./data/ 디렉토리를 만들어야함
# ==============================
#총 이력서의 건수를 입력하세요 
end_page = page_calc(40)

# 시작 페이지와 끝 페이지 입력 (url 불러오기)
init_data_url(1,end_page)

# url을 불러온 파일 이름을 실행시키면 됩니다!
read_context('init_1to179.csv')


df= pd.read_csv('data/init_1to2.csv', index_col=0)
df.head(5)

# 텍스트 클리닝
df = pd.read_csv('data/context_init_1to179_(1).csv')
test_list = df['답변'].apply(text_cleaning).tolist()
df["답변"] = test_list
df.head()

# null 행 제거

# answer 값이 비어있는 행 확인
null_answer = df[df['답변'] == '']
print(null_answer.count())

# answer 값이 비어있는 행 삭제
cleaned_df = df[df['답변'] != '']
cleaned_df.head()

# 답변 데이터를 형태소 단위로 쪼개기

okt = Okt()
Tokens = []
for question in tqdm(cleaned_df['답변']):
    token = tokenize(question)
    Tokens.append(token)

cleaned_df['답변'] = Tokens

# -크린- 데이터 저장
cleaned_df.to_csv('data/clean_data.csv')
