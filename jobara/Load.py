# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 17:48:47 2023

@author: 김충환
"""

# AIzaSyBw7dSCaWXCCqi4zG-TH8bjSHxpVEJ2CAw

# 360183029918-bni9lufkerv6h8teknk7p1o5u34j3nrn.apps.googleusercontent.com
# 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


# pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Google API credentials 파일 경로
creds_path = 'path/to/credentials.json'
token_path = 'path/to/token.json'

# 구글 API 권한 인증 및 빌드
creds = Credentials.from_authorized_user_file(token_path)
service = build('drive', 'v3', credentials=creds)

# 파일 ID와 파일명
file_id = 'your_file_id'
file_name = 'your_file_name'

try:
    # 파일 다운로드
    file = service.files().get_media(fileId=file_id).execute()
    with open(file_name, 'wb') as f:
        f.write(file)
    print(f'File "{file_name}" has been downloaded.')
except HttpError as error:
    print(f'An error occurred: {error}')
    file = None