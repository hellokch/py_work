# -*- coding: utf-8 -*-

from django.urls import path
from . import views
#같은 폴더 내 있는 py파일은 .만찍어도 댐

urlpatterns = [
    path("index/", views.index, name="index"), #localhost:5000/member/index/
    path("join/", views.join, name="join"),
    path("login/", views.login, name="login"),
    path("main/", views.main, name="main"),
    path('logout/', views.logout, name='logout'),
    path('info/<str:id>/', views.info, name='info'),
    path('update/<str:id>/', views.update, name='update'),
    path('delete/<str:id>/', views.delete, name='delete'),
    path('passchg/', views.passchg, name='passchg'),
    path('list/', views.list, name='list'),
    path('picture/', views.picture, name='picture'),
]