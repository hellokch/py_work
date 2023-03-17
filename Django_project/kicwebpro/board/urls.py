# -*- coding: utf-8 -*-

from django.urls import path
from . import views
#같은 폴더 내 있는 py파일은 .만찍어도 댐

urlpatterns = [
    path("index/", views.index, name="index"), #localhost:5000/board/index/
    path("write/", views.write, name="write"),
    path("list/", views.list, name="list"),
    path("info/<int:num>", views.info, name="info"),
    path("update/<int:num>", views.update, name="update"),
    path("delete/<int:num>", views.delete, name="delete"),
]