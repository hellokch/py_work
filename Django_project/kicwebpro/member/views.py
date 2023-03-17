from django.shortcuts import render, redirect
from django.utils import timezone
from django.http import HttpResponseRedirect
from django.core.paginator import Paginator
from .models import Member
import time
from django.contrib import auth
from intercept.intercepter import loginIdchk,loginchk,adminchk
import os
# localhost:5000/board/index/*


def index(request):
    print('view.index')
    return render(request, "member/index.html")

def main(request):
    if not request.session.get('login'):
        return redirect('member:login')
    else:
        return render(request, 'member/main.html')

def join(request):
    if request.method != "POST":
        return render(request, "member/join.html")
    else :
        print("join")
        member = Member(id=request.POST["id"],\
                        pass1=request.POST["pass"],\
                        name=request.POST["name"],\
                        gender=request.POST["gender"],\
                        tel=request.POST["tel"],\
                        email=request.POST["email"],\
                        picture=request.POST["picture"],\
                        )
        member.save()
        return HttpResponseRedirect("/member/login/")
    print('view.join')
    return render(request, "member/join.html")


def login(request):
    print("1:",request.session.session_key)
    if request.method != "POST" :
        return render(request,"member/login.html")
    else :
        id1=request.POST["id"]
        pass1=request.POST["pass"]
        
        try :
            member = Member.objects.get(id=id1)
            if member.pass1 == pass1 :
                request.session["login"] = id1
                time.sleep(1)
                print("2:", request.session.session_key)
                return HttpResponseRedirect("/member/main/")
            else:
                context = {"msg":"비밀번호가 틀립니다.",\
                           "url":"/member/login/"}
                return render(request,"alert.html",context)
        except :
            context = {"msg":"아이디를확인하세요"}
            return render(request,"member/login.html",context)
            
        return HttpResponseRedirect("/member/main/")

def logout(request) :
    print(request.session.session_key)
    auth.logout(request)
    return HttpResponseRedirect('/member/login/')


def info(request, id):
    print('info')
    member = Member.objects.get(id=id)
    return render(request, "member/info.html",{"mem":member})


def update(request, id):
    member = Member.objects.get(id=id)
    if request.method !="POST":
        return render(request, "member/update.html",{"mem":member})
    else :
        if request.POST["pass"] == member.pass1:
            member = Member(id=request.POST["id"],\
                            pass1=request.POST["pass"],\
                            name=request.POST["name"],\
                            gender=request.POST["gender"],\
                            tel=request.POST["tel"],\
                            email=request.POST["email"],\
                            picture=request.POST["picture"],\
                            )
            member.save()
            return HttpResponseRedirect("/member/info/"+id+"/")
        else :
            context = {"msg":"비밀번호 오류",\
                       "url":"/member/update/"+id+"/"}
            return render(request, "alert.html",context)
        
def delete(request,id):
    print("delete")
    if request.method != "POST":
        return render(request,"member/delete.html",{"id":id})
    else :
        member = Member.objects.get(id=id)
        if member.pass1 == request.POST["pass"]:
            member.delete()
            auth.logout(request)
            context={"msg":"탈퇴완료",\
                     "url":"/member/login/"}
            return render(request,"alert.html",context)
        else :
            member = Member.objects.get(id=id)
            context={"msg":"비밀번호 오류",\
                     "url":"/member/delete/"+id+"/"}
            return render(request, "alert.html",context)

@loginchk
def passchg(request):
    login = request.session["login"]
    if request.method != "POST":
        return render(request,"member/passchg.html")
    else :
        member= Member.objects.get(id=login)
        if member.pass1 == request.POST['pass']:
            member.pass1 = request.POST["passchg"]
            member.save()
            context={"msg":"비밀번호 수정완료",\
                     "url":"/member/info/"+login+"/",\
                         "closer":True}
            return render(request,"member/password.html",context)
        else :
            context={"msg":"비밀번호 오류",\
                     "url":"/member/password/"+login+"/",\
                         "closer":True}
            return render(request,"member/password.html",context)

@adminchk
def list(request):
    mlist = Member.objects.all()
    return render(request,"member/list.html",{"mlist":mlist})


def picture(request):
    if request.method != 'POST':
        return render(request, 'member/pictureForm.html')
    else :
        fname = request.FILES["picture"].name
        handle_upload(request.FILES["picture"])
        return render(request, "member/picture.html",{"fname":fname})
    
def handle_upload(f):
    folder_path = "../file/member/"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    with open(folder_path + f.name, "wb") as dest:
        for ch in f.chunks():
            dest.write(ch)




