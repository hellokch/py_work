# Create your views here.
from django.shortcuts import render
from .models import Board
from django.utils import timezone
from django.http import HttpResponseRedirect
from django.core.paginator import Paginator

# localhost:5000/board/index/*


def index(request):
    print('view.index')
    return render(request, "board/index.html")

# python 에서는 html에서 POST방식을 보낼 때, form문 안에서 token 태그를 입력해야함


def write(request):
    if request.method != "POST":  # POST가 아님 => GET방식의 요청
        return render(request, "board/write.html")
    else:  # POST방식 요청
        try:
            filename = request.FILES["file1"].name
            handle_upload(request.FILES["file1"])
        except:
            filename = ''
        b = Board(name=request.POST["name"],
                  pass1=request.POST["pass"],
                  subject=request.POST["subject"],
                  content=request.POST["content"],
                  regdate=timezone.now(),
                  readcnt=0,
                  file1=filename)
        b.save()
        return HttpResponseRedirect("../list")


def handle_upload(f):
    with open("file/board/"+f.name, "wb") as dest:
        for ch in f.chunks():
            dest.write(ch)


def list(request):
    pageNum = int(request.GET.get("pageNum", 1))
    all_boards = Board.objects.all().order_by("-num")
    paginator = Paginator(all_boards, 5)
    listcount = Board.objects.count()
    board_list = paginator.get_page(pageNum)
#    pagecount = board_list.paginator.count()
#    print(pagecount)
    return render(request, "board/list.html",
                  {"board": board_list, "listcount": listcount, "endpage": 2})


def info(request, num):
    board = Board.objects.get(num=num)
    board.readcnt += 1
    board.save()
    return render(request, "board/info.html", {"b": board})


def update(request, num):
    if request.method != "POST":
        board = Board.objects.get(num=num)
        return render(request, "board/update.html", {"b": board})
    else:
        try:
            board = Board.objects.get(num=num)
            pass1 = request.POST["pass"]
            if board.pass1 != pass1:
                context = {"msg" : "비밀번호 오류",\
                           "url" : "../../update/"+str(num) + "/"}
                return render(request, "alert.html", context)

            board.name = request.POST["name"]
            board.pass1 = request.POST["pass"]
            board.subject = request.POST["subject"]
            board.content = request.POST["content"]
            board.regdate = timezone.now()

            # 사진 업로드 처리
            if 'file1' in request.FILES:
                file = request.FILES['file1']
                handle_upload(file)
                board.file1 = file.name

            board.save()

            return HttpResponseRedirect("/board/info/"+str(num))

        except Exception as e:
            print(e)
            context = {"msg" : "게시물 수정 실패",\
                       "url" : "/board/update/"+str(num)}
            return render(request,"alert.html",context)

def delete(request, num):
    if request.method != 'POST':
        return render(request, 'board/delete.html', {"num": num})
    else:
        board = Board.objects.get(num=num)
        pass1 = request.POST["pass"]
        print(pass1)
        if board.pass1 != pass1:
            context = {"msg": "비밀번호 오류",
                       "url": "/board/delete/"+str(num)}
            return render(request, "alert.html", context)
        try:
            board.delete()
            return HttpResponseRedirect("/board/list")
        except:
            context = {"msg": "게시물 삭제 실패",
                       "url": "/board/delete/"+str(num)}
            return render(request, "alert.html", context)
