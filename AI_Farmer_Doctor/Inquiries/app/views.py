from django.contrib.auth.decorators import login_required
from django.shortcuts import render
@login_required
def index(request):
    return render(request, 'index.html')
@login_required
def search(request):
    return render(request, 'app/search.html')

