from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('configure/', views.configure, name='configure'),
    path('execute/', views.execute_agent, name='execute_agent'),
    path('history/', views.history_list, name='history_list'),
    path('history/<int:pk>/', views.history_detail, name='history_detail'),
]
