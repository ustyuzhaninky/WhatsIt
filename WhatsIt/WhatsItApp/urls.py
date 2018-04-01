from django.conf.urls import url
from django.contrib import admin
from .views import *

urlpatterns = [
    url(r'^$', index),
    url(r'^admin/', admin.site.urls),
    url(r'^push_feed$', push_feed),
    url(r'^pusher_authentication', pusher_authentication),
]
