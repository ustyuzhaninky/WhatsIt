from django.shortcuts import render

# Create your views here.
from PIL import Image
from django.views.generic.edit import FormView
from django.views.generic import DetailView
from django.utils.translation import ugettext as _
from django.core.urlresolvers import reverse

from django.shortcuts import render, HttpResponse
from django.http import JsonResponse
from .forms import *
from pusher import Pusher
import json

from .nn.model import predictor
from .nn.model import imgPreload

classificator = predictor()
classificator.load()
#instantiate pusher
pusher = Pusher(app_id=u'501634', key=u'05d3f559f85ac05ed32a', secret=u'1ddffc3f97ede376f5f3', cluster=u'eu', ssl=True)
#pusher.trigger('my-channel', 'my-event', {'message': 'hello world'})
# Create your views here.
# function that serves the welcome page
def index(request):
    # get all current photos ordered by the latest
    all_documents = Feed.objects.all().order_by('-id')
    # return the index.html template, passing in all the feeds
    return render(request, 'index.html', {'all_documents': all_documents})

#function that authenticates the private channel
def pusher_authentication(request):
    channel = request.GET.get('channel_name', None)
    socket_id = request.GET.get('socket_id', None)
    auth = pusher.authenticate(
        channel = channel,
        socket_id = socket_id
    )

    return JsonResponse(json.dumps(auth), safe=False)
#function that triggers the pusher request
def push_feed(request):
    # check if the method is post
    if request.method == 'POST':
        # try form validation
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.save()
            # trigger a pusher request after saving the new feed element
            resp = classificator.predict(imgPreload(f.document))
            lables = ['Керамика', 'Стекло', 'Металл', 'Пластик', 'Дерево']
            agg = ''
            for i in range(len(resp)):
                agg += lables[i] + ': ' + str(resp[i]*100) + '%\n'
            pusher.trigger(u'a_channel', u'an_event', {u'description': f.description, u'document': f.document.url})
            return HttpResponse('ok'+':'+agg)
        else:
            # return a form not valid error
            return HttpResponse('form not valid')
    else:
        # return error, type isnt post
        return HttpResponse('error, please try again')