# -*- coding: utf-8 -*-
# Generated by Django 1.11.3 on 2018-04-01 11:14
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('WhatsItApp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='feed',
            name='document',
            field=models.ImageField(upload_to='static/documents/'),
        ),
    ]