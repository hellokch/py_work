# Generated by Django 4.1.7 on 2023-03-13 12:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Board",
            fields=[
                ("num", models.AutoField(primary_key=True, serialize=False)),
                ("name", models.CharField(max_length=30)),
                ("pass1", models.CharField(max_length=20)),
                ("subject", models.CharField(max_length=100)),
                ("content", models.CharField(max_length=4000)),
                ("regdate", models.DateField(null=True)),
                ("readcnt", models.IntegerField(default=0)),
                ("file1", models.CharField(max_length=300)),
            ],
        ),
    ]
