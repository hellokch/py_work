# Generated by Django 4.1.7 on 2023-03-16 12:43

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("member", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="member",
            name="name",
            field=models.CharField(default=1, max_length=20),
            preserve_default=False,
        ),
    ]