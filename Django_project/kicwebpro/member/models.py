from django.db import models

# Create your models here.

class Member(models.Model):
    id = models.CharField(max_length=50, primary_key=True)
    pass1 = models.CharField(max_length=128)
    name = models.CharField(max_length=20)
    gender = models.CharField(max_length=10)
    tel = models.CharField(max_length=20)
    email = models.EmailField(max_length=254)
    picture = models.ImageField(upload_to="images/", null=True, blank=True)

    def __str__(self):
        return self.id
    