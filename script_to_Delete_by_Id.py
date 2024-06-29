import os
import django

### Define the environment variable DJANGO_SETTINGS_MODULE
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SmartFood.settings')

### Initialize django
django.setup()


### Import the model

from accounts.models import Food


def options():
    print('----------------------------')
    print('Deleting records')
    print('If you want to delete a record just enter the index')
    print('If you want to exit just enter -1')
    answer = int(input('Enter the number: '))
    return answer
    

def eliminate_instance(food_id):
    try:
        # Look for the instance
        food_to_delete = Food.objects.get(index=food_id)
        food_to_delete.delete()
        print(f"Registro con ID {food_id} eliminado exitosamente.")
    except Food.DoesNotExist:
        print(f"Registro con ID {food_id} no encontrado.")


def menu():
    answer = 0
    while answer != -1:
        answer = options()
        if answer != -1:
            eliminate_instance(answer)


menu()