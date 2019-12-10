import time

def timetest(input_func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        print("Method Name - {0}, Args - {1}, Kwargs - {2}, Execution Time - {3}".format(
            input_func.__name__, 
            args, 
            kwargs, 
            end_time - start_time 
            )
        )
        return result
    return timed

@timetest
def foobar(*args, **kwargs):
    time.sleep(0.3)
    print("inside foobar")
    print (args, kwargs)

foobar(["hello, world"], foo=2, bar=5)



''' METHOD DECORATOR '''

def method_decorator(method):
    
    def inner(city_instance):
        if city_instance.name == "SFO":
            print("Its a cool place to live in.")
        else:
            method(city_instance)
    return inner


class City(object):

    def __init__(self, name):
        self.name = name

    @method_decorator
    def print_test(self):
        print( self.name)

p1 = City("SFO")
p1.print_test()

''' Chaining Decorators '''

def makebold(f):
    return lambda: "<b>" + f() + "</b>"
def makeitalic(f):
    return lambda: "<i>" + f() + "</i>"

@makebold
@makeitalic
def say():
    return "Hello"
say()         # Result : '<i><b>Hello</b></i>'
print(say())  # Result :  <i><b>Hello</b></i>


