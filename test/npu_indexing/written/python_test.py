import pdb
from typing import Dict, Any
import dataclasses
import functools

import sympy
from sympy import *
import numpy
import operator
import os
class A:
    #dict : Dict[str, Any]
    def __init__(self, variable):
        self.a = 10
        self.a_variable = variable
        self.dict = {}
    def do_something(self):
        def inner() :
            print("A.do_something inner")
        inner()


class B(A):
    def __init__(self, variable=None):
        super().__init__(variable)
        self.b = 15
        self.a = 20

    def do_something(self):
        def inner() :
            print("B.do_something inner")
        super().do_something()

def derive_test() :
    a = A(None)
    print(f"a.a = {a.a}")
    b = B(None)
    # a.__class__ = B
    print(f"b.a = {b.a}, b.b = {b.b}")

    array: list[A] = []
    array.append(a)
    array.append(b)

    print(f"a.a= {getattr(array[0], 'a')}")
    print(f"b.b= {getattr(array[1], 'b')}")
    setattr(b, "c", [100])
    print(f"b.c= {getattr(array[1], 'c')}")

    b.do_something()
def dict_test() :
    dict:Dict[Any:Any] = {}
    dict["any"] = "anything"
    a = dict.get("dummy")
    print(a)

def ref_test():
    a = A(None)
    print(a.dict)
    dict1 = a.dict
    dict1["new"] = None
    list1 = [a.dict.values()]
    list1 += ["new value"]
    print(a.dict)
@dataclasses.dataclass
class Parent:
    a : int
    b : str

    @staticmethod
    def static_method():
        print("in parent's static_method")
    @classmethod
    def class_method(cls):
        print("in parrent's classmethod")
        cls.static_method()



@dataclasses.dataclass
class Child(Parent):
    c : list

    @classmethod
    def class_method(cls):
        print("in clild's classmethod")
        cls.static_method()

    @staticmethod
    def static_method():
        print("in clild's static_method")

def dataclass_test() :
   p = Parent(1, "parent")
   c = Child(2, "parent", ["child"])
   #print(p.b, c.c)
   print(p,c)


class NumelList(list):

    def numels(self):
        numel = functools.reduce(lambda a, b: a * b, self)
        return numel

    def __eq__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel == numel2

    def __mul__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel * numel2
    def __rmul__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel * numel2

    def __add__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel + numel2
    def __radd__(self, other):
        numel = self.numels()
        numel2 = other.numels() if isinstance(other, NumelList) else other
        return numel + numel2
def list_test():
    list = [2,4]
    list1 = NumelList(list)
    list2 = NumelList([8])
    print(list1 == list2)
    print(list1 == 8)
    print(8 == list1 )
    print(list2 * list1)
    print(list2 + list1)

    print(8 * list1)
    print(8 + list1)

    list = ["a", "existed", "hello","world"]
    index = list.index("existed")
    del list[index]
    if "not existing" in list :
        index = list.index("not existing")
        del list[index]

def classmethod_test():
    Parent.class_method()
    Child.class_method()


def sort_test() :
    test_list = [60, 20, 1, 5]
    print(sorted(test_list))
    print(sorted(test_list, reverse = True ))

def sympy_product(it):
    return functools.reduce(operator.mul, it, sympy.Integer(1))
def sympy_test():
    def eval(x, expr, value, use_numpy = True ):
        func = lambdify(x, expr, "numpy") if use_numpy else  lambdify(x, expr )
        result = func( value)
        if isinstance(result, numpy.ndarray) :
            result = set(result)
        return result
    # import sympy
    x = sympy.Symbol('x', integer=True, nonnegative=True)
    y = sympy.Symbol('y', integer=True, nonnegative=True)
    expr = ((x // 8 % 8) )
    range = numpy.arange(128, dtype=int)
    result = eval(x, expr, range )
    print(result)
    expr = x
    result = eval(x, expr, 10, False)
    print(result)

    expr = ((x // 8 +  2))
    expr1 = expr.subs(x//8, y)
    print(expr, expr1)
    expr = sympy.Integer(9)
    print((expr // 3))

    z = sympy.Integer(0)
    z = z + x *127 + y
    print(z)
    print(sympy.simplify(z))
    x1 = sympy.Symbol('x1', integer=True, nonnegative=True)
    y2 = sympy.Symbol('y2', integer=True, nonnegative=True)
    y3 = sympy.Symbol('y3', integer=True, nonnegative=True)
    expr = x1 + 8912896*((y2//1056)) + 278528*(y2 % 322) + 8192*(y2 //32)
    expr1 = 8912896*(y2//1056)
    result = expr.find(expr1)
    if result :
         print(type(result), result)
    replacements = {expr1:y3}
    result = expr.subs(replacements)
    print(result)

    def find_index_in_substitute(index) :
        return any([index.find(key) for key in replacements.keys()])

    print(find_index_in_substitute(expr))

   # x, y, z = symbols('x y z')

    # calling find() method on expression
    #geek = (3 * x + log(3 * x) + sqrt((x + 1) ** 2 + x)).find(expr(3 * x))
    #print(geek)

    #var_range = {'x0' : sympy.Integer(4096), 'x1':sympy.Integer(4), 'x2':sympy.Integer(1024)}
    #var_list = [sympy.Integer(4096), sympy.Integer(4), sympy.Integer(1024)]

   # numel = (sympy_product(s)  for s in var_list)
#    numel = functools.reduce(lambda a, b: a * b, (v for k,v in var_range if k != 'x0'))
 #   print(f"numel={numel}")

def dict_test():
    z1="z1"
    dict = {z1:1, "x1":1, "y1":1}
    print(dict.keys())
    for x in dict.keys() :
        print(x)
    dict = {"x1":{1:2}}
    print("value =", dict["x1"][1])
def reduce_test ():
    numels = [1024,2048, 1]
    total_numels = sum(1 if x > 1 else 0 for x in numels)
    print(total_numels)
    numels = [1024,  1]
    total_numels = sum(1 if x > 1 else 0 for x in numels)
    print(total_numels)

    numels = [1024,2048,]
    total_numels = sum(1 if x > 1 else 0 for x in numels)
    print(total_numels)
def slice_test() :
    numels = (8,1024,8, 2048)
    slice =  numels[0:-1]
    print(max(slice))

if __name__ == "__main__":
    #list_test()
    #slice_test()
    #reduce_test()
    dict_test()
    sympy_test()
    #dataclass_test()
    #sympy_test()
    #sort_test()
    #classmethod_test()
    #derive_test()
    #constructor_test()
