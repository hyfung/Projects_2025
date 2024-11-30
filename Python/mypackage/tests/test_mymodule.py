import pytest
from .. import mymodule1
from .. import mymodule2

def test_mymodule1_myfunction1():
    assert mymodule1.myfunction1() == "myfunction1"

def test_mymodule2_myfunction2():
    assert mymodule2.myfunction2() == "myfunction2"
