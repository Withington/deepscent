import hello_add

def func(x):
    return x + 1

def test_answer():
    assert func(3) == 4
    
def test_add_two():
    assert hello_add.add_two(3) == 5
