"""
#############################################################

YOUR CODE CANNOT BE GRADED IF:
1) ...you change function or class names in this file.
2) ...you write code outside of the scope of the functions given below.
3) ...your file cannot run because it contains syntax errors.

#############################################################
"""


# You may adjust this number to any positive integer for testing purposes



def question_ba(testnum):
    """
    =QUESTION=
    The relative change from some number A to some number B can be obtained by
    taking B, then subtracting A, and then dividing the result by A.   
    Write code to return the relative change from testnum (A) to
    testnum squared (B).
    """
    A = testnum
    B = testnum**2
    Rel = B-A
    Rel_change = Rel/A
    return Rel_change


def question_bb(testnum):
    """
    =QUESTION=
    Write code to return a list with the numbers on the range of 0 up to
    and including testnum, in steps of 100. The numbers must be ordered
    from smallest to largest.
    """
    my_list = range(0,testnum+1,100)
    return my_list


def question_bc(num):
    """
    =QUESTION=
    Write code that returns the number of times that testnum can be divided
    by 2, without that division resulting in a decimal number. 
    Example: if testnum is 1000, then the returned value should be 3,
    because 1000 can be divided by 2 three times (yielding 500, 250, 125)
    before resulting in a decimal number.
    """
    # use for if you know in advace how many iterations are needed
    # use while if you/re not sure how many iterations are needed
    my_number = []
    print(isinstance(num, int))
    while isinstance(num,int) is True:
        my_result = num/2
        my_number.append(my_result)
    print(my_number)
    return my_number



def question_bd(testnum):
    """
    =QUESTION=
    Write code that returns a list, that contains a single tuple,
    that in turn contains testnum as a string. 
    That is: a string inside a tuple inside a list.
    """
    my_list = (tuple(str(testnum)))
    return my_list


def question_be(testnum):
    """
    =QUESTION=
    A prime number can only be divided by 1 and by itself to yield an integer.
    A division by any other number results in a decimal number.
    Write code to return a list of integers that contains all prime numbers
    on the range from 2 up to and including testnum.
    """

    my_list = []
    for element in range(2,int(testnum)+1):
       if ( testnum % element) == 0:
           my_list.append(testnum)

    return



#class Counter(object):
#    """
#    Do NOT adjust the class definition or the __init__ function.
#    """
#
#    def __init__(self, step_size=1):
#        self.count = 0
#        self.step_size = step_size
#        return
#
#
#    def increment(self, counter):
#        """
#        =INPUT=
#            counter - instance of the Counter class
#        =OUTPUT=
#            Returns nothing
#        
#        =QUESTION=
#        Write code in this function that realizes the following behavior:
#    
#        If the count of the input counter is higher than the internal count,
#        set the internal count equal to the count of the input counter.
#        In all other cases, increase the internal count by step_size.
#        """
#        
#        return



testnum = 1000
a = question_bc(testnum)
print(a)