import math


def H(n, p):
    return -(n/(n+p))*math.log(n/(n+p), 2) - (p/(n+p))*math.log(p/(n+p), 2)


def IV(lst_ratios):
    iv = 0
    for ratio in lst_ratios:
        iv -= ratio*math.log(ratio, 2)
    return iv

def reminder(n, p, Ni_Pi):
    #Ni_Pi = list of tuples (n_i, p_i)
    rem_sum = 0
    for ni_pi in Ni_Pi:
        ni = ni_pi[0]
        pi = ni_pi[1]
        rem_sum += ((pi + ni)/(n + p))*H(ni/(ni+pi), pi/(ni+pi))
    return rem_sum

def IG()