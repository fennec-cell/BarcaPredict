def elo_expected(r_home, r_away, home_adv=100):
    return 1 / (1 + 10 ** ((r_away - (r_home + home_adv)) / 400))


def elo_update(r_home, r_away, result, k=20, home_adv=100):
    # result: 'H' победа хозяев, 'D' ничья, 'A' победа гостей
    s = {'H': 1.0, 'D': 0.5, 'A': 0.0}[result]
    delta = k * (s - elo_expected(r_home, r_away, home_adv))
    return r_home + delta, r_away - delta
