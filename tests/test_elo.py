#каждая функция test_... утверждает что-то, что должно быть верно. Они же проверяют ваши ручные числа: 1510, 1507,2, сумма не меняется.
from src.elo import elo_update


def test_equal_ratings_no_home_advantage():
    h, a = elo_update(1500, 1500, 'H', home_adv=0)
    assert abs(h - 1510) < 1e-9 and abs(a - 1490) < 1e-9


def test_home_win_with_advantage():
    h, a = elo_update(1500, 1500, 'H')
    assert abs(h - 1507.2) < 0.01


def test_draw_with_advantage_lowers_home():
    h, a = elo_update(1500, 1500, 'D')
    assert h < 1500 < a


def test_sum_is_constant():
    h, a = elo_update(1432, 1587, 'A')
    assert abs((h + a) - (1432 + 1587)) < 1e-9