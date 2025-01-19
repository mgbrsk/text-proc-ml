import math


def get_zipf_proba(
    rank: int,
    s: int,
    N: int,
) -> float:
    """f(rank;s,N) = 1/(Z(s,N)*rank^s)
    rank - порядковый номер слова после сортировки по убыванию частоты
    s - коэффициент скорости убывания вероятности

    Z(s,N) = sum(i=1->N)(i^-s) - нормализационная константа.
    Args:
        rank - порядковый номер слова после сортировки по убыванию частоты
        s - коэффициент скорости убывания вероятности
        N - количество слов
    """
    zipf_proba = 1 / ((rank**s) * sum([i ** (-s) for i in [*range(1, N + 1)]]))
    return round(zipf_proba, 4)


def get_cross_entropy_for_two_examples(y_predict_list: list[float]) -> float:
    """Кросс-энтропия для двух примеров y_true=1,1 y'=y_predict=[y'1,y'2]

    Returns:
        _type_: _description_
    """
    cross_entropy = -math.log(y_predict_list[0]) - math.log(y_predict_list[1])
    return cross_entropy
