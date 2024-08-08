"""Contains the different decay functions"""

from .transactions import SimpleTransaction, Transaction, DecayedTransaction
from typing import Any, List, Dict
from copy import deepcopy
from numpy import exp

# Fuzzy requirements :
# Separate the local trust from the decay transactions containing the crisp values.
# Only return the decayed transactions.

# TODO control wether reputation modify transaction.
#   -> Voir ou se balade la fonction decayed

# TODO control wether the direct opinion function or indirect opinion function
# should be used for the double and adaptive windows.


def exponential_decay(
    transactions: List[Transaction],
    lmbda: float,
    reputation: callable,
    reputation_function_parameter: List[Any],
    **kwargs
) -> List[Transaction]:
    """Decay a list of transaction following exponential decay

    Args:
        transactions (List[Transaction]): List of transaction that should be decayed
        lmbda (float, optional): Exponential factor for decay, 0 mean only last interaction counts, 1 that all interactions are equal. Defaults to 0.5.
        reputation (callable): reputation function that take List[Transaction] as an input.
        reputation_function_parameter (Dict[str,Any]): parameters to pass to the reputation function.
    Returns:
        List[Transaction]: Transaction with decayed weight
    """
    # Decay as exposed in the Beta reputation system implementation.
    size = len(transactions)
    # Broken if lambda is set at 0 which SHOULD NOT happen,
    for i, t in enumerate(transactions):
        t.weight = lmbda ** (size - i + 1)
    return transactions

    # Old non fuzzy behavior.
    # return reputation(*reputation_function_parameter,transactions)


def _simple_window(
    transactions: List[Transaction], hist_size: float, **kwargs
) -> List[Transaction]:
    """Decay with simple window
    Args:
        transactions (List[Transaction]): List of transaction to apply decay upon.
        hist_size (float): size of the simple window
        reputation (callable): reputation function that take List[Transaction] as an input.
        reputation_function_parameter (Dict[str,Any]): parameters to pass to the reputation function.

    Returns:
        List[Transaction]: List of Transaction that are in the observed window.
    """
    if not transactions:
        return []
    # Necessary if we want to take the latest n transition and not
    key_func = lambda Transaction: Transaction.timestamp
    # Not necessary if we can pass the time stamp to this function.
    max_t = max(transactions, key=key_func).timestamp
    for i, t in enumerate(transactions):
        if t.timestamp < max_t - hist_size:
            continue
        else:
            return transactions[i:]
    return transactions


def simple_window(
    transactions: List[Transaction],
    hist_size: float,
    reputation: callable,
    reputation_function_parameter: List[Any],
    **kwargs
) -> List[Transaction]:
    """Decay with simple window
    Args:
        transactions (List[Transaction]): List of transaction to apply decay upon.
        hist_size (float): size of the simple window
        reputation (callable): reputation function that take List[Transaction] as an input.
        reputation_function_parameter (Dict[str,Any]): parameters to pass to the reputation function.

    Returns:
        List[Transaction]: List of Transaction that are in the observed window.
    """
    return reputation(
        *reputation_function_parameter, _simple_window(transactions, hist_size)
    )


def double_window(
    transactions: List[Transaction],
    base_win_size: float,
    small_win_size: float,
    threshold: float,
    reputation: callable,
    reputation_function_parameter: List[Any],
    **kwargs
) -> float:
    """Dual windows as defined in Peertrust.

    Args:
        transactions (List[Transaction]): Past Transactions from the peer
        win_size (float): Size of the normal window
        wins_size (float): Size of the small window
        threshold (float): Threshold use to switch to the smaller windows
        reputation (callable): reputation function that take List[Transaction] as an input.
        reputation_function_parameter (Dict[str,Any]): parameters to pass to the reputation function.

    Returns:
        float: reputation of the peer
    """
    win: List[Transaction] = _simple_window(transactions, base_win_size)
    wins: List[Transaction] = _simple_window(transactions, small_win_size)
    rwin: float = reputation(*reputation_function_parameter, win)
    rwins: float = reputation(*reputation_function_parameter, wins)

    # TODO absolute(rwin-rwins) ?
    # içi aussi c'est faux non ?
    return wins if rwin - rwins > threshold else win

    # old non fuzzy behavior.
    # return rwins if rwin - rwins > threshold else rwin


def adaptive_window(
    transactions: List[Transaction],
    base_win_size: float,
    small_win_size: float,
    reputation: callable,
    reputation_function_parameter: List[Any],
    timestamp: float,
    **kwargs
) -> float:
    """Dual windows evolution proposed to better recover from outage.

    Args:
        transactions (List[Transaction]): Past Transactions from the peer
        base_win_size (float): Size of the normal window
        small_win_size (float): Size of the small window
        reputation (callable): reputation function that take List[Transaction] as an input.
        reputation_function_parameter (Dict[str,Any]): parameters to pass to the reputation function.
        timestamp (float): Current time in the simulation
    Returns:
        float: reputation of the peer
    """
    # TODO ajouter le timestamp pour tout le monde.
    # Changer le fontionnement des fenêtres pour avoir un nombre de transactions ?
    # Mettre une petite fenêtre vraiment petite (0.05 ~=15 interactions)
    # Regarder le  nombre d'interactions totales + pannes.
    win: List[Transaction] = deepcopy(_simple_window(transactions, base_win_size))
    wins: List[Transaction] = deepcopy(_simple_window(transactions, small_win_size))
    # Fuzzy : change the reputation function to
    rwin: float = reputation(*reputation_function_parameter, win)
    rwins: float = reputation(*reputation_function_parameter, wins)
    if transactions:
        t_last_transaction: float = transactions[len(transactions) - 1].timestamp

    # alpha: float = rwin - rwins
    if rwin:
        alpha: float = (rwin - rwins) / rwin
    else:
        alpha = 1
    # only take into account the smaller windows when it have a worst
    # reputation than the good windows
    if alpha <= 0:
        return win

    # beta: float = timestamp - t_last_transaction
    beta: float = 1 - exp(-(timestamp - t_last_transaction))
    omega: float

    # When smaller windows have a worst reputation we take a combination of both.
    # Pourquoi beta / alpha <= 1 ? Il y a un certain nombre de cas ou il sera > 1, que se passe-t-il dans ces cas.
    # Permet de contraindre omega entre 0 et 1
    omega = alpha * (1 - beta)

    # Ponderate the transactions from each win according to the adaptive scheme
    win_len = len(win)
    for t in win:
        t.weight = (1 - omega) / win_len
    wins_len = len(wins)
    for t in wins:
        t.weight = omega / wins_len

    return win + wins

    # Older non Fuzzy version
    # return omega*rwins + (1-omega)*rwin
