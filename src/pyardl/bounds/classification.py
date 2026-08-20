r"""The three-test framework of Sam, McNown & Goh (2019).

Rejecting the overall F test is not enough to claim a long-run
relationship, and the reason is worth stating plainly. That test asks
whether *all* the level terms are jointly zero. It rejects if any of
them is not — including in two cases where there is no cointegration at
all:

**Degeneracy of type 1** — :math:`\lambda \ne 0` but
:math:`\gamma = 0`. The dependent variable pulls back towards its own
past, and the regressors carry nothing. What looks like error correction
is ``y`` correcting towards a constant.

**Degeneracy of type 2** — :math:`\gamma \ne 0` but
:math:`\lambda = 0`. The levels of the regressors are jointly
significant, but ``y`` shows no pull back to equilibrium. There is no
mechanism restoring the relationship, so nothing holds it together.

Telling those apart from genuine cointegration takes a third test on the
regressors' levels alone, alongside the two the library already ran:

===============  ==============================  =======================
Test             Null                            Rejection required
===============  ==============================  =======================
``F_overall``    :math:`\lambda = \gamma = 0`     yes
``t_BDM``        :math:`\lambda = 0`             yes, left-tailed
``F_indep``      :math:`\gamma = 0`              yes
===============  ==============================  =======================

Cointegration is established **only** when all three reject. Every other
combination has a name, and this module gives it one rather than leaving
the user to read three numbers and guess.

References
----------
.. [1] Sam, C. Y., McNown, R. & Goh, S. K. (2019). An augmented
       autoregressive distributed lag bounds test for cointegration.
       *Economic Modelling*, 80, 130-141.
.. [2] McNown, R., Sam, C. Y. & Goh, S. K. (2018). Bootstrapping the
       autoregressive distributed lag test for cointegration.
       *Applied Economics*, 50(13), 1509-1521.
"""

from __future__ import annotations

from typing import Literal

__all__ = ["Classification", "classify", "CLASSIFICATIONS"]

Classification = Literal[
    "cointegration",
    "degenerate_1",
    "degenerate_2",
    "no_cointegration",
    "inconclusive",
]

#: Every value :func:`classify` can return, with a one-line reading.
CLASSIFICATIONS: dict[str, str] = {
    "cointegration": "all three tests reject: a genuine long-run relationship",
    "degenerate_1": (
        "y adjusts towards its own past, the regressors carry no long-run relationship"
    ),
    "degenerate_2": (
        "the regressors' levels matter but nothing pulls y back: no "
        "equilibrium is restored"
    ),
    "no_cointegration": "no level relationship",
    "inconclusive": "the three tests do not settle the question",
}

_Decision = str | None


def classify(
    decision_f: _Decision,
    decision_t: _Decision,
    decision_indep: _Decision,
) -> tuple[Classification, str]:
    r"""Combine the three verdicts into one classification and its reason.

    Parameters
    ----------
    decision_f, decision_t, decision_indep : str or None
        Outcome of each test: ``'cointegration'``,
        ``'no_cointegration'`` or ``'inconclusive'``. ``None`` means the
        test could not be run — which happens to ``t_BDM`` under the
        deterministic cases 2 and 4, where no bounds were tabulated for
        it.

    Returns
    -------
    classification : str
        One of the keys of :data:`CLASSIFICATIONS`.
    reason : str
        Which test decided, in one sentence. The classification alone
        says *what*; this says *why*, which is what a reader needs to
        judge whether to believe it.

    Notes
    -----
    The mapping is total: every combination of three three-state verdicts
    lands on a named outcome, and none falls through to a default. A
    silent ``else`` here would be the most dangerous line in the library
    — it would return a verdict nobody chose.

    Examples
    --------
    >>> from pyardl.bounds.classification import classify
    >>> classify("cointegration", "cointegration", "cointegration")[0]
    'cointegration'
    >>> classify("cointegration", "cointegration", "no_cointegration")[0]
    'degenerate_1'
    >>> classify("cointegration", "no_cointegration", "cointegration")[0]
    'degenerate_2'
    """
    if decision_t is None or decision_indep is None:
        missing = "t_BDM" if decision_t is None else "F_indep"
        return (
            "inconclusive",
            f"{missing} is unavailable for this configuration, so the "
            "degeneracies cannot be ruled out; the overall F alone cannot "
            "establish cointegration.",
        )

    undecided = [
        name
        for name, value in (
            ("F_overall", decision_f),
            ("t_BDM", decision_t),
            ("F_indep", decision_indep),
        )
        if value == "inconclusive"
    ]
    if undecided:
        return (
            "inconclusive",
            f"{', '.join(undecided)} fell between the bounds. Bootstrap "
            "critical values remove the inconclusive zone; see "
            "pyardl.bootstrap.",
        )

    f_rej = decision_f == "cointegration"
    t_rej = decision_t == "cointegration"
    i_rej = decision_indep == "cointegration"

    if f_rej and t_rej and i_rej:
        return (
            "cointegration",
            "F_overall, t_BDM and F_indep all reject: the level terms are "
            "jointly significant, y adjusts back towards equilibrium, and "
            "the regressors carry the long-run relationship.",
        )
    if f_rej and t_rej and not i_rej:
        return (
            "degenerate_1",
            "F_overall and t_BDM reject but F_indep does not: y adjusts "
            "towards its own past while the regressors' levels carry "
            "nothing. This is a type 1 degeneracy, not cointegration.",
        )
    if f_rej and i_rej and not t_rej:
        return (
            "degenerate_2",
            "F_overall and F_indep reject but t_BDM does not: the "
            "regressors' levels are jointly significant, yet nothing pulls "
            "y back towards equilibrium. This is a type 2 degeneracy, not "
            "cointegration.",
        )
    if f_rej and not t_rej and not i_rej:
        return (
            "inconclusive",
            "F_overall rejects but neither t_BDM nor F_indep does. The "
            "joint test is driven by something neither component test can "
            "attribute; the evidence does not support any conclusion.",
        )
    if not f_rej and not t_rej and not i_rej:
        return (
            "no_cointegration",
            "F_overall, t_BDM and F_indep all fail to reject: there is no "
            "level relationship.",
        )
    # F_overall does not reject while a component test does. The joint
    # restriction cannot hold when one of its parts is refused, so the
    # three results contradict one another.
    rejecting = [name for name, rej in (("t_BDM", t_rej), ("F_indep", i_rej)) if rej]
    return (
        "inconclusive",
        f"F_overall does not reject, yet {' and '.join(rejecting)} "
        "does. These verdicts contradict one another — usually a sign of "
        "low power, a misspecified lag order, or a sample too short for "
        "the three tests to agree.",
    )
