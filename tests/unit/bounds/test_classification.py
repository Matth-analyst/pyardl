"""Spec 15 §2.3 — table de décision du cadre à trois tests.

Le verrou de ce module : la classification doit être TOTALE. Trois tests
à trois états, plus l'absence possible du t sous les cas II et IV, font
plus de combinaisons qu'on n'en énumère spontanément. Une seule tombant
dans un `else` silencieux rendrait un verdict que personne n'a choisi.
"""

from __future__ import annotations

import itertools

import pytest

from pyardl.bounds.classification import CLASSIFICATIONS, classify

_STATES = ("cointegration", "no_cointegration", "inconclusive")


class TestTotality:
    """La table couvre tout, sans branche par défaut."""

    def test_every_combination_is_named(self) -> None:
        """Les 3^3 combinaisons + celles où le t manque."""
        seen = set()
        for f, t, i in itertools.product(_STATES, repeat=3):
            label, reason = classify(f, t, i)
            assert label in CLASSIFICATIONS, (f, t, i)
            # Une justification se lit : elle est une phrase, et elle
            # nomme le ou les tests qui ont tranché.
            assert reason.endswith("."), (f, t, i)
            assert any(name in reason for name in ("F_overall", "t_BDM", "F_indep")), (
                f,
                t,
                i,
            )
            seen.add(label)
        # Les quatre verdicts substantiels doivent être atteignables.
        assert {
            "cointegration",
            "degenerate_1",
            "degenerate_2",
            "no_cointegration",
            "inconclusive",
        } <= seen

    @pytest.mark.parametrize("missing", ["t", "indep"])
    def test_missing_test_is_inconclusive_not_silent(self, missing: str) -> None:
        """Un test absent ne doit jamais laisser conclure.

        Sous les cas II et IV, PSS ne tabule pas le t. Sans lui, les
        dégénérescences ne peuvent pas être écartées — et le F global
        seul ne suffit pas, c'est tout le propos de la spec.
        """
        args = ["cointegration", "cointegration", "cointegration"]
        args[1 if missing == "t" else 2] = None  # type: ignore[call-overload]
        label, reason = classify(*args)  # type: ignore[arg-type]
        assert label == "inconclusive"
        assert "unavailable" in reason

    def test_reason_names_the_deciding_test(self) -> None:
        """La justification cite le test qui a tranché."""
        for f, t, i in itertools.product(_STATES, repeat=3):
            _, reason = classify(f, t, i)
            assert any(name in reason for name in ("F_overall", "t_BDM", "F_indep")), (
                f,
                t,
                i,
            )


class TestCanonicalVerdicts:
    """§1 — les quatre situations que l'article nomme."""

    def test_all_three_reject_is_cointegration(self) -> None:
        label, reason = classify("cointegration", "cointegration", "cointegration")
        assert label == "cointegration"
        assert "all reject" in reason

    def test_type_1_degeneracy(self) -> None:
        """lambda != 0, gamma = 0 : y se corrige vers son propre passé."""
        label, reason = classify("cointegration", "cointegration", "no_cointegration")
        assert label == "degenerate_1"
        assert "type 1" in reason
        assert "F_indep does not" in reason

    def test_type_2_degeneracy(self) -> None:
        """gamma != 0, lambda = 0 : aucune force de rappel."""
        label, reason = classify("cointegration", "no_cointegration", "cointegration")
        assert label == "degenerate_2"
        assert "type 2" in reason
        assert "t_BDM does not" in reason

    def test_nothing_rejects_is_no_cointegration(self) -> None:
        label, _ = classify("no_cointegration", "no_cointegration", "no_cointegration")
        assert label == "no_cointegration"

    def test_overall_alone_is_reinforced_inconclusive(self) -> None:
        """§1 — « F_overall seul -> non concluant renforcé ».

        Le F global rejette sans qu'aucun de ses composants ne le fasse :
        le rejet n'est attribuable à rien.
        """
        label, reason = classify(
            "cointegration", "no_cointegration", "no_cointegration"
        )
        assert label == "inconclusive"
        assert "neither" in reason


class TestContradictions:
    """Les combinaisons que l'article ne nomme pas, et qui existent."""

    def test_component_rejects_but_joint_does_not(self) -> None:
        """Contradiction logique : signalée, jamais arbitrée en silence.

        Si la restriction jointe tient, aucune de ses parties ne peut
        être rejetée. Quand cela arrive quand même, c'est un symptôme —
        puissance faible, ordre de retards inadapté, échantillon court —
        et l'utilisateur doit le lire, pas hériter d'un verdict.
        """
        for t, i in (
            ("cointegration", "no_cointegration"),
            ("no_cointegration", "cointegration"),
            ("cointegration", "cointegration"),
        ):
            label, reason = classify("no_cointegration", t, i)
            assert label == "inconclusive"
            assert "contradict" in reason

    def test_inconclusive_component_names_itself(self) -> None:
        label, reason = classify("inconclusive", "cointegration", "cointegration")
        assert label == "inconclusive"
        assert "F_overall" in reason
        assert "bootstrap" in reason.lower()

    def test_several_inconclusive_all_named(self) -> None:
        _, reason = classify("inconclusive", "inconclusive", "cointegration")
        assert "F_overall" in reason
        assert "t_BDM" in reason


class TestNoSilentDefault:
    """Aucune combinaison ne doit produire un verdict non justifié."""

    def test_no_empty_reason(self) -> None:
        for f, t, i in itertools.product((*_STATES, None), repeat=3):
            label, reason = classify(f, t, i)
            assert len(reason) > 30, (f, t, i, reason)
            assert label in CLASSIFICATIONS

    def test_cointegration_requires_exactly_three_rejections(self) -> None:
        """Le verdict positif n'est atteint que par un seul chemin."""
        positives = [
            (f, t, i)
            for f, t, i in itertools.product(_STATES, repeat=3)
            if classify(f, t, i)[0] == "cointegration"
        ]
        assert positives == [("cointegration", "cointegration", "cointegration")]
