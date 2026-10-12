"""The dialects of the Arabic model and their tag tokens.

A request's text starts with its dialect's tag: one of the tokenizer's unused <reserved_N> tokens (one id each, an
embedding row trained with the model). Corpus labels map onto the tags; product names that have no data of their own
are aliases (emirati / qatari speak the Gulf tag, jordanian the Palestinian one).
"""

from __future__ import annotations

DIALECTS = ["msa", "egyptian", "palestinian", "lebanese", "syrian", "iraqi", "saudi", "najdi", "hejazi", "gulf",
            "bahraini", "yemeni", "sudanese", "libyan", "tunisian", "algerian", "moroccan"]
FIRST_RESERVED = 100  # <reserved_100> is the first tag

# (corpus, label) -> dialect
CORPUS_LABELS = {
    ("lahgtna", "eg"): "egyptian", ("lahgtna", "ps"): "palestinian", ("lahgtna", "lb"): "lebanese",
    ("lahgtna", "sy"): "syrian", ("lahgtna", "iq"): "iraqi", ("lahgtna", "sa"): "saudi", ("lahgtna", "bh"): "bahraini",
    ("lahgtna", "ye"): "yemeni", ("lahgtna", "sd"): "sudanese", ("lahgtna", "ly"): "libyan",
    ("lahgtna", "tn"): "tunisian", ("lahgtna", "dz"): "algerian", ("lahgtna", "ma"): "moroccan",
    ("sada", "najdi"): "najdi", ("sada", "hejazi"): "hejazi", ("sada", "khaliji"): "gulf", ("sada", "msa"): "msa",
    ("sada", "eg"): "egyptian", ("sada", "ye"): "yemeni", ("sada", "iq"): "iraqi", ("sada", "sa_unknown"): "saudi",
    ("sada", "shamali"): "saudi", ("sada", "janubi"): "saudi",
}
CORPUS_LABELS.update({("synth", d): d for d in DIALECTS})  # synth/filter.py writes canonical names

ALIASES = {"emirati": "gulf", "qatari": "gulf", "khaliji": "gulf", "jordanian": "palestinian", "ab": "palestinian",
           "levantine": "palestinian"}


def tag_token(dialect: str) -> str:
    dialect = ALIASES.get(dialect, dialect)
    return f"<reserved_{FIRST_RESERVED + DIALECTS.index(dialect)}>"


def of_row(row: dict) -> str | None:
    return CORPUS_LABELS.get((row.get("source"), row.get("dialect")))
