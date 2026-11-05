"""Topic masking: hide country / nationality / place / party / politician tokens.

Swedish MEPs talk about Sweden, British and Irish MEPs about the UK and
Ireland, so a classifier can score well just by spotting "Sverige" or
"Storbritannien". `mask(text)` replaces such tokens with PLACEHOLDER so the
harness can report metrics with and without them (`--mask-topic`).

The list is deliberately broad (all EU member states + a few neighbours,
their nationality/language adjectives, capitals and big cities, national
parliaments, parties and well-known national politicians of SE/UK/IE).
Extend it when you find a new leak; keep it symmetric across countries.
"""
import re

PLACEHOLDER = "XTOPIC"

# Stems matched case-insensitively at a word start, followed by any letters
# (so "svensk" covers svenska, svenskt, svenskarna, ...).
STEMS = """
sverige svensk
storbritannien brittisk britt england engelsk engelsm skottland skotsk wales walesisk
nordirland nordirländ irland irländ
danmark dansk danskar finland finsk finländ finnar norge norsk island isländ
tyskland tysk frankrike fransk fransm spanien spansk spanjor portugal portugis
italien italiensk grekland grek nederländ holland holländ belgien belgisk belgare
luxemburg österrik tjeck slovak slovensk slovenien ungern ungersk ungrare estland estnisk
estländ lettland lettisk litau malta maltes cypern cypriot rumän bulgar kroat
schweiz nordisk skandinav östersjö
stockholm göteborg malmö uppsala norrland lappland gotland
london dublin edinburgh belfast cardiff glasgow manchester liverpool birmingham
köpenhamn helsingfors oslo berlin paris madrid lissabon wien amsterdam warszawa
budapest
labour tory tories torypartiet ukip liberaldemokrat
moderaterna folkpartiet centerpartiet vänsterpartiet miljöpartiet kristdemokraterna
junilistan sverigedemokrat piratpartiet
riksdag underhuset överhuset westminster
thatcher blair cameron bildt reinfeldt paisley trimble
kronor kronan pund sterling
snus systembolag allemansrätt
gibraltar falkland
"""

# Exact words (case-sensitive, optional genitive -s): ambiguous lowercase forms.
EXACT = """
Polen Rom Aten Haag Prag Cork Major Brown Persson Carlsson Lindh Ahern Hume SNP
"""

PHRASES = [r"förenade\s+kungariket", r"sinn\s+féin", r"fianna\s+fáil", r"fine\s+gael",
           r"plaid\s+cymru", r"downing\s+street", r"moderata\s+samlingspartiet"]

_stem_re = re.compile(r"(?<![\w-])(?:%s)[\wåäöéü]*" % "|".join(sorted(STEMS.split(), key=len, reverse=True)),
                      re.IGNORECASE)
_exact_re = re.compile(r"(?<![\w-])(?:%s)s?(?![\w])" % "|".join(EXACT.split()))
_phrase_re = re.compile(r"(?<![\w-])(?:%s)[\wåäö]*" % "|".join(PHRASES), re.IGNORECASE)


def mask(text, placeholder=PLACEHOLDER):
    text = _phrase_re.sub(placeholder, text)
    text = _stem_re.sub(placeholder, text)
    text = _exact_re.sub(placeholder, text)
    return text


def count(text):
    return (len(_phrase_re.findall(text)) + len(_stem_re.findall(text))
            + len(_exact_re.findall(text)))
