import promptsource.augmentation as aug

CASE_AG_NEWS_1 = (
    "{{text}}"
    "\nIs this a piece of news regarding "
    "{{\"world politics\"}}, {{\"sports\"}}, {{\"business\"}}, or {{\"science and technology\"}}? |||"
    "\n{{ [\"World politics\", \"Sports\", \"Business\", \"Science and technology\"][label] }}"
)


def test_back_translation_strategy():
    aug_tmpl = aug.back_translation_strategy(CASE_AG_NEWS_1)
    assert aug_tmpl == CASE_AG_NEWS_1.replace(
        "a piece of news regarding",
        "a new one about"
    ).replace(
        ", or",
        " or"
    )
