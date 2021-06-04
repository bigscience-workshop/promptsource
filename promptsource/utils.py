# coding=utf-8

import datasets
import requests


# Hard-coded additional English datasets
# These datasets have either metadata missing or language informations missing
# so we manually identified them.
_ADDITIONAL_ENGLISH_DATSETS = [
    "aeslc",
    "ai2_arc",
    "amazon_us_reviews",
    "anli",
    "art",
    "aslg_pc12",
    "asnq",
    "biomrc",
    "blended_skill_talk",
    "blimp",
    "blog_authorship_corpus",
    "bookcorpus",
    "bookcorpusopen",
    "boolq",
    "break_data",
    "cfq",
    "civil_comments",
    "com_qa",
    "common_gen",
    "commonsense_qa",
    "conll2000",
    "coqa",
    "cornell_movie_dialog",
    "cos_e",
    "cosmos_qa",
    "crd3",
    "crime_and_punish",
    "daily_dialog",
    "definite_pronoun_resolution",
    "discofuse",
    "docred",
    "doqa",
    "drop",
    "emo",
    "emotion",
    "empathetic_dialogues",
    "eraser_multi_rc",
    "esnli",
    "event2Mind",
    "fever",
    "gap",
    "gigaword",
    "guardian_authorship",
    "hans",
    "hellaswag",
    "hotpot_qa",
    "hyperpartisan_news_detection",
    "imdb",
    "jeopardy",
    "lc_quad",
    "math_dataset",
    "math_qa",
    "mlqa",
    "movie_rationales",
    "ms_marco",
    "multi_news",
    "mwsc",
    "natural_questions",
    "newsgroup",
    "newsroom",
    "openbookqa",
    "openwebtext",
    "opinosis",
    "pg19",
    "qa_zre",
    "qangaroo",
    "qanta",
    "qasc",
    "quail",
    "quarel",
    "quartz",
    "quora",
    "quoref",
    "race",
    "reddit",
    "reddit_tifu",
    "reuters21578",
    "rotten_tomatoes",
    "scan",
    "scicite",
    "scientific_papers",
    "scifact",
    "sciq",
    "scitail",
    "search_qa",
    "sem_eval_2010_task_8",
    "sentiment140",
    "social_i_qa",
    "squad_v2",
    "squadshifts",
    "super_glue",
    "trec",
    "trivia_qa",
    "tydiqa",
    "ubuntu_dialogs_corpus",
    "web_nlg",
    "web_of_science",
    "web_questions",
    "wiki40b",
    "wiki_qa",
    "wiki_snippets",
    "wiki_split",
    "wikipedia",
    "wikisql",
    "wikitext",
    "winogrande",
    "wiqa",
    "wnut_17",
    "xnli",
    "xquad",
    "xsum",
    "xtreme",
    "yelp_polarity",
]


def removeHyphen(example):
    example_clean = {}
    for key in example.keys():
        if "-" in key:
            new_key = key.replace("-", "_")
            example_clean[new_key] = example[key]
        else:
            example_clean[key] = example[key]
    example = example_clean
    return example


def renameDatasetColumn(dataset):
    col_names = dataset.column_names
    for cols in col_names:
        if "-" in cols:
            dataset = dataset.rename_column(cols, cols.replace("-", "_"))
    return dataset


#
# Helper functions for datasets library
#

def get_dataset_builder(path, conf=None):
    "Get a dataset builder from name and conf."
    module_path = datasets.load.prepare_module(path, dataset=True)
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    if conf:
        builder_instance = builder_cls(name=conf, cache_dir=None)
    else:
        builder_instance = builder_cls(cache_dir=None)
    return builder_instance


def get_dataset(path, conf=None):
    "Get a dataset from name and conf."
    builder_instance = get_dataset_builder(path, conf)
    fail = False
    if builder_instance.manual_download_instructions is None and builder_instance.info.size_in_bytes is not None:
        builder_instance.download_and_prepare()
        dts = builder_instance.as_dataset()
        dataset = dts
    else:
        dataset = builder_instance
        fail = True
    return dataset, fail


def get_dataset_confs(path):
    "Get the list of confs for a dataset."
    module_path = datasets.load.prepare_module(path, dataset=True)
    # Get dataset builder class from the processing script
    builder_cls = datasets.load.import_main_class(module_path[0], dataset=True)
    # Instantiate the dataset builder
    confs = builder_cls.BUILDER_CONFIGS
    if confs and len(confs) > 1:
        return confs
    return []


def render_features(features):
    """Recursively render the dataset schema (i.e. the fields)."""
    if isinstance(features, dict):
        return {k: render_features(v) for k, v in features.items()}
    if isinstance(features, datasets.features.ClassLabel):
        return features.names

    if isinstance(features, datasets.features.Value):
        return features.dtype

    if isinstance(features, datasets.features.Sequence):
        return {"[]": render_features(features.feature)}
    return features


#
# Loads dataset information
#


def filter_english_datasets():
    """Filter English datasets based on language tags in metadata"""
    english_datasets = []

    response = requests.get("https://huggingface.co/api/datasets?full=true")
    tags = response.json()

    for dataset in tags:
        dataset_name = dataset["id"]

        is_community_dataset = "/" in dataset_name
        if is_community_dataset:
            continue

        if "card_data" not in dataset:
            continue
        metadata = dataset["card_data"]

        if "languages" not in metadata:
            continue
        languages = metadata["languages"]

        if "en" in languages:
            english_datasets.append(dataset_name)

    all_english_datasets = list(set(english_datasets + _ADDITIONAL_ENGLISH_DATSETS))
    return sorted(all_english_datasets)


def list_datasets(template_collection, _priority_filter, _priority_max_templates, _state):
    """Get all the datasets to work with."""
    dataset_list = filter_english_datasets()
    count_dict = template_collection.get_templates_count()
    if _priority_filter:
        dataset_list = list(
            set(dataset_list)
            - set(
                list(
                    d
                    for d in count_dict
                    if count_dict[d] > _priority_max_templates and d != _state.working_priority_ds
                )
            )
        )
        dataset_list.sort()
    else:
        dataset_list.sort(key=lambda x: DATASET_ORDER.get(x, 1000))
    return dataset_list


DATASET_ORDER = dict(
    [
        ("glue", 0),
        ("squad", 1),
        ("bookcorpusopen", 2),
        ("wikipedia", 3),
        ("wikitext", 4),
        ("imdb", 5),
        ("super_glue", 6),
        ("cnn_dailymail", 7),
        ("openwebtext", 8),
        ("common_voice", 9),
        ("xsum", 10),
        ("wmt16", 11),
        ("conll2003", 12),
        ("ag_news", 13),
        ("universal_dependencies", 14),
        ("wiki_qa", 15),
        ("bookcorpus", 16),
        ("wiki40b", 17),
        ("wiki_dpr", 18),
        ("xnli", 19),
        ("squad_kor_v1", 20),
        ("emotion", 21),
        ("wikiann", 22),
        ("amazon_us_reviews", 23),
        ("squad_v2", 24),
        ("amazon_reviews_multi", 25),
        ("librispeech_asr", 26),
        ("blimp", 27),
        ("scitail", 28),
        ("anli", 29),
        ("samsum", 30),
        ("lambada", 31),
        ("multi_nli", 32),
        ("daily_dialog", 33),
        ("snli", 34),
        ("opus_euconst", 35),
        ("rotten_tomatoes", 36),
        ("scientific_papers", 37),
        ("trec", 38),
        ("reddit_tifu", 39),
        ("ai2_arc", 40),
        ("patrickvonplaten", 41),
        ("gigaword", 42),
        ("swag", 43),
        ("timit_asr", 44),
        ("oscar", 45),
        ("tweet_eval", 46),
        ("newsgroup", 47),
        ("billsum", 48),
        ("gem", 49),
        ("blended_skill_talk", 50),
        ("eli5", 51),
        ("ade_corpus_v2", 52),
        ("race", 53),
        ("wikihow", 54),
        ("piqa", 55),
        ("xtreme", 56),
        ("commonsense_qa", 57),
        ("wiki_snippets", 58),
        ("mlsum", 59),
        ("multi_news", 60),
        ("wmt14", 61),
        ("asnq", 62),
        ("toriving", 63),
        ("crime_and_punish", 64),
        ("few_rel", 65),
        ("code_search_net", 66),
        ("universal_morphologies", 67),
        ("ms_marco", 68),
        ("trivia_qa", 69),
        ("lama", 70),
        ("newsroom", 71),
        ("hellaswag", 72),
        ("adversarial_qa", 73),
        ("hatexplain", 74),
        ("hans", 75),
        ("kilt_tasks", 76),
        ("xglue", 77),
        ("amazon_polarity", 78),
        ("meta_woz", 79),
        ("opus_books", 80),
        ("wmt18", 81),
        ("covid_qa_deepset", 82),
        ("emotion\\dataset_infos.json", 83),
        ("wmt19", 84),
        ("discofuse", 85),
        ("mrqa", 86),
        ("winogrande", 87),
        ("go_emotions", 88),
        ("tydiqa", 89),
        ("yelp_polarity", 90),
        ("banking77", 91),
        ("math_dataset", 92),
        ("pubmed_qa", 93),
        ("opus_ubuntu", 94),
        ("acronym_identification", 95),
        ("math_qa", 96),
        ("babi_qa", 97),
        ("dbpedia_14", 98),
        ("ted_multi", 99),
        ("allocine", 100),
        ("hotpot_qa", 101),
        ("cc_news", 102),
        ("conll2002", 103),
        ("cuad", 104),
        ("mc_taco", 105),
        ("silicone", 106),
        ("discovery", 107),
        ("mt_eng_vietnamese", 108),
        ("quac", 109),
        ("conllpp", 110),
        ("ubuntu_dialogs_corpus", 111),
        ("esnli", 112),
        ("doc2dial", 113),
        ("squad_kor_v2", 114),
        ("opus_gnome", 115),
        ("german_legal_entity_recognition", 116),
        ("openbookqa", 117),
        ("tapaco", 118),
        ("xquad_r", 119),
        ("imdb\\dataset_infos.json", 120),
        ("opus_wikipedia", 121),
        ("amr", 122),
        ("wnut_17", 123),
        ("empathetic_dialogues", 124),
        ("cbt", 125),
        ("opus_rf", 126),
        ("narrativeqa", 127),
        ("mnist", 128),
        ("sick", 129),
        ("swda", 130),
        ("aeslc", 131),
        ("art", 132),
        ("coqa", 133),
        ("opus100", 134),
        ("sst", 135),
        ("big_patent", 136),
        ("germeval_14", 137),
        ("liar", 138),
        ("un_pc", 139),
        ("alt", 140),
        ("circa", 141),
        ("scan", 142),
        ("wikisql", 143),
        ("reddit", 144),
        ("wino_bias", 145),
        ("financial_phrasebank", 146),
        ("social_i_qa", 147),
        ("newsqa", 148),
        ("cosmos_qa", 149),
        ("classla", 150),
        ("scicite", 151),
        ("codah", 152),
        ("ehealth_kd", 153),
        ("wikicorpus", 154),
        ("ccaligned_multilingual", 155),
        ("cos_e", 156),
        ("thaisum", 157),
        ("cfq", 158),
        ("yahoo_answers_topics", 159),
        ("wmt", 160),
        ("natural_questions", 161),
        ("cc100", 162),
        ("paws", 163),
        ("boolq", 164),
        ("break_data", 165),
        ("pragmeval", 166),
        ("arabic_speech_corpus", 167),
        ("text\\dataset_infos.json", 168),
        ("md_gender_bias", 169),
        ("mlqa", 170),
        ("arabic_billion_words", 171),
        ("dialog_re", 172),
        ("tweets_hate_speech_detection", 173),
        ("ecthr_cases", 174),
        ("json\\dataset_infos.json", 175),
        ("conv_ai_2", 176),
        ("dream", 177),
        ("kor_ner", 178),
        ("youtube_caption_corrections", 179),
        ("spider", 180),
        ("air_dialogue", 181),
        ("arxiv_dataset", 182),
        ("data", 183),
        ("quora", 184),
        ("docred", 185),
        ("guardian_authorship", 186),
        ("quartz", 187),
        ("yelp_review_full", 188),
        ("xquad", 189),
        ("ted_talks_iwslt", 190),
        ("orange_sum", 191),
        ("indonlu", 192),
        ("tweet_qa", 193),
        ("multi_woz_v22", 194),
        ("s2orc", 195),
        ("clarin-pl", 196),
        ("cord19", 197),
        ("emo", 198),
        ("indic_glue", 199),
        ("ethos", 200),
        ("persiannlp", 201),
        ("totto", 202),
        ("wongnai_reviews", 203),
        ("bavard", 204),
        ("europa_ecdc_tm", 205),
        ("google_wellformed_query", 206),
        ("paws-x", 207),
        ("emea", 208),
        ("fever", 209),
        ("asset", 210),
        ("kilt_wikipedia", 211),
        ("clinc_oos", 212),
        ("conv_ai_3", 213),
        ("ncbi_disease", 214),
        ("sentiment140", 215),
        ("quarel", 216),
        ("txt", 217),
        ("ajgt_twitter_ar", 218),
        ("ambig_qa", 219),
        ("ptb_text_only", 220),
        ("stsb_multi_mt", 221),
        ("web_questions", 222),
        ("winograd_wsc", 223),
        ("eurlex", 224),
        ("muchocine", 225),
        ("app_reviews", 226),
        ("aqua_rat", 227),
        ("bible_para", 228),
        ("wiki_auto", 229),
        ("cifar10", 230),
        ("eli5\\dataset_infos.json", 231),
        ("quail", 232),
        ("hyperpartisan_news_detection", 233),
    ]
)
