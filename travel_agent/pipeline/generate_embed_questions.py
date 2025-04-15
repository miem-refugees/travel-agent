import argparse
import string
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml
from loguru import logger
from pydantic import BaseModel


class Params(BaseModel):
    dataset: str
    out_dir: str


class SearchParams(BaseModel):
    question: str
    keywords: Optional[str] = None
    rubrics: Optional[str] = None


keywords_by_phrases = [
    SearchParams(question="Где поесть вкусную пиццу?", keywords="вкусн", rubrics="Пиццерия"),
    SearchParams(
        question="Салоны красоты с хорошими мастерами",
        keywords="спасибо",
        rubrics="Салон красоты",
    ),
    SearchParams(
        question="Недорогие отели в центре города",
        keywords="(недорого|дешево)",
        rubrics="Гостиница",
    ),
    SearchParams(question="Лучшие бары с коктейлями", keywords="коктейли", rubrics="Бар, паб"),
    SearchParams(question="Где быстро купить продукты?", rubrics="Магазин продуктов|Супермаркет"),
    SearchParams(
        question="Доставка суши с хорошими отзывами",
        keywords="доставка",
        rubrics="Суши-бар",
    ),
    SearchParams(question="Рестораны куда можно с детьми", keywords="дети", rubrics="Ресторан"),
    SearchParams(question="Кофейни с десертами", keywords="десерты", rubrics="Кофейня"),
    SearchParams(
        question="Где купить подарки на праздник?",
        keywords="подарки",
        rubrics="Магазин подарков и сувениров",
    ),
    SearchParams(
        question="Массажные салоны с расслабляющей атмосферой",
        keywords="расслаб",
        rubrics="Массажный салон",
    ),
    SearchParams(
        question="Аптеки с круглосуточным режимом",
        keywords="круглосуточн",
        rubrics="Аптека",
    ),
    SearchParams(
        question="Магазины бытовой химии с акциями",
        keywords="акции",
        rubrics="Магазин хозтоваров и бытовой химии",
    ),
    SearchParams(question="Пиццерии с доставкой", keywords="доставк", rubrics="Пиццерия"),
    SearchParams(question="Супермаркеты рядом с домом", rubrics="Супермаркет"),
    SearchParams(
        question="Где вкусно пообедать недорого?",
        keywords="(недорого|дешево)",
        rubrics="Столовая|Быстрое питание",
    ),
    SearchParams(
        question="Стоматологии с хорошими врачами",
        keywords="спасибо",
        rubrics="Стоматологическая клиника",
    ),
    SearchParams(question="Музеи, где интересно детям", keywords="дети", rubrics="Музей"),
    SearchParams(question="Салоны сделать брови", keywords="бров", rubrics="Ногтевая студия"),
    SearchParams(question="Где заказать торт на день рождения?", rubrics="Кондитерская"),
    SearchParams(question="Лучшие шиномонтажи в районе", rubrics="Шиномонтаж"),
    SearchParams(question="Где можно заказать бизнес-ланч?", keywords="бизнес.?ланч"),
    SearchParams(
        question="Где хорошо делают педикюр?",
        keywords="педикюр",
        rubrics="Ногтевая студия",
    ),
    SearchParams(
        question="Торговые центры с фуд-кортом",
        keywords="фуд-?корт",
        rubrics="Торговый центр",
    ),
    SearchParams(question="Фитнес-клубы с бассейном", keywords="бассейн", rubrics="Фитнес-клуб"),
    SearchParams(
        question="Парикмахерские с детскими мастерами",
        keywords="(дет|ребен)",
        rubrics="Парикмахерская",
    ),
    SearchParams(question="Где есть магазины электроники?", rubrics="Магазин электроники"),
    SearchParams(
        question="Магазины алкоголя с широким выбором",
        keywords="выбор",
        rubrics="Магазин алкогольных напитков",
    ),
    SearchParams(question="Кальянные с уютной атмосферой", keywords="уют", rubrics="Кальян-бар"),
    SearchParams(
        question="Клиники с консультацией терапевта",
        keywords="терапевт",
        rubrics="Медцентр, клиника",
    ),
    SearchParams(question="Автомойки", rubrics="Автомойка"),
    SearchParams(
        question="Строительные магазины с доставкой",
        keywords="доставка",
        rubrics="Строительный магазин",
    ),
    SearchParams(question="Места для свиданий", keywords="(романтическ|свидание)"),
    SearchParams(
        question="Магазины одежды с примерочными",
        keywords="примерочные",
        rubrics="Магазин одежды",
    ),
    SearchParams(question="Спа-салоны с массажем", keywords="массаж", rubrics="Спа-салон"),
    SearchParams(question="Покажи авто детейлинги", keywords="детейлинг"),
    SearchParams(
        question="Автосалоны с тест-драйвом",
        keywords="тест-?драйв",
        rubrics="Автосалон",
    ),
    SearchParams(question="Где подают хороший капучино?", keywords="капучино"),
    SearchParams(
        question="Магазины с большим выбором цветов",
        keywords="большой выбор цветов",
        rubrics="Магазин цветов",
    ),
    SearchParams(question="Эпиляция в салонах красоты", rubrics="Эпиляция"),
    SearchParams(
        question="Где можно покрасить волосы?",
        keywords="краси(ть|ла)",
        rubrics="Салон красоты|Парикмахерская",
    ),
    SearchParams(
        question="Диагностические центры с УЗИ",
        keywords="УЗИ",
        rubrics="Диагностический центр",
    ),
    SearchParams(question="Пекарни со свежим хлебом", keywords="свежее", rubrics="Пекарня"),
    SearchParams(question="Где можно поесть роллы?", keywords="роллы", rubrics="Суши-бар"),
    SearchParams(question="Пункты выдачи", rubrics="Пункт выдачи"),
    SearchParams(
        question="Где есть развлечения для детей?",
        keywords="(для детей|детям)",
        rubrics="Развлекательный центр",
    ),
    SearchParams(
        question="Лучшие косметологии с лазерными процедурами",
        keywords="лазер",
        rubrics="Косметология",
    ),
    SearchParams(
        question="Магазины с посудой и товарами для кухни",
        keywords="посуд",
        rubrics="Товары для дома",
    ),
]


def main():
    params = Params.model_validate(yaml.safe_load(open("params.yaml"))["generate_embed_questions"])
    logger.debug("Loaded params: {}", params)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="csv-dataset path")

    args = parser.parse_args()
    input_data_path = args.dataset

    df = pd.read_csv(input_data_path)

    df["text"] = df["text"].apply(lambda s: "".join(ch for ch in s if ch not in set(string.punctuation)))

    not_found_keywords = []
    low_results = []
    low_results_treshold = 2
    result_data = []

    for item in keywords_by_phrases:
        searchres = df

        if item.rubrics:
            searchres = searchres[searchres["rubrics"].str.contains(item.rubrics, regex=True, case=False, na=False)]
        if item.keywords:
            searchres = searchres[searchres["text"].str.contains(item.keywords, regex=True, case=False, na=False)]

        if len(searchres) == 0:
            not_found_keywords.append(item.keywords or item.rubrics)
            continue

        if len(searchres) < low_results_treshold:
            low_results.append(item.keywords or item.rubrics)

        searchres["question"] = item.keywords or item.rubrics

        result_data.append(
            {
                "question": item.question,
                "names_ru": ";".join(list(set(searchres["name_ru"].values.tolist()))),
            }
        )

    if len(not_found_keywords) > 0:
        logger.error("Not found: {}", "\n- ".join(not_found_keywords))

    if len(low_results) > 0:
        logger.warning("Low result count:\n- {}", "\n- ".join(low_results))

    out_dir = Path(params.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    output_path = f"{params.out_dir}/{Path(input_data_path).name}"
    pd.DataFrame(result_data).to_csv(output_path, index=False)
    logger.info("Saved to {}", output_path)


if __name__ == "__main__":
    main()
