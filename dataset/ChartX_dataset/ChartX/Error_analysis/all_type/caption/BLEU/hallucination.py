import os
import json
import re
import pycountry
import spacy
import pandas as pd
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

def extract_years_from_caption(text):
    doc = nlp(text)
    return set(re.findall(r"\b(?:19|20)\d{2}\b", " ".join(ent.text for ent in doc.ents if ent.label_ == "DATE")))

def extract_years_regex(text):
    return set(re.findall(r"\b(?:19|20)\d{2}\b", text))

def extract_countries_from_csv(csv_text):
    countries = set()
    csv_text = csv_text.replace("\\t", "\t").replace("\\n", "\n")
    lines = csv_text.split("\n")
    for line in lines:
        cols = line.strip().split("\t")
        if not cols:
            continue
        first_col = cols[0].strip()
        if not first_col or first_col.lower() == "country":
            continue
        try:
            countries.add(pycountry.countries.lookup(first_col).name)
        except LookupError:
            alias_map = {
                "USA": "United States", "US": "United States", "U.S.": "United States",
                "UK": "United Kingdom", "U.K.": "United Kingdom", "Korea": "South Korea"
            }
            if first_col.upper() in alias_map:
                countries.add(alias_map[first_col.upper()])
    return countries

def extract_and_standardize_countries(text):
    country_set = set()
    doc = nlp(text)

    # ✅ 合法正式国家名集合（精确匹配）
    official_country_names = set(country.name for country in pycountry.countries)

    # ✅ 别名映射（大写key统一匹配）
    alias_map = {
        "USA": "United States", "US": "United States", "U.S.": "United States",
        "UK": "United Kingdom", "U.K.": "United Kingdom", "KOREA": "South Korea"
    }

    # ✅ 禁止误识别的一些固定短语（例如单位）
    forbidden_phrases = [
        "U.S. dollars", "US dollars", "United States dollars", "U.S. Dollar", "US Dollar"
    ]
    lower_text = text.lower()  # 全部小写便于匹配完整短语

    # === 1. 精确匹配 NER 中的 GPE/LOC 实体 ===
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            ent_text = ent.text.strip()
            ent_upper = ent_text.upper()
            ent_lower = ent_text.lower()

            if ent_text in official_country_names:
                country_set.add(ent_text)
            elif ent_upper in alias_map:
                # ✅ 如果实体作为 forbidden 短语一部分存在则跳过
                if any(ent_lower in phrase and phrase in lower_text for phrase in forbidden_phrases):
                    continue
                country_set.add(alias_map[ent_upper])

    # === 2. 严格正则匹配疑似国家名（避免 "and" → "Andorra"）===
    tokens = re.findall(r"\b[A-Z][a-z]{3,}(?:\s+[A-Z][a-z]+)*\b", text)
    for token in tokens:
        if token in official_country_names:
            country_set.add(token)

    # === 3. 匹配常见缩写别名（排除单位类短语）===
    for abbr in re.findall(r"\b[A-Z][A-Z\.]+\b", text):
        abbr_clean = abbr.replace(".", "").upper()
        if abbr_clean in alias_map:
            # ✅ 如果该缩写在 forbidden phrase 中作为前缀（如 U.S. in U.S. dollars），则跳过
            skip = False
            for phrase in forbidden_phrases:
                if phrase.lower().startswith(abbr.lower()) and phrase.lower() in lower_text:
                    skip = True
                    break
            if skip:
                continue
            country_set.add(alias_map[abbr_clean])

    return country_set


def analyze_one_model(json_path, preview_limit=5):
    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    model_name = data[0].get("model_name", os.path.splitext(os.path.basename(json_path))[0])
    total, year_hall, country_hall, overall = 0, 0, 0, 0
    preview_printed = 0

    print(f"\n📊 模型: {model_name}")
    for item in tqdm(data, desc=model_name):
        caption = item.get("generated_caption", "")
        title = item.get("title", "")
        csv_text = item.get("csv", "")

        caption_years = extract_years_from_caption(caption)
        context_years = extract_years_regex(title) | extract_years_regex(csv_text)
        caption_countries = extract_and_standardize_countries(caption)
        context_countries = extract_and_standardize_countries(title) | extract_countries_from_csv(csv_text)

        year_hallucination = any(y not in context_years for y in caption_years)
        country_hallucination = any(c not in context_countries for c in caption_countries)

        if year_hallucination: year_hall += 1
        if country_hallucination: country_hall += 1
        if year_hallucination or country_hallucination: overall += 1
        total += 1

        if preview_printed < preview_limit:
            print(f"\n—— Sample {preview_printed+1} ——")
            print(f"📷 图像名: {item.get('imgname', '')}")
            print(f"🕓 Caption 中年份: {list(caption_years)}  |  Title/CSV 中年份: {list(context_years)}")
            print(f"🧠 时间幻觉: {year_hallucination}")
            print(f"🌍 Caption 中国家: {list(caption_countries)}  |  Title/CSV 中国家: {list(context_countries)}")
            print(f"🧠 地点幻觉: {country_hallucination}")
            preview_printed += 1

    return {
        "model_name": model_name,
        "total": total,
        "year_hallucination": year_hall,
        "country_hallucination": country_hall,
        "overall_hallucination": overall
    }

def batch_process_model_files(folder_path, output_csv_path, preview_limit=5):
    results = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".json"):
            json_path = os.path.join(folder_path, filename)
            try:
                result = analyze_one_model(json_path, preview_limit=preview_limit)
                results.append(result)
            except Exception as e:
                print(f"❌ 跳过 {filename}，错误: {e}")
    pd.DataFrame(results).to_csv(output_csv_path, index=False)
    print(f"\n✅ 所有模型统计结果已保存到: {output_csv_path}")

if __name__ == "__main__":
    input_folder = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/deep_analysis/caption_all"
    output_csv = "/data/jguo376/project/dataset/ChartX_dataset/ChartX/Error_analysis/all_type/caption/BLEU/hallucination_stats_by_model.csv"
    batch_process_model_files(input_folder, output_csv, preview_limit=5)
