import json
import os
import evaluate
from tqdm import tqdm
from prettytable import PrettyTable

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

bleu = evaluate.load("bleu")
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')
chrf = evaluate.load("chrf")
comet = evaluate.load('comet')

org_tur = os.listdir("wikihow-tr-archive/original-wikihow-tr/")
org_eng = os.listdir("wikihow-tr-archive/original-wikihow-tr-english-versions/")
tr_tur = os.listdir("wikihow-tr-archive/original-wikihow-tr-english-versions-translated/")

mode = "with"

def main():
    def build_content_without_desc(data):
        import re
        text = ""
        text += data["title"] + "\n"

        for i in range(len(data["methods"])):
            method = data["methods"][i]
            for i in range(len(method["steps"])):
                if "headline" in method["steps"][i].keys():
                    if (method["steps"][i]["headline"] != ""):
                        text += method["steps"][i]["headline"].strip() + "\n" 

        for i in range(len(data["parts"])):
            part = data["parts"][i]
            for i in range(len(part["steps"])):
                if "headline" in part["steps"][i].keys():
                    if (part["steps"][i]["headline"] != ""):
                        text += part["steps"][i]["headline"].strip() + "\n" 

        for i in range(len(data["steps"])):
            if "headline" in data["steps"][i].keys():
                if (data["steps"][i]["headline"] != ""):
                    text += data["steps"][i]["headline"].strip() + "\n" 
            elif "name" in data["steps"][i].keys():
                for i2 in range(len(data["steps"][i]["steps"])):
                    text += data["steps"][i]["steps"][i2]["headline"].strip() + "\n" 

        text = re.sub(r'{[^}]*}*', '', text)
        return text

    def build_content_with_desc(data):
        import re
        text = ""
        text += data["title"] + "\n"

        for i in range(len(data["methods"])):
            method = data["methods"][i]
            for i in range(len(method["steps"])):
                if "headline" in method["steps"][i].keys() and "description" in method["steps"][i].keys():
                    if (method["steps"][i]["headline"] != "") and ((method["steps"][i]["description"] != "")):
                        text += method["steps"][i]["headline"].strip() + " " + method["steps"][i]["description"].strip() + "\n" 

        for i in range(len(data["parts"])):
            part = data["parts"][i]
            for i in range(len(part["steps"])):
                if "headline" in part["steps"][i].keys() and "description" in part["steps"][i].keys():
                    if (part["steps"][i]["headline"] != "") and (part["steps"][i]["description"] != ""):
                        text += part["steps"][i]["headline"].strip() + " " + part["steps"][i]["description"].strip() + "\n" 

        for i in range(len(data["steps"])):
            if "headline" in data["steps"][i].keys():
                if (data["steps"][i]["headline"] != "") and (data["steps"][i]["description"] != ""):
                    text += data["steps"][i]["headline"].strip() + " " + data["steps"][i]["description"].strip() + "\n" 
            elif "name" in data["steps"][i].keys():
                for i2 in range(len(data["steps"][i]["steps"])):
                    text += data["steps"][i]["steps"][i2]["headline"].strip() + " " + data["steps"][i]["steps"][i2]["description"].strip() + "\n" 

        text = re.sub(r'{[^}]*}*', '', text)
        return text

    chrf_list = []
    chrf_pp_list = []
    comet_list = []
    bleu_list = []
    meteor_list = []
    rouge_list = []

    for file in tqdm(org_eng):
        eng_data = json.load(open("wikihow-tr-archive/original-wikihow-tr-english-versions/"+file, "r", encoding="utf-8"))
        if ("Cars and Other Vehicles" in eng_data["category_hierarchy"]) or ("Computers and Electronics" in eng_data["category_hierarchy"]) or ("Home and Garden" in eng_data["category_hierarchy"]) or ("Pets and Animals" in eng_data["category_hierarchy"]) or ("Hobbies and Crafts" in eng_data["category_hierarchy"]) or ("Health" in eng_data["category_hierarchy"]):
            try:
                tr_article_title = eng_data["other_languages"]["Türkçe"]["title"]
                org_tur_data = json.load(open("wikihow-tr-archive/original-wikihow-tr/"+tr_article_title.replace("'", "")+".json", "r", encoding="utf-8"))
                tra_tur_data = json.load(open("wikihow-tr-archive/original-wikihow-tr-english-versions-translated/"+file, "r", encoding="utf-8"))


                if mode == "without":
                    eng_content = build_content_without_desc(eng_data).lower()
                    org_tur_content = build_content_without_desc(org_tur_data).lower()
                    tra_tur_content = build_content_without_desc(tra_tur_data).lower()

                elif mode == "with":
                    eng_content = build_content_with_desc(eng_data).lower()
                    org_tur_content = build_content_with_desc(org_tur_data).lower()
                    tra_tur_content = build_content_with_desc(tra_tur_data).lower()

                chrf_results = chrf.compute(predictions=[tra_tur_content], references=[org_tur_content])["score"]
                chrf_pp_results = chrf.compute(predictions=[tra_tur_content], references=[org_tur_content], word_order=2)["score"]
                bleu_results = bleu.compute(predictions=[tra_tur_content], references=[org_tur_content])["bleu"] * 100
                meteor_results = meteor.compute(predictions=[tra_tur_content], references=[org_tur_content])["meteor"] * 100
                rouge_results = rouge.compute(predictions=[tra_tur_content], references=[org_tur_content])["rougeL"] * 100
                comet_results = comet.compute(predictions=[tra_tur_content], references=[org_tur_content], sources=[eng_content])["scores"][0] * 100

                chrf_list.append(chrf_results)
                chrf_pp_list.append(chrf_pp_results)
                comet_list.append(comet_results)
                bleu_list.append(bleu_results)
                meteor_list.append(meteor_results)
                rouge_list.append(rouge_results)
            except:
                pass
    chrf_txt = open("chrf.txt", "w", encoding="utf-8")
    for score in chrf_list:
        chrf_txt.writelines(str(score) + "\n")
    chrf_txt.close()

    chrf_pp_txt = open("chrf_pp.txt", "w", encoding="utf-8")
    for score in chrf_pp_list:
        chrf_pp_txt.writelines(str(score) + "\n")
    chrf_pp_txt.close()

    comet_txt = open("comet.txt", "w", encoding="utf-8")
    for score in comet_list:
        comet_txt.writelines(str(score) + "\n")
    comet_txt.close()

    bleu_txt = open("bleu.txt", "w", encoding="utf-8")
    for score in bleu_list:
        bleu_txt.writelines(str(score) + "\n")
    bleu_txt.close()

    rouge_txt = open("rouge.txt", "w", encoding="utf-8")
    for score in rouge_list:
        rouge_txt.writelines(str(score) + "\n")
    rouge_txt.close()

    meteor_txt = open("meteor.txt", "w", encoding="utf-8")
    for score in meteor_list:
        meteor_txt.writelines(str(score) + "\n")
    meteor_txt.close()

    print(mode)
if __name__ == "__main__":
    main()    