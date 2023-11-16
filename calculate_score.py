from rouge_score import rouge_scorer
import os

HUMAN_SUM_PATH = "D:/Hochanhmetmoi/NLP/BT/textSummarization/data/clusters/cluster_"
SUMMARY_PATH = "D:/Hochanhmetmoi/NLP/BT/textSummarization/summaries/summary_"
OUTPUT_PATH = "D:/Hochanhmetmoi/NLP/BT/textSummarization/scores/score.txt"

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)


def find_score(human_summary, summary):
    score = scorer.score(human_summary, summary)

    new_score = {}
    for key in score.keys():
        new_score[key] = {
            'precision': score[key].precision,
            'recall': score[key].recall,
            'fmeasure': score[key].fmeasure
        }
    return new_score


def to_string(score):
    content = ""
    for key in score.keys():
        content += str(key) + ": precision = " + str(round(score[key]["precision"], 5)) + ", recall = " + str(
            round(score[key]["recall"], 5)) + ", fmeasure = " + str(round(score[key]["fmeasure"], 5)) + "\n"
    return content


def find_sum(dic_1, dic_2):
    average = {}
    for key in dic_1.keys():
        average[key] = dic_1[key] + dic_2[key]
    return average


def find_average_score(score_array):
    new_score = {}
    for score in score_array:
        for key in score.keys():
            if key not in new_score:
                new_score[key] = score[key]
            else:
                new_score[key] = find_sum(score[key], new_score[key])

    for key in new_score.keys():
        for key_child in new_score[key].keys():
            new_score[key][key_child] /= 200
    return new_score


human_sum_path = []
summary_path = []
# output_path = []
for i in range(200):
    human_sum_path.append(HUMAN_SUM_PATH + str(i + 1))
    summary_path.append(SUMMARY_PATH + str(i + 1) + ".txt")


def read_text_file(file_path):
    # Đọc dữ liệu
    read_file = open(file_path, "r", encoding="utf8", errors="ignore")
    data = read_file.read()
    read_file.close()
    return data


def write_text_file(file_path, text):
    write_file = open(file_path, "w", encoding="utf8", errors="ignore")
    write_file.write(text)
    write_file.close()


content = ""
score_array = []
for index, path in enumerate(human_sum_path):
    os.chdir(path)
    summary = read_text_file(summary_path[index])
    content += "cluster " + str(index + 1) + ": \n"
    for file in os.listdir():
        if file.endswith(".ref1.txt"):
            human_sum_path = f"{path}/{file}"
            human_summary = read_text_file(human_sum_path)

            score = find_score(human_summary, summary)
            score_array.append(score)

            content = content + "---- Score between human summary and our summary: \n" + to_string(score) + "\n"
    content += "\n"

average_score = find_average_score(score_array)
content = ("---- Average score between human summary and our summary: \n" + to_string(average_score) + "\n" +
           "-----------------------------------------------------------------------------------\n" + content)

write_text_file(OUTPUT_PATH, content)
print("Write in scores successfully!")
