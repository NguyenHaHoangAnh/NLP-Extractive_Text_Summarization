import os
import py_vncorenlp
import string
import math
import numpy as np
import networkx as nx


LIB_URL = "D:/Hochanhmetmoi/NLP/BT/textSummarization/vncorenlp"
INPUT_PATH = "D:/Hochanhmetmoi/NLP/BT/textSummarization/data/clusters/cluster_"
OUTPUT_PATH = "D:/Hochanhmetmoi/NLP/BT/textSummarization/summaries/summary_"

# Automatically download VnCoreNLP components from the original repository
# and save them in some local working folder
# py_vncorenlp.download_model(save_dir=LIB_URL)


# Tách nhanh
# Chỉ tách câu, không gắn POS tag cho từ
# KHÔNG khai báo model
# (Vì khai báo model sẽ tự động dùng cả POS tag, ... -> Lâu)
# rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=LIB_URL)
# text = "Trung bình người dùng sử dụng Internet hiện nay phải nhớ hơn vài chục mật khẩu cho các trang web."
# output = rdrsegmenter.word_segment(text)
# print(output)


# Load VnCoreNLP from the local working folder that contains both `VnCoreNLP-1.2.jar` and `models`
model = py_vncorenlp.VnCoreNLP(save_dir=LIB_URL)
# Equivalent to: model = py_vncorenlp.VnCoreNLP(annotators=["wseg", "pos", "ner", "parse"], save_dir=LIB_URL)


stop_words = [
    "bị", "bởi", "cả", "các", "cái", "cần", "càng", "chỉ", "chiếc", "cho", "chứ", "chưa",
    "chuyện", "có", "có_thể", "cứ", "của", "cùng", "cũng", "đã", "đang", "đây", "để", "đến_nỗi", "đều",
    "điều", "do", "đó", "được", "dưới", "gì", "khi", "không", "là", "lại", "lên", "lúc", "mà", "mỗi", "một_cách",
    "này", "nên", "nếu", "ngay", "nhiều", "như", "nhưng", "những", 'nơi', "nữa", "phải", "qua", "ra", "rằng", "rằng",
    "rất", "rất", "rồi", "sau", "sẽ", "so", "sự", "tại", "theo", "thế", "thì", "trên", "trước", "từ", "từng", "và",
    "vẫn", "vào", "vậy", "vì", "việc", "với", "vừa"
]
punc = list(string.punctuation + "\n" + "“" + "”")


def document_segment(document):
    doc = model.input_word_segment(document)
    doc = " ".join(doc)
    return doc


def document_to_sentence_array(doc):
    sentence_array = doc.split(" . ")
    return sentence_array


def sentence_to_word_array(sentence):
    # Xóa dấu câu (trừ dấu "_")
    remove = string.punctuation + "\n" + "“" + "”"
    remove = remove.replace("_", "")
    sentence = sentence.translate(str.maketrans('', '', remove))
    # Xóa khoảng trắng thừa giữa 2 từ
    sentence = " ".join(sentence.split())
    # Chia câu thành các mảng từ đơn
    word_array = sentence.split(" ")
    # Loại bỏ các dấu "..."
    punctual = "..."
    if punctual in word_array:
        word_array.remove("...")
    new_word_array = []
    for word in word_array:
        if word not in punc:
            new_word_array.append(word.lower())
    return new_word_array


def document_to_word_array(sentence_array, noun_array, verb_array):
    new_word_array = []
    for sentence in sentence_array:
        # Xóa dấu câu (trừ dấu "_")
        remove = string.punctuation + "\n" + "“" + "”"
        remove = remove.replace("_", "")
        sentence = sentence.translate(str.maketrans('', '', remove))
        # Xóa khoảng trắng thừa giữa 2 từ
        sentence = " ".join(sentence.split())
        # Chia câu thành các mảng từ đơn
        word_array = sentence.split(" ")
        # Loại bỏ các dấu "..."
        punctual = "..."
        if punctual in word_array:
            word_array.remove("...")
        for word in word_array:
            if word.lower() not in punc and word.lower() not in stop_words:
                if word.lower() in noun_array or word.lower() in verb_array:
                    if word.lower() not in new_word_array:
                        new_word_array.append(word.lower())

    return new_word_array


def find_tf_df(tf, df, sentence_array, text_word_array):
    for sentence in sentence_array:
        word_array = sentence_to_word_array(sentence)
        length = len(word_array)
        # calculate tf in each sentence
        word_frequency = {}
        for text_word in text_word_array:
            if text_word in word_array:
                for word in word_array:
                    if word not in punc and word not in stop_words:
                        if word == text_word:
                            if word not in word_frequency.keys():
                                word_frequency[word] = 1 / length
                            else:
                                word_frequency[word] += 1 / length
            else:
                word_frequency[text_word] = 0
        tf.append(word_frequency)
        # calculate df in document
        for word in word_frequency.keys():
            if word_frequency[word] > 0:
                if word not in df.keys():
                    df[word] = 1
                else:
                    df[word] += 1


def find_tf_idf(tf, df, sentence_array, text_word_array):
    find_tf_df(tf, df, sentence_array, text_word_array)
    length = len(sentence_array)
    tf_idf = []
    for index, sentence in enumerate(sentence_array):
        word_score = {}
        for word in tf[index].keys():
            word_score[word] = tf[index][word] * math.log(length / df[word], 2)
        tf_idf.append(word_score)
    return tf_idf


def find_cosine(i, j, tf_idf, text_word_array):
    x = tf_idf[i]
    y = tf_idf[j]

    tu_so = 0
    mau_so_x = 0
    mau_so_y = 0
    for text_word in text_word_array:
        tu_so = tu_so + x[text_word] * y[text_word]
        mau_so_x = mau_so_x + x[text_word] ** 2
        mau_so_y = mau_so_y + y[text_word] ** 2

    mau_so = math.sqrt(mau_so_x * mau_so_y)

    if mau_so == 0:
        return 0.0
    phan_so = tu_so / mau_so
    if phan_so > 0.6:
        return 0.0
    return phan_so


def get_similarity_matrix(sentence_array, text_word_array, tf_idf):
    length = len(sentence_array)
    similarity_matrix = np.zeros((length, length))
    for i in range(length):
        for j in range(length):
            if i != j:
                similarity_matrix[i][j] = find_cosine(i, j, tf_idf, text_word_array)
    return similarity_matrix


def get_summary_position(rank_sentences, max_length):
    max_position_array = []
    for i in range(max_length):
        max_position_array.append(rank_sentences[i][1])
    summary_position_array = sorted(max_position_array)
    return summary_position_array


def get_summary(sentence_array, summary_position_array):
    summary = ""
    for index in summary_position_array:
        if sentence_array[index].endswith(" ."):
            sentence_array[index] = sentence_array[index].replace(" .", "")
        summary = summary + sentence_array[index] + ". "

    summary = summary.replace("_", " ")
    summary = summary.replace(" , ", ", ")
    return summary


def text_summarization(document):
    noun_array = model.get_noun_list(document)
    verb_array = model.get_verb_list(document)
    doc = document_segment(document)
    sentence_array = document_to_sentence_array(doc)
    text_word_array = document_to_word_array(sentence_array, noun_array, verb_array)
    tf = []
    df = {}
    tf_idf = find_tf_idf(tf, df, sentence_array, text_word_array)
    max_length = int(math.ceil(len(sentence_array) ** 0.5))
    similarity_matrix = get_similarity_matrix(sentence_array, text_word_array, tf_idf)
    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)
    rank_sentences = sorted(((scores[i], i) for i, s in enumerate(sentence_array)), reverse=True)
    summary_position_array = get_summary_position(rank_sentences, max_length)
    summary = get_summary(sentence_array, summary_position_array)
    return summary


input_path = []
output_path = []
for i in range(200):
    input_path.append(INPUT_PATH + str(i + 1))
    output_path.append(OUTPUT_PATH + str(i + 1) + ".txt")


def read_text_file(file_path):
    # Đọc dữ liệu
    read_file = open(file_path, "r", encoding="utf8", errors="ignore")
    data = read_file.read()
    if not data.endswith("."):
        data += "."
    read_file.close()
    return data


def write_text_file(file_path, text):
    # Ghi dữ liệu
    write_file = open(file_path, "w", encoding="utf8", errors="ignore")
    write_file.write(text)
    write_file.close()


for index, path in enumerate(input_path):
    os.chdir(path)

    document = ""
    for file in os.listdir():
        if file.endswith(".body.txt"):
            file_path = f"{path}/{file}"

            document = document + read_text_file(file_path)
    summary = text_summarization(document)

    write_text_file(output_path[index], summary)


print("Write in summaries successfuly!")





