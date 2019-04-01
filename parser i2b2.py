import os
from nltk import pos_tag, RegexpParser
import numpy as np
import pandas as pd


a_ids = []
e_ids = []

for filename in os.listdir("./data/annotations"):
    if filename[0] != ".":  # ignore hidden files
        a_ids.append(int(filename))
for filename in os.listdir("./data/entries"):
    if filename[0] != ".":
        e_ids.append(int(filename))

a_ids = tuple(sorted(a_ids))
e_ids = tuple(sorted(e_ids))

intersection = list(set(a_ids) & set(e_ids))
if len(intersection) == len(a_ids):
    print("Success: all anotations have a corresponding entry.", len(intersection))


# build annotation and entry corpora

a_corpus = []
e_corpus = []

#for each file in a_ids get the whole file content and add to a_corpus and get same file from entries and add to e_corpus--this is for genrating annotations and entry corpus
# only annotations and corresponding files
for file in a_ids:
    path = "./data/annotations/" + str(file)
    with open(path) as f:
        content = f.read().splitlines()
        a_corpus.append(content)

    path = "./data/entries/" + str(file)
    with open(path) as f:
        # content = f.readlines()
        content = f.read().splitlines()
        e_corpus.append(content)


#  ["id", "row", "offset", "word", "POS", "chunk", "NER"]
entries_cols = ["id", "row", "offset", "word"]
entries_df = pd.DataFrame(columns=entries_cols)
entries_df.head()


annotations_cols = ["id", "NER_tag", "row", "offset", "length"]
annotations_df = pd.DataFrame(columns=annotations_cols)
annotations_df.head()

#we get medication counts, dosage counts etc. Each a_copus has all annotations files. Each file in a-corpus is document and in each document
# we go line by line and check if nm is there, if not then increment med etc counts

med_count = 0
dosage_count = 0
mode_count = 0
freq_count = 0
dur_count = 0
reason_count = 0

for document in a_corpus:
    for line in document:
        if "m=\"nm\"" not in line:
            med_count += 1
        if "do=\"nm\"" not in line:
            dosage_count += 1
        if "mo=\"nm\"" not in line:
            mode_count += 1
        if "f=\"nm\"" not in line:
            freq_count += 1
        if "du=\"nm\"" not in line:
            dur_count += 1
        if "r=\"nm\"" not in line:
            reason_count += 1

print("Medication annotations: ", med_count)
print("Dosage annotations: ", dosage_count)
print("Mode annotations: ", mode_count)
print("Frequency annotations: ", freq_count)
print("Duration annotations: ", dur_count)
print("Reason annotations: ", reason_count)



annotations_df = pd.DataFrame(columns=annotations_cols)  # reset df
tmp_list = []

for i, document in enumerate(a_corpus):

    for row in document:
        row = row.split("||")
        # print(row, "\n")

        for tag in row:
            # print(tag)
            tag = tag.split("=")
            if ":" in tag[1]:
                tag_label = tag[0].lstrip(" ")
                tag_row_a = tag[1].split(" ")[-2:][0].split(":")[0]
                tag_row_b = tag[1].split(" ")[-2:][1].split(":")[0]

                # some annotations have non-standard formatting (losing 64 instances)
                try:
                    tag_offset_a = int(tag[1].split(" ")[-2:][0].split(":")[1])
                    tag_offset_b = int(tag[1].split(" ")[-2:][1].split(":")[1])
                    length = tag_offset_b - tag_offset_a + 1

                    # 1 row = 1 token with a tag
                    first = True
                    BIO_tag = "B-"
                    if length > 1 and tag_row_a == tag_row_b:
                        for offset in range(tag_offset_a, tag_offset_b + 1):
                            if first:
                                tag_label = BIO_tag + tag_label
                                first = False
                            else:
                                tag_label = tag_label.replace("B-", "I-")
                            tmp_list.append([a_ids[i], tag_label, tag_row_a, offset, 1])
                    # TODO: tags over line breaks
                    else:
                        tmp_list.append([a_ids[i], BIO_tag + tag_label, tag_row_a, tag_offset_a, length])
                except:
                    pass

annotations_df = pd.DataFrame(tmp_list, columns=annotations_cols)
annotations_df.reset_index(inplace=True)
annotations_df = annotations_df.drop(columns=["index", "length"])
annotations_df.shape
annotations_df.head()
entries_df.head()
e_corpus[0][0].split(" ")

entries_df = pd.DataFrame(columns=entries_cols)  # reset df
tmp_list = []

for doc_i, document in enumerate(e_corpus):

    tmp_list.append([0, 0, 0, "-DOCSTART-"])
    tmp_list.append([0, 0, 0, "-EMPTYLINE-"])

    for row_i, row in enumerate(document):
        row_split = row.split(" ")
        for word_i, word in enumerate(row_split):
            word = word.rstrip(".")  # strip "." from end of word
            word = word.replace("\t", "")
            word_id = a_ids[doc_i]
            word_row = row_i + 1  # 1-based indexing
            word_offset = word_i  # 0-based indexing

            if len(word) > 0 and "|" not in word:
                tmp_list.append([word_id, word_row, word_offset, word])

    tmp_list.append([0, 0, 0, "-EMPTYLINE-"])

entries_df = pd.DataFrame(tmp_list, columns=entries_cols)
entries_df.head()
annotations_df.head()

ner_counter = [1 for i in annotations_df["NER_tag"] if "B-" in i]
print(len(ner_counter), "named entities")


# ensure correct dtypes
annotations_df[['id', 'row', 'offset']] = annotations_df[['id', 'row', 'offset']].apply(pd.to_numeric)
annotations_df['NER_tag'] = annotations_df["NER_tag"].astype(str)
entries_df[['id', 'row', 'offset']] = entries_df[['id', 'row', 'offset']].apply(pd.to_numeric)
entries_df["word"] = entries_df["word"].astype(str)
result_df = pd.merge(entries_df, annotations_df, how="left", on=['id', 'row', 'offset'])

# replace NaNs with "O"
print("columns with missing data:\n", result_df.isna().any())
result_df = result_df.fillna("O")

print("columns with missing data:\n", result_df.isna().any())
result_df = result_df.drop(columns=["id", "row", "offset"])
result_df.head()
result_df.shape

# 71 fewer annotations than expected as annotations over line breaks are not included
ner_counter = [1 for i in result_df["NER_tag"] if "B-" in i]
print(len(ner_counter), "named entities")

import nltk
from nltk.chunk.regexp import RegexpChunkParser, ChunkRule, RegexpParser
from nltk.tree import Tree

nltk.download('averaged_perceptron_tagger')

text = result_df["word"].tolist()
text_pos = pos_tag(text)
text_pos_list = [i[1] for i in text_pos]
len(text_pos_list)
result_df.columns
result_df["POS_tag"] = text_pos_list
result_df.head()


text_test = "EU rejects German call to boycott British lamb.".split(" ")
text_pos_test = pos_tag(text_test)
text_pos_test