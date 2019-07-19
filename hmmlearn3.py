import json
import math
import copy
import sys
from collections import defaultdict



'''
while going through each each, 
keep track of init state, all state, only final state 
for each tag, enumerate it in dict, 
maintain numpy array for init state, all state and final state 


calculate the denominator values initially for each tag 

generate all possible states for transition states 
'''
vocabulary = set()
tag_set  = set()
tag_dict = defaultdict(int)
initial_freq = defaultdict(int)
overall_freq = defaultdict(int)
total_freq = defaultdict(int)
transition = list()
tag_freq = list()
observation_matrix = [[]]
tag_transition = [[]]
initial_state_transition = []

def getWordTagPair(wt_string):
    w1 = ""
    t1 = ""
    t_wt_split = wt_string.split("/")
    if len(t_wt_split) == 2:
        w1, t1 = t_wt_split[0], t_wt_split[1]
    else:
        n = len(t_wt_split)
        t1 = t_wt_split[n - 1]
        w1 = "/".join(t_wt_split[0:n - 1])

    return w1, t1

def HMMtagger(input_data):
    '''

    :param input_data: tagged corpus
    each line contains a sentence whose words are tagged with
    their part of speech in the format -
        word/TAG
    :return: Null
    '''
    global vocabulary
    input_ds = list()
    for sentence in input_data.splitlines():
        word_tag_array = []
        wordtag_pair = sentence.split()
        wt_len = len(wordtag_pair)

        w1, t1 = getWordTagPair(wordtag_pair[0])
        word_tag_array.append(w1)
        word_tag_array.append(t1)
        tag_set.add(t1)
        vocabulary.add(w1)
        initial_freq[t1] += 1
        overall_freq[t1] += 1
        total_freq[t1] += 1
        for i in range(1, wt_len-1):
            wi, ti = getWordTagPair(wordtag_pair[i])
            word_tag_array.append(wi)
            word_tag_array.append(ti)
            tag_set.add(ti)
            vocabulary.add(wi)
            overall_freq[ti] += 1
            total_freq[ti] += 1

        last_word, last_tag = getWordTagPair(wordtag_pair[wt_len-1])
        word_tag_array.append(last_word)
        word_tag_array.append(last_tag)
        total_freq[last_tag] += 1
        vocabulary.add(last_word)
        tag_set.add(last_tag)

        input_ds.append(word_tag_array)
    # for q in input_ds:
    #     print(q)
    '''
    Create a 2D matrix for calculating state(tag) transitions
    '''
    # map tags to numbers
    tag_num_map = {}
    ctr = 0
    for tag in tag_set:
        tag_num_map[tag] = ctr
        ctr += 1
    vocab_num_map = {}
    ctr = 0
    for v in vocabulary:
        vocab_num_map[v] = ctr
        ctr += 1

    num_tag_map = {}
    for t, n in tag_num_map.items():
        num_tag_map[n] = t
    num_vocab_map = {}
    for v, n in num_vocab_map.items():
        num_vocab_map[n] = v


    tag_transition = [[0 for x in range(len(tag_set))] for y in range(len(tag_set))]
    initial_state_transition = [0 for x in range(len(tag_set))]
    no_of_samples = len(input_ds)

    for sentence in input_ds:
        initial_state_id = tag_num_map[sentence[1]]
        initial_state_transition[initial_state_id] += 1
        for i in range(0,len(sentence)-3,2):
            tag_transition[tag_num_map[sentence[i+1]]][tag_num_map[sentence[i+3]]] += 1

    # print("total_freq")
    # print(total_freq)
    # print("Transition counts ")


    tag_set_size = len(tag_set)
    for i in range(len(tag_transition)):
        initial_state_transition[i] = math.log((initial_state_transition[i]+1)/(no_of_samples*1.0 + tag_set_size))
        f = overall_freq[num_tag_map[i]]
        for j in range(len(tag_transition)):
            tag_transition[i][j] = math.log((tag_transition[i][j]+1)/(f*1.0 + tag_set_size))

    '''
    Create a matrix for observation|state
    cols = vocab
    row = tag

    '''
    observation_matrix = [[0 for x in range(len(vocabulary))] for y in range(len(tag_set))]

    for sentence in input_ds:
        for i in range(0, len(sentence)-1,2):
            word = sentence[i]
            tag = sentence[i+1]
            word_id = vocab_num_map[word]
            tag_id = tag_num_map[tag]
            observation_matrix[tag_id][word_id] += 1

    # this observation matrix is also needed
    emission_freq = copy.copy(observation_matrix)
    # print(" s_id ",vocab_num_map["'s"])
    # print("PART ", tag_num_map["PART"])
    # print(emission_freq[tag_num_map["PART"]][vocab_num_map["'s"]])
    # write to file
    for i in range(len(tag_set)):
        for j in range(len(vocabulary)):
            tag = num_tag_map[i]
            freq = total_freq[tag]
            observation_matrix[i][j] /= freq*1.0
            if observation_matrix[i][j] != 0.0:
                observation_matrix[i][j] = math.log(observation_matrix[i][j])
            else:
                observation_matrix[i][j] = "undef"



    HMM_model = {}
    HMM_model["vocabulary"] = list(vocabulary)
    HMM_model["tag_set"] = list(tag_set)
    HMM_model["initial_state_transition"] = initial_state_transition
    HMM_model["tag_transition"] = tag_transition
    HMM_model["emission_freq"] = emission_freq
    HMM_model["observation_matrix"] = observation_matrix
    HMM_model["num_tag_map"] = num_tag_map
    HMM_model["num_vocab_map"] = num_vocab_map
    HMM_model["tag_num_map"] = tag_num_map
    HMM_model["vocab_num_map"] = vocab_num_map
    # print("In HMM learn ")
    # print(HMM_model["emission_freq"])
    # print("'s/PART")
    with open("hmmmodel.txt","w") as model_file:
        model_file.write(json.dumps(HMM_model, indent=4))


if __name__ == '__main__':
    with  open(sys.argv[1],encoding="utf8") as training_file:
        training_corpus = training_file.read()
        HMMtagger(training_corpus)




