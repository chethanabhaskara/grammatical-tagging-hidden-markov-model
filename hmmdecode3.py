import json
import math
import operator
import sys

import numpy

vocabulary = list()
tag = list()
initial_state_transition = list()
tag_transition = [[]]
observation_matrix = [[]]
emission_freq = [[]]
num_tag_map = dict()
tag_num_map = dict()
vocab_num_map = dict()
num_vocab_map = dict()


def parseModel(hmm_model):
    global vocabulary
    global tag
    global initial_state_transition
    global tag_transition
    global observation_matrix
    global num_tag_map
    global tag_num_map
    global vocab_num_map
    global num_vocab_map
    global emission_freq

    vocabulary = set(hmm_model["vocabulary"])
    tag = set(hmm_model["tag_set"])
    initial_state_transition = hmm_model["initial_state_transition"]
    tag_transition = hmm_model["tag_transition"]
    observation_matrix = hmm_model["observation_matrix"]
    num_tag_map = hmm_model["num_tag_map"]
    tag_num_map = hmm_model["tag_num_map"]
    vocab_num_map = hmm_model["vocab_num_map"]
    num_vocab_map = hmm_model["num_vocab_map"]
    emission_freq = hmm_model["emission_freq"]

    # print("emissionfreq********")
    # print(emission_freq)
    #
    # print("emission fre 's" , emission_freq[34][8956])
    # print("emission log 's ", observation_matrix[34][8956])

def maxProbability(probability, q, t, o_t):
    global observation_matrix
    global tag_transition
    global emission_freq
    vals = []

    # if o_t == -1 vocab not present
    for q1 in range(len(probability)):
        if probability[q1][t-1]!="invalid":
            v = probability[q1][t - 1] + tag_transition[q1][q]
            if o_t != -1:
                if observation_matrix[q][o_t] != "undef":
                    v += observation_matrix[q][o_t]
                elif observation_matrix[q][o_t] == "undef":
                    v = "invalid"

            if v != "invalid":
                vals.append(v)

    #need to check if len(val) is zero
    if len(vals) > 0:
        return max(vals)
    else:
        return "invalid"

def argmaxProbability(probability, q, t):
    global tag_transition

    vals = []
    val_dict = {}
    for q1 in range(len(probability)):
        if probability[q1][t-1]!= "invalid":
            val = probability[q1][t-1] + tag_transition[q1][q]
            val_dict[q1]  = val
            vals.append(val)
    return max(val_dict.items(), key=operator.itemgetter(1))[0]
    # return numpy.argmax(vals)

def backtrack(probability, backpointer, T):
    global num_tag_map

    vals = []
    vals_dict = {}
    for q1 in range(len(probability)):
        vals.append(probability[q1][T-1])
        if probability[q1][T-1]!="invalid":
            vals_dict[q1] = probability[q1][T-1]
    endstate = max(vals_dict.items(), key = operator.itemgetter(1))[0]
    # print(endstate)
    sequence = [endstate]
    current_state = endstate
    for i in range(T-1, 0, -1):
        next_state = backpointer[current_state][i]
        sequence.append(next_state)
        current_state = next_state

    sequence.reverse()
    # print(sequence)
    tag_seq = []
    # print(num_tag_map)
    for s in sequence:
        tag_seq.append(num_tag_map[str(s)])
    return (tag_seq)

def HMMdecode(test_data):
    global vocabulary
    global tagglobal
    global initial_state_transition
    global tag_transition
    global observation_matrix
    global num_tag_map
    global tag_num_map
    global vocab_num_map
    global num_vocab_map
    global emission_freq

    no_of_tags = len(tag)

    no_of_observations = len(vocabulary)
    output_file = open("hmmoutput.txt","w")
    for sentence in test_data.splitlines():

        words = sentence.split()
        T = len(words)
        probability = [[0 for x in range(T)] for y in range(no_of_tags)]
        backpointer = [[0 for x in range(T)] for y in range(no_of_tags)]


        for i in range(no_of_tags):
            probability[i][0] = initial_state_transition[i]
            if words[0] in vocab_num_map:
                w_id = vocab_num_map[words[0]]
                if observation_matrix[i][w_id] == "undef":
                    probability[i][0] = "invalid"
                else:
                    probability[i][0] += observation_matrix[i][w_id]

        # print(tag_num_map)
        # print("Initial prob")
        # for p in probability:
        #     print(p)

        for t in range(1, T):
            for q in range(no_of_tags):
                o_t = -1
                if words[t] in vocab_num_map:
                    o_t = vocab_num_map[words[t]]

                probability[q][t] = maxProbability(probability, q, t, o_t)
                backpointer[q][t] = argmaxProbability(probability, q, t)

            # print("TPM***********")
            # for p in probability:
            #     print(p)

        # print("Backpointer")
        # for arr in backpointer:
        #     print(arr)
        # print(tag_num_map)
        # print("Probability")
        # for arr in probability:
        #     print(arr)
        Seq = backtrack(probability, backpointer, T)
        # write to file
        output_string = ""
        for i in range(T):
            output_string += words[i]+"/"+ Seq[i]+" "
        output_string = output_string[:len(output_string)-1] + "\n"
        output_file.write(output_string)

    output_file.close()

if __name__ == '__main__':
    with open("hmmmodel.txt","r") as hmm_model_file:
        hmm_model = json.load(hmm_model_file)
    parseModel(hmm_model)

    with open(sys.argv[1],"r") as test_file:
        test_data = test_file.read()
    HMMdecode(test_data)
