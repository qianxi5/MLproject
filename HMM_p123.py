import math
import collections

#---------------------------------part 1------------------------------------

def estimate_parameters_without_UNK(path):
    with open(path, "r", encoding="cp437") as f:
        data=f.readlines()
    f.close()
    
    y_states = ['B-positive', 'I-positive', 'B-negative', 'I-negative','B-neutral','I-neutral','O']

    out = {}
    transition_count = {}
    y_count = {}
    x_set=set()

    for i in range(len(data)):
        line = data[i]
        line = line.strip()
        try:
            x, y = line.split(" ")
        except:
            continue
                
        key=y+'魑魅魍魉'+x
        x_set.add(x)
        #print(key)
        transition_count[key] = transition_count.get((key), 0) + 1
        y_count[y] = y_count.get(y, 0)+1
                
    #Count the output
    for yx in transition_count.keys():
        out[yx]=(transition_count.get(yx) / (y_count.get((yx.split('魑魅魍魉')[0]),0)))

def estimate_parameters(path):
    with open(path, "r", encoding="cp437") as f:
        data=f.readlines()
    f.close()
    
    k=1
    out = {}
    transition_count = {}
    y_count = {}
    x_set=set()

    for i in range(len(data)):
        line = data[i]
        line = line.strip()
        try:
            x, y = line.split(" ")
        except:
            continue
            
        key=y+'魑魅魍魉'+x
        x_set.add(x)
        #print(key)
        transition_count[key] = transition_count.get((key), 0) + 1
        y_count[y] = y_count.get(y, 0)+1
            
    #fix the token
    for yx in transition_count.keys():
        out[yx]=(transition_count.get(yx) / (y_count.get((yx.split('魑魅魍魉')[0]),0) + k))
    for y in y_count.keys():
        out[y+'魑魅魍魉#UNK#'] = k / (y_count.get(y)+k)

    

    return out,set(y_count.keys()),x_set

# Prediction part 1
#RU part
#training
emission_c, obs_y, x_set = estimate_parameters('RU/train')
#print(obs_y)
#print(emission_c)
#evaluate using dev.in
with open("RU/dev.in", "r", encoding="cp437", errors='ignore') as f:
    lines = f.readlines()
    output = []
    all_prediction = []


    for line in lines:        
        if line == "":
            output.append((line+"\n"))
            
        elif line=="\n":
            output.append(line)
        else:
            predicted_state = ""
            highest_prob = -1
            
            line = line.strip()
            if line not in x_set:
                x = "#UNK#"
            else:
                x=line
                
            for y in obs_y:
                emission_prob = emission_c.get((y+"魑魅魍魉"+x),0)

                if emission_prob > highest_prob:
                    highest_prob = emission_prob
                    predicted_state = y

            output.append((line+' '+predicted_state+"\n"))

with open("RU/dev.p1.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in output:
        g.write(each)
    g.close()
print("RU p1 done")
    

#ES part
emission_c, obs_y, x_set = estimate_parameters('ES/train')
#print(obs_y)

#evaluate using dev.in
with open("ES/dev.in", "r", encoding="cp437", errors='ignore') as f:
    lines = f.readlines()
    output = []
    all_prediction = []


    for line in lines:        
        if line == "":
            output.append((line+"\n"))
            
        elif line=="\n":
            output.append(line)
        else:
            predicted_state = ""
            highest_prob = -1
            
            line = line.strip()
            if line not in x_set:
                x = "#UNK#"
            else:
                x=line
                
            for y in obs_y:
                emission_prob = emission_c.get((y+"魑魅魍魉"+x),0)

                if emission_prob > highest_prob:
                    highest_prob = emission_prob
                    predicted_state = y

            output.append((line+' '+predicted_state+"\n"))

with open("ES/dev.p1.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in output:
        g.write(each)
    g.close()
print("ES p1 done")


#--------------------------------part 2------------------------------------------

states = ['START','B-positive', 'I-positive', 'B-negative', 'I-negative','B-neutral','I-neutral','O' ,'STOP']
start = states.copy()
end = states.copy()

start.remove('STOP')
raw_states = start.copy()
end.remove('START')
raw_states.remove('START')

# estimate transition parameters
def init_count_uv():
    count_uv ={}
    for i in start:
        for j in end:
            count_uv[i+' '+j]=0
    return count_uv
# print(init_count_uv(states))

def init_count_u():
    count_u = {}
    for i in states:
        count_u[i]=0
    return count_u

def estimate_transition(filepath):
    with open(filepath, "r", encoding="cp437") as f:
        data=f.readlines()
    f.close()

    count_uv = init_count_uv()
    count_u = init_count_u()
    transition_para = {}

    # get labels
    y_ls = ['START']
    for i in range(len(data)):
        line = data[i]
         
        if line=='\n':
            y_ls.append('STOP')
            y_ls.append('START')
            continue
        line = line.strip()    
        try:
            x, y = line.split(" ")
        except:
            continue
        y_ls.append(y)
    y_ls.append('STOP')

   

    # counts
    for j in range(len(y_ls)-1):
        u = y_ls[j]
        v = y_ls[j+1]
        count_u[u] = count_u.get(u,0) +1
        key = u +" " + v
        count_uv[key] = count_uv.get(key,0) +1
        
    count_uv.pop('STOP START') #empty line

    # transition parameters
    for u_to_v in count_uv.keys():
        # print(u_to_v)
        transition_para[u_to_v] = count_uv.get(u_to_v)/ count_u.get((u_to_v.split(" ")[0]),0)

    return transition_para

def Forward(X, transition, emission, x_set):
    n=len(X)
    scores = {}
    # Base case:
    for v in states:
        key = (0, v)
        if v == 'START': scores[key] = 1
        else: scores[key] = 0
    # Moving forword recursively:
    for k in range(1, n+1):
        if X[k-1] not in x_set:
            x = "#UNK#"
        else:
            x=X[k-1]
        for v in states:# each node in position k
            key = (k,v)
            max_score = []
            for u in start: #find max score over state set.
                # print("scores:", scores.get((k-1,u)))
                # print("trans:", transition.get((u+" "+v),0))
                # print("emiss:", emission.get((v+'魑魅魍魉'+x),0)) 

                tmp = scores.get((k-1,u)) * transition.get((u+" "+v),0) * emission.get((v+'魑魅魍魉'+x),0)
                # print(tmp)
                max_score.append(tmp)
            scores[key] = max(max_score)
    # print(scores)
    return scores

def backtracking(scores, transition,n):
    max_yscore = -1
    #yn
    yn=''
    sequence=[]
    for v in raw_states:
        tmp = scores.get((n, v)) * transition.get((v+' '+'STOP'),0)
        if tmp > max_yscore:
            max_score = tmp
            yn = v
    sequence.insert(0,yn)
    # print(sequence)

    # yi
    for i in reversed(range(1,n)):
        max_score = -1
        yi=''
        # print(sequence)
        for u in raw_states:
            tmp_max = scores.get((i,u)) * transition.get((u+' '+sequence[0]),0)
            if tmp_max >= max_score:
                max_score = tmp_max
                yi = u
        # print(yi)
        sequence.insert(0,yi)
    return sequence

def viterbi(path, transition, emission, x_set):
    with open(path, "r", encoding="cp437") as f:
        lines = f.readlines()
    f.close()

    #print(lines)
    output = []
    n = len(lines)
    X=[]
    # print(lines[5559])
    # print(n)
    
    temp=[]

    for line in lines:
        if line !="\n":
            line = line.strip()
            temp.append(line)
        else:
            X.append(temp)
            temp=[]
            
    #print(X)
    sequence=[]
    for each in X:

        scores = Forward(each, transition, emission, x_set)
        # print(len(scores))
        sequence_small = backtracking(scores, transition,len(each))
        # print(sequence)
        #print(len(sequence))
        #print(sequence_small)
        sequence.extend(sequence_small)
        sequence.append("\n_pass")

    #print(n)
    for i in range(n):
        if lines[i] == "":
            output.append(lines[i]+'\n')
        elif lines[i] == '\n':
            output.append(lines[i])
        else:
            output.append(lines[i].strip()+' '+sequence[i]+'\n')
    
    return output


#------------------------------Part 3------------------------------------------

## part 3
# Viterbi algorithm

def find_ith(scores, i):
  return sorted(scores, reverse=True)[i-1]

global_dict = {}

def Forward_5th(X, transition, emission, x_set):
    n=len(X)
    scores = {}
    # Base case:
    for v in states:
        for i in range(1,6):
            key = (0, v, i)
            if v == 'START': scores[key] = 1
            else: scores[key] = 0
    # Moving forword recursively:
    # k = next position
    # v = next state
    # i = ith best
    
    # Start state, deal with the second state
    k = 1
    if X[k-1] not in x_set:
        x = "#UNK#"
    else:
        x=X[k-1]
    for v in states:
        # each node in position k
        # we need to find all possible choices
        score = []
        for u in start: # each node in position k-1
            tmp = scores.get((k-1,u,1)) * transition.get((u+" "+v),0) * emission.get((v+'魑魅魍魉'+x),0)
            score.append(tmp)
            # Now the score list saves all possible choices from k-1 to k's v state   
        # Here we should record 1st to 5th choices for node k
        for i in range(1, 6):
          # print(score)
          scores[(k,v,i)] = find_ith(score, i)

    for k in range(2, n+1):# Position
        if X[k-1] not in x_set:
            x = "#UNK#"
        else:
            x=X[k-1]
        for v in states:
            # each node in position k
            # we need to find all possible choices
            score = []
            for u in start: # each node in position k-1
                # Here we should find 5th best for each of next state
                for i in range(1, 6):
                    tmp = scores.get((k-1,u,i)) * transition.get((u+" "+v),0) * emission.get((v+'魑魅魍魉'+x),0)
                    score.append(tmp)
                # Now the score list saves all possible choices from k-1 to k's v state   
            # Here we should record 1st to 5th choices for node k
            for i in range(1, 6):
              # print(score)
              scores[(k,v,i)] = find_ith(score, i)
    #print(scores)
    return scores

def backtracking_5th(scores, transition,n):
    max_yscore = -1
    #yn
    # yn=''
    sequence=[]

    tag_dict = {}
    for v in raw_states: # For each nodes in the last column, 
        # find out which nodes has global 5th best value
        for i in range(1,6):
            tmp = scores.get((n,v,i)) * transition.get((v+' '+'STOP'),0)
            # if tmp > max_yscore:
            #     max_score = tmp
            #     yn = v
            tag_dict[tmp] = (n,v,i)
    yn_sorted=list(collections.OrderedDict(sorted(tag_dict.items(), reverse=True)[:5]).values())[-1]
    sequence.insert(0,yn_sorted[1])
    # print(sequence)

    # yi
    for ni in reversed(range(1,n)):
        # print("N: "+str(ni))
        # max_score = -1
        # yi=''
        # print(sequence)
        tag_dict = {}
        # tag_dict[0.0] = (ni, "0", )
        for u in raw_states:
            # print(u)
            for i in range(1,6):
                tmp = scores.get((ni,u,i)) * transition.get((u+' '+sequence[0]),0)
                # if tmp_max >= max_score:
                #     max_score = tmp_max
                #     yi = u
                tag_dict[tmp] = (ni,u,i)
        # print(tag_dict)
        yi_sorted=list(collections.OrderedDict(sorted(tag_dict.items(), reverse=True)[:5]).values())[-1]
        # print(yi_sorted)
        sequence.insert(0,yi_sorted[1])
    return sequence

def viterbi_5th(path, transition, emission, x_set):
    with open(path, "r", encoding="cp437") as f:
        lines = f.readlines()
    f.close()

    #print(lines)
    output = []
    n = len(lines)
    X=[]
    # print(lines[5559])
    print(n)
    
    temp=[]

    for line in lines:
        if line !="\n":
            line = line.strip()
            temp.append(line)
        else:
            X.append(temp)
            temp=[]
            
    #print(X)
    sequence=[]
    for each in X:

        scores = Forward_5th(each, transition, emission, x_set)
        # print(len(scores))
        sequence_small = backtracking_5th(scores, transition,len(each))
        # print(sequence)
        #print(len(sequence))
        #print(sequence_small)
        sequence.extend(sequence_small)
        sequence.append("\n_pass")

    #print(n)
    for i in range(n):
        if lines[i] == "":
            output.append(lines[i]+'\n')
        elif lines[i] == '\n':
            output.append(lines[i])
        else:
            output.append(lines[i].strip()+' '+sequence[i]+'\n')
    
    return output

#==========================Prediction================================

# part2
ES_transition_para = estimate_transition("ES/train")
ES_emission_para,ES_y,ES_xset = estimate_parameters("ES/train")
RU_transition_para = estimate_transition("RU/train")
RU_emission_para,_,RU_xset = estimate_parameters("RU/train")


ES_p2_out = viterbi('ES/dev.in', ES_transition_para, ES_emission_para,ES_xset)
RU_p2_out = viterbi('RU/dev.in', RU_transition_para, RU_emission_para, RU_xset)

# write outputs to dev.p2.out
with open("ES/dev.p2.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in ES_p2_out:
        g.write(each)
    g.close()
print('ES p2 done')

with open("RU/dev.p2.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in RU_p2_out:
        g.write(each)
    g.close()
print('RU p2 done')

# part 3
ES_p3_out = viterbi_5th('ES/dev.in', ES_transition_para, ES_emission_para,ES_xset)
RU_p3_out = viterbi_5th('RU/dev.in', RU_transition_para, RU_emission_para, RU_xset)

# write outputs to dev.p3.out
with open("ES/dev.p3.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in ES_p2_out:
        g.write(each)
    g.close()
print('ES p3 done')

with open("RU/dev.p3.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in RU_p3_out:
        g.write(each)
    g.close()
print('RU p3 done')