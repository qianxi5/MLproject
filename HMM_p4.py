import copy

def estimate_parameters(path):
    with open(path, "r", encoding="cp437") as f:
        data=f.readlines()
    f.close()
    
    y_states = ['B-positive', 'I-positive', 'B-negative', 'I-negative','B-neutral','I-neutral','O']
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
    for y in y_states:
        out[y+'魑魅魍魉#UNK#'] = k / (y_count.get(y)+k)

    

    return out,set(y_count.keys()),x_set
states = ['START','B-positive', 'I-positive', 'B-negative', 'I-negative','B-neutral','I-neutral','O' ,'STOP']

start = states.copy()
end = states.copy()

start.remove('STOP')
raw_states = start.copy()
end.remove('START')
raw_states.remove('START')


def init_count_uv():
    count_uv ={}
    for i in start:
        for j in start:
            for k in end:
                key=i+j+' '+k
                count_uv[key]=0
    return count_uv
# print(init_count_uv(states))

def init_count_u():
    count_u = {}
    for i in start:
        for j in states:
            key=i+j
            count_u[key]=0
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
            y_ls.append('START')
            continue
        line = line.strip()    
        try:
            x, y = line.split(" ")
        except:
            continue
        y_ls.append(y)
    y_ls.append('STOP')

   

    #print(count_uv.keys())
    # counts
    for j in range(len(y_ls)-2):
        u = y_ls[j]
        v = y_ls[j+1]
        w = y_ls[j+2]
        
        key_uv = u + v
        count_u[key_uv] = count_u.get(key_uv,0) +1
        key_uvw = u+v + " " + w
        #print(key_uvw)
        count_uv[key_uvw] = count_uv.get(key_uvw,0) +1
        
    #count_uv.pop('STOP START') #empty line

    # transition parameters
    for uv_to_uvw in count_uv.keys():
        # print(u_to_v)
        transition_para[uv_to_uvw] = count_uv.get(uv_to_uvw)/ max(count_u.get((uv_to_uvw.split(" ")[0]),0),1)

    #print(transition_para)
    return transition_para


# Viterbi algorithm
import math
def Forward(X, transition, emission, x_set):
    n=len(X)
    scores = {}
    # Base case:
    for w in raw_states:
        key = (1,"START", w)
        if X[0] not in x_set:
            x = "#UNK#"
        else:
            x=X[0]
            
        #print(X[0])
        scores[key] = emission.get((w+'魑魅魍魉'+X[0]),0)*transition.get(("START"+"START"+" "+w),0)
    # Moving forword recursively:
    if(len(X)==1):
        return scores
    
    #level 2
    k=2
    if X[1] not in x_set:
        x = "#UNK#"
    else:
        x=X[1]
            
    u="START"
    for w in end:# each node in position k
        #key = (k,v)
        max_score = []
            
        for v in raw_states:
            max_score=-999
            decide_next=""

            #print(u+v)
            this_score=scores.get((k-1,u,v))

            tmp = this_score * transition.get((u+v+" "+w),0) * emission.get((w+'魑魅魍魉'+x),0)
            if tmp>max_score:
                max_score=tmp
                key=(k,v,w)
                saved_u=u
                saved_v=v
                saved_w=w
                #print("scores:", scores.get((k-1,u)))
                #print("trans:", transition.get((u+" "+v),0))
                #print("emiss:", emission.get((v+'魑魅魍魉'+x),0)) 


            scores[(k,v,w)] = max_score
    
    #level 3 and above
    for k in range(3, n+1):
        
        if X[k-1] not in x_set:
            x = "#UNK#"
        else:
            x=X[k-1]
            
        for w in raw_states:# each node in position k
            #key = (k,v)
            max_score = []
            
            for v in raw_states:
                for u in raw_states:
                    max_score=-999
                    decide_next=""

                    #print(u+v)
                    #print(scores.keys())
                    #print([k,u,v])
                    tmp = scores.get((k-1,u,v)) * transition.get((u+v+" "+w),0) * emission.get((w+'魑魅魍魉'+x),0)
                    if tmp>max_score:
                        max_score=tmp
                        key=(k,v,w)
                        saved_u=u
                        saved_v=v
                        saved_w=w
                    #print("scores:", scores.get((k-1,u)))
                    #print("trans:", transition.get((u+" "+v),0))
                    #print("emiss:", emission.get((v+'魑魅魍魉'+x),0)) 
                scores[(k,v,w)] = max_score
                
    #print(max_score)
    
    #print(scores)
    return scores

def backtracking(X,scores, transition,n):
    if len(X)==1:
        max_yscore=-1
        for w in raw_states:
            tmp=scores.get((1,"START",w))*transition.get(("START"+w+' '+'STOP'),0)
            if tmp > max_yscore:
                max_score = tmp
                yn = w
        return[yn]
        
        
    
    max_yscore = -1
    #yn
    yn=''
    sequence=[]
    for v in raw_states:
        for w in raw_states:
            tmp = scores.get((n, v,w),0) * transition.get((v+w+' '+'STOP'),0)

            if tmp > max_yscore:
                max_score = tmp
                yn = w
    sequence.insert(0,yn)
    # print(sequence)

    # yi
    for i in reversed(range(2,n)):
        max_score = -1
        yi=''
        # print(sequence)
        for v in raw_states:
            for w in raw_states:
                tmp_max = scores.get((i,v,w),0) * transition.get((v+w+' '+sequence[0]),0)
                if tmp_max >= max_score:
                    max_score = tmp_max
                    yi = w
        # print(yi)
        sequence.insert(0,yi)
    
    # i=1
    v="START"
    for w in raw_states:
        tmp_max = scores.get((1,v,w)) * transition.get((v+w+' '+sequence[0]),0)
        if tmp_max >= max_score:
            max_score = tmp_max
            yi = w
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
        sequence_small = backtracking(each,scores, transition,len(each))
        # print(sequence)
        #print(len(sequence))
        #print(sequence_small)
        sequence.extend(sequence_small)
        sequence.append("\n_pass")
    

    #print("_____________")
    #print(len(lines))
    #print(len(sequence))
    #print(n)
    for i in range(n):
        if lines[i] == "":
            output.append(lines[i]+'\n')
        elif lines[i] == '\n':
            output.append(lines[i])
        else:
            output.append(lines[i].strip()+' '+sequence[i]+'\n')
    
    return output


ES_transition_para = estimate_transition("ES/train")
ES_emission_para,ES_y,ES_xset = estimate_parameters("ES/train")
RU_transition_para = estimate_transition("RU/train")
RU_emission_para,_,RU_xset = estimate_parameters("RU/train")

# Run on dev datasets
ES_out = viterbi('ES/dev.in', ES_transition_para, ES_emission_para,ES_xset)
RU_out = viterbi('RU/dev.in', RU_transition_para, RU_emission_para, RU_xset)

# write outputs to dev.p4.out
with open("ES/dev.p4.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in ES_out:
        g.write(each)
    g.close()
print('ES p4 done')

with open("RU/dev.p4.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in RU_out:
        g.write(each)
    g.close()
print('RU p4 done')


# Run on test datasets
ES_test_out = viterbi('ES-test/test.in', ES_transition_para, ES_emission_para,ES_xset)
RU_test_out = viterbi('RU-test/test.in', RU_transition_para, RU_emission_para, RU_xset)

# write outputs to test.p4.out
with open("ES/test.p4.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in ES_test_out:
        g.write(each)
    g.close()
print('ES test p4 done')

with open("RU/test.p4.out", "w",  encoding="cp437", errors='ignore') as g:
    for each in RU_test_out:
        g.write(each)
    g.close()
print('RU test p4 done')