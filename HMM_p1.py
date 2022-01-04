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
