# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 08:53:11 2016

@author: ductr
"""

import pandas as pd
import preprocessing as pp

training_data, test_data, raw_training, raw_test, raw = pp.load()
data_new = pd.concat((training_data, test_data))


base_directory = "F:\\code\\python\\lvtn\\"
socal = pd.read_csv(base_directory + "so-cal.csv")

old = socal[socal.id<500]
old.index = range(old.shape[0])

new = socal[socal.id>=500]
new.index = range(new.shape[0])

b=[]
g=[]
m=[]
l=[]
def training_change_phrase(corpus):
    BAD = ["suffer", "adverse", "hazards", "risk", "death", "insufficient",
           "infection", "recurrence", "restlessness", "mortality", "hazard",
           "chronic", "pain", "negative", "severity","complication","risk","adverse","mortality","morbidity","death","fatal","danger","no benefit","discourage","short-term risk","long-term risk","damage","little information","not been well studies","ineffective","suffer","depression","acute","sore","outpatient","disabling","diabetes","difficulties","dysfunction","distorted","poorer","unable","prolonged","irritation","disruptive","pathological","mutations","disease","infection","harms","difficulty","weakened","inactive","stressors","hypertension","adverse","insomnia","relapsing","malignant","suffer","exacerbate","dryness","fever","overestimate","constipation","deposition","colic","tension","hazards","diarrhoea","weakness","irritability","insidious","distress","weak","cancer","emergency","risk","block","unsatisfactory ","blinding","nausea","traumatic","wound","intention","loses","intensive","relapse","recurrent","extension","die","cancers","malaise","crying","toxic","injury","confounding","complaints","misuse","insignificant","poisoning","anoxic","amputation","death","nightmares","deteriorate","fatal","injuries","fatigue","invasive","suicide","chronic","relapsed","disturbances","confusion","died","fluctuating","severities","delusions","compulsions","conflict","trauma","cried","impair","severe","tremor","weaker","illness","inpatients","worry","rebound","worse","reversible","dizziness","attacks","pointless","disorders","dyskinesia","risks","fatty","negative","conflicting","upset","fishy","hard","harm","bleeding","inflammatory","hampered","underpowered","obstruction","headache","problem","bleeds","panic","loss","odds","retardation","dysfunctional","render","difficult","drowsiness","lack","suicidal","obsessions","impaired","cough","severity","suffering","violent","strokes","virus","stroke","flatulence","fibrates","blind ","burning ","faintness","suffered","threatening","misdiagnosing","bitter","excessive","diabetics","malfunction","abnormal","deterioration","bad","confounded","sadness","mortality","disturbance","agitated","attack","infections","negativistic","deaths","poor","wrong","worsening","adversely","insufficient","scarring","headaches","disability","overdose ","serious","delayed","discomfort","sweating","morbidity","nerve","parkinson","toxicity","nervous","pain","stress","weakens","incorrect","disorder","worsened","malformations","blinded","rigidity","prolong","adversity","abuse","lacked","dyspepsia","sads ","onset","failure","inadequate","sensitivity","impairment","dementia","harmful"]
    GOOD = ["benefit", "improvement", "advantage", "accuracy", "great",
            "effective", "support", "potential", "superior", "mild", "achieved",
           "Supplementation", "beneficial", "positive","benefit","beneficial","improve","advantage","resolve","good","fantastic","relief","superior","efficacious","effective","improve effectiveness","importance of protecting","significant advantage","significant therapeutic advantage","may be effective","effective approach","simple and effective","simple and effective treatment","safe","well tolerated","well-tolerated","useful","maybe useful","illustrate the benefits","significant improvement","significantly improve","clinically worthwhile","worthwhile","recover rapid","satisfactory outcome","satisfactory","similarly effective","supports","approve","more effective","high efficacy","cured","vitality","relaxing","benefit","tolerability","improvement","right","effective","stable","best","better","pleasurable","relaxation","favour","beneficial","safety","prevents","successful","satisfaction","significant","superior","contributions","reliability","robust","tolerated","improving","survival","favourable","reliable","recovered","judiciously","consciousness","efficacy","prevented","satisfied","prevent","advantage","encouraging","tolerance","success","significance","improved","improves","improve","improvements"]
    MORE = ["enhance", "higher", "exceed", "increase", "improve", "somewhat",
            "quite", "very", "higher", "more", "augments", "highest","enhance","augment","increase","amplify","raise","boost","add to","higher","exceed","rise","go up","surpass","more","additional","extra","added","greater","positive","high","prolonged","prolong","increase","enhance","elevation","higher","exceed","enhancement","peaked","more","excess"]
    LESS = ["reduce", "decline", "fall", "less", "little", "slightly", "only", 
            "mildly", "smaller", "lower", "reduction", "drop","fewer","slump",
            "fall","down","pummel","less","lower","low","decrease","reduce",
            "decline","descend","collapse","fail","subside","lesser","poorer",
            "Worse","smaller","negative","prevent","reduced","prevents","below",
            "lower","decrease","fall","low","reduce","decline","less","little",
            "mild","drop","fewer"]
    
    BAD = preprocessing.stemming(preprocessing.lemmatization(BAD))
    GOOD = preprocessing.stemming(preprocessing.lemmatization(GOOD))
    MORE = preprocessing.stemming(preprocessing.lemmatization(MORE))
    LESS = preprocessing.stemming(preprocessing.lemmatization(LESS))
    
    def sen2vec(sen):
        words = sen.split(' ')
        vecs = [0,0,0,0] #MORE GOOD, MORE BAD, LESS GOOD, LESS BAD
        for i in range(len(words)):
            if words[i] in MORE:
                m.append(words[i])
                #print 'more='+words[i]
                for k in range(i, len(words)):
                    if words[k] in GOOD:
                        g.append(words[k])
                        vecs[0] = 1
                        break
                    if words[k] in BAD:
                        b.append(words[k])
                        vecs[1] = 1
                        break
            elif words[i] in LESS:
                l.append(words[i])
                for k in range(i, len(words)):
                    if words[k] in GOOD:
                        g.append(words[k])
                        vecs[2] = 1
                        break
                    if words[k] in BAD:
                        b.append(words[k])
                        vecs[3] = 1
                        break
        return vecs
    result = [sen2vec(sen) for sen in corpus]
    return result

neg_bin = map(int, all['neg']!=0)
all['neg_bin'] = neg_bin
all_pos = all[all.lab==2]
all_neu = all[all.lab==1]
all_neg = all[all.lab==0]

plt.figure()
plt.subplot(1,3,1)
plt.hist(list(all_pos['neg_bin']))
plt.subplot(1,3,2)
plt.hist(list(all_neu['neg_bin']))
plt.subplot(1,3,3)
plt.hist(list(all_neg['neg_bin']))