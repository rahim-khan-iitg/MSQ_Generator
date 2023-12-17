# download and install pke link is in the notebook
# import the required libraries
# here i have used huggingface model google-flan-t5-large
# it may give some time key error but try to run it again
import random
import spacy
import time
from transformers import pipeline
import requests
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
import pke
import traceback
import requests
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"
headers = {"Authorization": "Bearer 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"}

# example context
text=""" Photosynthesis is a process used by plants and other organisms to convert light energy into
 chemical energy that, through cellular respiration, can later be released to fuel the organism's
 activities. Some of this chemical energy is stored in carbohydrate molecules, such as sugars and
 starches, which are synthesized from carbon dioxide and water– hence the name photosynthesis,
 from the Greek phōs, "light", and synthesis , "putting together". Most plants, algae, and
 cyanobacteria perform photosynthesis; such organisms are called photoautotrophs. Photosynthesis
 is largely responsible for producing and maintaining the oxygen content of the Earth's atmosphere,
 and supplies most of the energy necessary for life on Earth.
 Although photosynthesis is performed differently by different species, the process always begins
 when energy from light is absorbed by proteins called reaction centers that contain green chlorophyll
 (and other colored) pigments/chromophores. In plants, these proteins are held inside organelles
 called chloroplasts, which are most abundant in leaf cells, while in bacteria they are embedded in
 the plasma membrane. In these light-dependent reactions, some energy is used to strip electrons
 from suitable substances, such as water, producing oxygen gas. The hydrogen freed by the splitting
 of water is used in the creation of two further compounds that serve as short-term stores of energy,
 enabling its transfer to drive other reactions: these compounds are reduced nicotinamide adenine
 dinucleotide phosphate (NADPH) and adenosine triphosphate (ATP), the "energy currency" of cells.
 In plants, algae and cyanobacteria, sugars are synthesized by a subsequent sequence of
 light-independent reactions called the Calvin cycle. In the Calvin cycle, atmospheric carbon dioxide
 is incorporated into already existing organic carbon compounds, such as ribulose bisphosphate
 (RuBP).[5] Using the ATP and NADPH produced by the light-dependent reactions, the resulting
 compounds are then reduced and removed to form further carbohydrates, such as glucose. In other
 bacteria, different mechanisms such as the reverse Krebs cycle are used to achieve the same end.
 The first photosynthetic organisms probably evolved early in the evolutionary history of life and most
 likely used reducing agents such as hydrogen or hydrogen sulfide, rather than water, as sources of
 electrons. Cyanobacteria appeared later; the excess oxygen they produced contributed directly to
 the oxygenation of the Earth, which rendered the evolution of complex life possible. Today, the
 average rate of energy capture by photosynthesis globally is approximately 130 terawatts, which is
 about eight times the current power consumption of human civilization. Photosynthetic organisms
 also convert around 100–115 billion tons (91–104 Pg petagrams, or billion metric tons), of carbon
 into biomass per year. That plants receive some energy from light– in addition to air, soil, and water– was first discovered in 1779 by Jan Ingenhousz."""


# here i am using multipartite rank algorithm to extract the important keywords
# this algorithm is present in pke library. it depends on spacy.

def get_nouns_multipartite(content):
    out=[]
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content,language='en')
        pos = {'PROPN','NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1,threshold=0.75,method='average')
        keyphrases = extractor.get_n_best(n=15)
        
        for val in keyphrases:
            out.append(val[0])
    except:
        out = []
        traceback.print_exc()
    return out

# function for fetching data from the model

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

# Load the spaCy model
nlp = spacy.load("en_core_web_md")

# rank similarity
def get_most_similar_words(keywords:list,word:str,n:int)->list:
    ans=dict()
    for keyword in keywords:
        token1=nlp(keyword)
        token2=nlp(word)
        similarity=token1.similarity(token2)
        ans[similarity]=token1
    ans=[str(x) for y,x in sorted(ans.items(),reverse=True)]
    return ans[0:n]

# sometimes it might happen that it returns a keyerror:0 becuase of the model used  but try to run it again
def get_mca_questions(context:str):
    if not isinstance(context,str):
        return ["please provide valid context"]
    # load model
    response=query({"inputs":"do sommething"})
    # print(response)
    count=0
    while(response[0].get('generated_text')==None):
        time.sleep(3)
        response=query({"inputs":"do sommething"})
        count=count+1
        if(count>4):
            return["model not working"]
    
    # initialize the answer
    # here we are generating 4 mcas

    # get the 10 important keywords
    keywords=get_nouns_multipartite(context) 
    # initialize the answer
    mca_questions=[]
    no_questions=4
    selected_unique_keywords=set()
    while(len(selected_unique_keywords)!=no_questions):
        key=random.choice(keywords)
        selected_unique_keywords.add(key)
    
    # generate question for each choice
    for i,keyword in enumerate(selected_unique_keywords,1):
        # get four similar keywords from the keywords for the current keyword
        options=get_most_similar_words(keywords,keyword,4)
        no_correct_ans=random.randint(1,2)
        correct_ans=options[0:no_correct_ans]
        if(len(correct_ans)==1):
            correct_ans=correct_ans[0]
        else:
            correct_ans=",".join(correct_ans)
        # get the question from the generative model
        response=query({"inputs":f"generate question for the answers {correct_ans} from the context:{context}"})
        question=response[0].get("generated_text")
        random.shuffle(options)
        # it might happen that we didnot have the most relevant and correct ans in the options
        ans_query={"inputs":f"Answer the question:{question}. from the context:{context}"}
        output=query(ans_query)
        ans=output[0].get('generated_text')
        if(ans not in correct_ans.split(",")):
            options.insert(random.randint(0,3),ans)
            correct_ans=correct_ans+","+ans
        # form the question
        q=f"Q({i}) {question}? (a){options[0]} (b){options[1]} (c){options[2]} (d) {options[3]} answer : {correct_ans}"
        mca_questions.append(q)
    return mca_questions


if __name__=="__main__":
    print("install dependency pke first if not install ")
    print("some times it may give key error please run it again ")
    print(get_mca_questions(text))