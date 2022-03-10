# -*- coding: utf-8 -*-
"""
env sbert
"""
import sys
from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer('all-mpnet-base-v2')

from sentence_transformers.util import cos_sim


    
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, \
                         DPRQuestionEncoder, DPRQuestionEncoderTokenizer
                         
ctx_model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')

question_model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')       

questions = [
    "what is the capital city of australia?",
    "what is the best selling sci-fi book?",
    "how many searches are performed on Google?"
]

contexts = [
    "canberra is the capital city of australia",
#    "what is the capital city of australia?",
    "the capital city of france is paris",
#    "what is the best selling sci-fi book?",
    "sc-fi is a popular book genre read by millions",
    "the best-selling sci-fi book is dune",
#    "how many searches are performed on Google?",
    "Google serves more than 2 trillion queries annually",
    "Google is a popular search engine"
]    

str1='''

1.1 What is Testing? 

Software systems are an integral part of life, from business applications (e.g., banking) to consumer 
products (e.g., cars). Most people have had an experience with software that did not work as expected. 
Software that does not work correctly can lead to many problems, including loss of money, time, or 
business reputation, and even injury or death. Software testing is a way to assess the quality of the 
software and to reduce the risk of software failure in operation. 

A common misperception of testing is that it only consists of running tests, i.e., executing the software 
and checking the results. As described in section 1.4, software testing is a process which includes many 
different activities; test execution (including checking of results) is only one of these activities. The test 
process also includes activities such as test planning, analyzing, designing, and implementing tests, 
reporting test progress and results, and evaluating the quality of a test object.  

Some testing does involve the execution of the component or system being tested; such testing is called 
dynamic testing. Other testing does not involve the execution of the component or system being tested; 
such testing is called static testing. So, testing also includes reviewing work products such as 
requirements, user stories, and source code.  

Another common misperception of testing is that it focuses entirely on verification of requirements, user 
stories, or other specifications. While testing does involve checking whether the system meets specified 
requirements, it also involves validation, which is checking whether the system will meet user and other 
stakeholder needs in its operational environment(s). 

Test activities are organized and carried out differently in different lifecycles (see section 2.1). 

1.1.1 Typical Objectives of Testing 

For any given project, the objectives of testing may include:  

 To prevent defects by evaluate work products such as requirements, user stories, design, and 
code  

 To verify whether all specified requirements have been fulfilled  

 To check whether the test object is complete and validate if it works as the users and other 
stakeholders expect 

 To build confidence in the level of quality of the test object 
'''

str2='''
To find defects and failures thus reduce the level of risk of inadequate software quality 

 To provide sufficient information to stakeholders to allow them to make informed decisions, 
especially regarding the level of quality of the test object 

 To comply with contractual, legal, or regulatory requirements or standards, and/or to verify the 
test object’s compliance with such requirements or standards 

The objectives of testing can vary, depending upon the context of the component or system being tested, 
the test level, and the software development lifecycle model. These differences may include, for example: 

 During component testing, one objective may be to find as many failures as possible so that the 
underlying defects are identified and fixed early. Another objective may be to increase code 
coverage of the component tests.  

 During acceptance testing, one objective may be to confirm that the system works as expected 
and satisfies requirements. Another objective of this testing may be to give information to 
stakeholders about the risk of releasing the system at a given time. 

'''

str3='''
1.1.2 Testing and Debugging 

Testing and debugging are different. Executing tests can show failures that are caused by defects in the 
software. Debugging is the development activity that finds, analyzes, and fixes such defects. Subsequent 
confirmation testing checks whether the fixes resolved the defects. In some cases, testers are 
responsible for the initial test and the final confirmation test, while developers do the debugging, 
associated component and component integration testing (continues integration). However, in Agile 
development and in some other software development lifecycles, testers may be involved in debugging 
and component testing.  

ISO standard (ISO/IEC/IEEE 29119-1) has further information about software testing concepts. 

1.2 Why is Testing Necessary? 

Rigorous testing of components and systems, and their associated documentation, can help reduce the 
risk of failures occurring during operation. When defects are detected, and subsequently fixed, this 
contributes to the quality of the components or systems. In addition, software testing may also be 
required to meet contractual or legal requirements or industry-specific standards. 

1.2.1 Testing’s Contributions to Success 

Throughout the history of computing, it is quite common for software and systems to be delivered into 
operation and, due to the presence of defects, to subsequently cause failures or otherwise not meet the 
stakeholders’ needs. However, using appropriate test techniques can reduce the frequency of such 
problematic deliveries, when those techniques are applied with the appropriate level of test expertise, in 
the appropriate test levels, and at the appropriate points in the software development lifecycle. Examples 
include:  

 Having testers involved in requirements reviews or user story refinement could detect defects in 
these work products. The identification and removal of requirements defects reduces the risk of 
incorrect or untestable features being developed.  

 Having testers work closely with system designers while the system is being designed can 
increase each party’s understanding of the design and how to test it. This increased 
understanding can reduce the risk of fundamental design defects and enable tests to be identified 
at an early stage. 
'''

str4='''
Having testers work closely with developers while the code is under development can increase 
each party’s understanding of the code and how to test it. This increased understanding can 
reduce the risk of defects within the code and the tests.  

 Having testers verify and validate the software prior to release can detect failures that might 
otherwise have been missed, and support the process of removing the defects that caused the 
failures (i.e., debugging). This increases the likelihood that the software meets stakeholder needs 
and satisfies requirements.  

In addition to these examples, the achievement of defined test objectives (see section 1.1.1) contributes 
to overall software development and maintenance success. 

1.2.2 Quality Assurance and Testing 

While people often use the phrase quality assurance (or just QA) to refer to testing, quality assurance and 
testing are not the same, but they are related. A larger concept, quality management, ties them together. 
Quality management includes all activities that direct and control an organization with regard to quality. 
Among other activities, quality management includes both quality assurance and quality control. Quality 
assurance is typically focused on adherence to proper processes, in order to provide confidence that the 
appropriate levels of quality will be achieved. When processes are carried out properly, the work products 
created by those processes are generally of higher quality, which contributes to defect prevention. In 
addition, the use of root cause analysis to detect and remove the causes of defects, along with the proper 
application of the findings of retrospective meetings to improve processes, are important for effective 
quality assurance. 

Quality control involves various activities, including test activities, that support the achievement of 
appropriate levels of quality. Test activities are part of the overall software development or maintenance 
process. Since quality assurance is concerned with the proper execution of the entire process, quality 
assurance supports proper testing. As described in sections 1.1.1 and 1.2.1, testing contributes to the 
achievement of quality in a variety of ways. 

1.2.3 Errors, Defects, and Failures 

A person can make an error (mistake), which can lead to the introduction of a defect (fault or bug) in the 
software code or in some other related work product. An error that leads to the introduction of a defect in 
one work product can trigger an error that leads to the introduction of a defect in a related work product. 
For example, a requirements elicitation error can lead to a requirements defect, which then results in a 
programming error that leads to a defect in the code. 

If a defect in the code is executed, this may cause a failure, but not necessarily in all circumstances. For 
example, some defects require very specific inputs or preconditions to trigger a failure, which may occur 
rarely or never.  

Errors may occur for many reasons, such as: 

 Time pressure  

 Human fallibility 

 Inexperienced or insufficiently skilled project participants 

 Miscommunication between project participants, including miscommunication about requirements 
and design 

 Complexity of the code, design, architecture, the underlying problem to be solved, and/or the 
technologies used  

 Misunderstandings about intra-system and inter-system interfaces, especially when such intra-
system and inter-system interactions are large in number 

 New, unfamiliar technologies  
'''

contexts2=[str1,str2,str3,str4]


questions2 = [
    "what are the causes of errors ?",
    "is testing necessary ?",
    "what does testing consists of ?",
    "what are the causes of errors ?",
    "What is the longest river in Europe ?",
    "Who is the president of USA ?",
    "what is the capital city of australia?",
    "what is the best selling sci-fi book?",
    "how many searches are performed on Google?"
]




xb_tokens = ctx_tokenizer(contexts2, max_length=512, padding='max_length',
                          truncation=True, return_tensors='pt')
xb = ctx_model(**xb_tokens)


import time

for j in range(1):    
    t0 = time.time()    

    xq_tokens = question_tokenizer(questions2, max_length=512, padding='max_length',
                                   truncation=True, return_tensors='pt')
    
    xq = question_model(**xq_tokens)
    
    xb.pooler_output.shape, xq.pooler_output.shape
    

    
    for i, xq_vec in enumerate(xq.pooler_output):
        probs = cos_sim(xq_vec, xb.pooler_output)
        argmax = torch.argmax(probs)
        print(probs)
        print('QQQQQQQQQQ')
        print(questions2[i])
        print('AAAAAAAAAAA')
        print(contexts2[argmax])
        print('---')    
        
    t1 = time.time()

    total = t1-t0
    
    print("time exe")
    print(total)
    
    
print(" You can ask me 5 questions!!!\n\n")    
    
for j in range(5):    
    questions2 = input("What is your question? I a waiting  ")
    print(questions2)
    xq_tokens = question_tokenizer(questions2, max_length=512, padding='max_length',
                                   truncation=True, return_tensors='pt')
    
    xq = question_model(**xq_tokens)
    
    

    
    for i, xq_vec in enumerate(xq.pooler_output):
        probs = cos_sim(xq_vec, xb.pooler_output)
        argmax = torch.argmax(probs)
        print(probs)
        print('QQQQQQQQQQ')
        print(questions2)
        print('AAAAAAAAAAA')
        print(contexts2[argmax])
        print('---')    
        

          