# DA401_ASL : CONSTRUCTION PROCESS

## The American Sign Language Analysis in Drive-through & Remote Customer Service Industry

## Overall description: 

Over the last 200 years of development, American Sign Language has become one of the top 5 most popular sign languages being used by nearly 2 million people nowadays (Commission on the Deaf and Hard of Hearing, 2023). Drive-through customer service has impacted the interactions of deaf and hard-to-hearing communities with item-ordering systems. This leads to the birth of different ASL variations and lexical choices among industries such as fast-food franchises, pharmacy clinics, banking teller services, and convenience stores. In this project, I hypothesize that there is a significant difference in ASL gestures among three different drive-through industries: fast-food franchises, pharmaceutical clinics, and banking teller service based on the frequency the words appear in three different corpora.

## Data Resource:

Considering the culture of sign language is based on the word-level interpretation for each of the gestures, the potential source of the dataset that I plan to use to train the data is the Word-Level American Sign Language (WLASL) video dataset (Li, 2020). This new large-scale dataset consists of more than 2000 commonly-used words performed by over 100 signers and is now made publicly for research and academic community (Li et al., 2019). The data helps me in training my Reinforcement Learning model to identify the gestures of the ASL users with higher accuracy and later perform translation with high precision. The research continues with identifying the variation of ASL vocabulary patterns in the drive-through industry by examining the TF-IDF, Word Mover Distance (WMD), and Cosine Similarity. In this process, my approach is based on the live data that I need to collect from the local ASL people around Licking County in order to test the reinforcement learning model in the translating process. The model will translate the ASL videos into text documents having respective information with what ASL users signed previously. Since the collection process consists of human subject information, I also need to conduct the IRB approval in order to proceed with this survey for the research.

## Python Package Prerequisites:
1. Scikit-learn
2. OpenCV2
3. TensorFlow
4. Pytorch
5. Keras
6. Gensim

## Method Approaches: 

*IN-CONSTRUCTION REPORT*
As for right now, I'm working on detailing and debugging the code from WLASL authors to imitate the behavior of the model. The model code is running at the first stage, but the set up for the authors' testing dataset was outdated. Therefore, I need to start working on collecting the data and try to convert my data into a similar original testing dataset.  

The research continues with identifying the variation of ASL vocabulary patterns in the drive-through industry by examining the TF-IDF, Word Mover Distance (WMD), and Cosine Similarity. I am going to put the translated text documents into three different corpora, where each of the corpora will contain 30 - 50 documents having all the tokens. Each term from the translated documents will be represented as a TF-IDF (Term Frequency - Inverse Document Frequency) vector. The TF-IDF vectors will help analyze the frequency of one term appearing in one document (one ASL collection of gestures). With each ASL corpus, after eliminating the stop words, I will look for 20 terms having the highest TF-IDF and turn them into vectors (word embedding the tokens). After each document of 3 corpora is converted into multiple vector representations, the Word Mover Distance (WMD) method is used to calculate the distance between two documents by finding the optimal way to "move" the word embeddings from one document to another using Euclidean Distance. This is done by finding the minimum "cost" of transforming one document into another, where the cost is calculated based on the distance between word embeddings and their respective frequencies in the documents. The distances determined between each vector will determine the difference between a pair of documents. However, my plan is to conduct the Bag of Word (BoW) comparison among different corpora, with each pair of whom involving in paired comparisons. By using the BoW method, I can calculate the word mover distance or the “cost” of moving words from one ASL corpus to another, which can later determine the difference in ASL variation among different corpora. Finally, I plan to use the KNN model to classify the WMD results into different groups of ASL variation and detect the difference in sign language gestures among three corpora. The KNN Classifier helps construct the foundation of the WMD, measuring the difference (distances) among each document across the corpora and producing the final output which helps understand the variation as well as similarities in three different 3 drive-through customer service areas.

## Work Cited:
Li, D. (2020). WLASL: A large-scale dataset for Word-Level American Sign Language (WACV 20’ Best Paper Honourable Mention) [dataset]. dxli94. https://github.com/dxli94/WLASL
Li, D., Opazo, C. R., Yu, X., & Li, H. (2019). Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison. https://doi.org/10.48550/ARXIV.1910.11006
