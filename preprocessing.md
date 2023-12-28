# Preprocessing. 
1. remove any character rather than
   1.  arabic character 
   2.  tashkeel
   3.  commas
   4.  semicolons
   5.  colons
   6.  fullstops. 
2. split on full stops.
   1. if the sentence length was more than 600 chars, we then apply another split on the following:
      1. :
      2. ,
      3. ; 
3. split the unicodes from the characters. 
   1. feh shwyt tashkel byb2o 2 fo2 b3d, zy el shadda w 3leha fat7a, w dama w keda. 
4. some chars will not has any tashkel. 
   1. so we need to create a new class called noDiacratics. 
5. apply tokenization on each sentence. 

6. now our result will be 2 lists 
   1. one for the chars.
   2. the other for the diacritics. 

7. 
