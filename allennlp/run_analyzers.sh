#!/usr/bin/env bash

allennlp analyze https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz > mac.dat
allennlp analyze https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz > cp.dat
allennlp analyze https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gza > elmo.dat
allennlp analyze https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz > srl.dat
allennlp analyze https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz > coref.dat
allennlp analyze https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz > ner.dat

rm allennlp.dat
echo "im M N alpha D minsval" > allennlp.dat
cat mac.dat cp.dat elmo.dat srl.dat coref.dat ner.dat | grep -v "nan" >> allennlp.dat


