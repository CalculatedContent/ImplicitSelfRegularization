#!/usr/bin/env bash

allennlp rank https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz > mac.ranks
allennlp rank https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz > cp.ranks
allennlp rank https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gza > elmo.ranks
allennlp rank https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz > srl.ranks
allennlp rank https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz > coref.ranks
allennlp rank https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.04.26.tar.gz > ner.ranks

