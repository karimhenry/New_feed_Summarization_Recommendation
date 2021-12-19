#import necessary libraries

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

# Evaluation
from rouge import Rouge
rouge = Rouge()



class Summarizer:
    def extracitve(text,model,SENTENCES_COUNT ):
        """ Summarizer function to run extracitve summary prediction with different models"""
        language='english'


        parser = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer = Stemmer(language)

        summarizer = model(stemmer)

        summarizer.stop_words = get_stop_words(language)
        summary = []
        for sentence in summarizer(parser.document, SENTENCES_COUNT):
            summary.append(str(sentence))

        return " ".join(summary)
    
    
    def abstractive (text,tokenizer,model ):
        """ Summarizer function to run abstractive summary prediction with different models"""
        inputs = tokenizer([text], padding=True, truncation=True, max_length=512,return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], num_beams=4, max_length=200,min_length=120, early_stopping=True)
        summary= " ".join([tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False) for output in outputs])

        return summary
    def summaryScore(summary,ref,Avg=False):
        """Computes ROUGE-N,ROUGE-L  of two text collections of sentences."""
        return rouge.get_scores(summary,ref, avg=Avg)

    


