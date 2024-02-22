# Jordan Boyd-Graber
# 2023
#
# Feature extractors to improve classification to determine if an answer is
# correct.

from collections import Counter
from math import log
from numpy import mean
import gzip
import json
import spacy

class Feature:
    """
    Base feature class.  Needs to be instantiated in params.py and then called
    by buzzer.py
    """

    def __init__(self, name):
        self.name = name

    def __call__(self, question, run, guess, guess_history):
        raise NotImplementedError(
            "Subclasses of Feature must implement this function")

    
"""
Given features (Length, Frequency)
"""
class LengthFeature(Feature):
    """
    Feature that computes how long the inputs and outputs of the QA system are.
    """

    def __call__(self, question, run, guess, guess_history):
        # How many characters long is the question?
        yield ("char", (len(run) - 450) / 450)

        # How many words long is the question?
        yield ("word", (len(run.split()) - 75) / 75)
   
        # How many characters long is the guess?
        if guess is None or guess=="":
            yield ("guess", -1)
        else:
            yield ("guess", log(1 + len(guess)))
    

class GuessBlankFeature(Feature):
    """
    Is guess blank?
    """        
    def __call__(self, question, run, guess, guess_history):
        yield ('true', len(guess) == 0)

class CorrelationFeature(Feature):
    '''
    calculates the correlation score between the guess and the quesiton using spacy
    '''
    def __call__(self, question, run, guess, guess_history):
        nlp = spacy.load("en_core_web_sm")
        guess = nlp(guess)
        run = nlp(run)

        similarity = guess.similarity(run)
        yield("similarity", similarity)
        print(similarity)

class EntityFeature(Feature):
    def __call__(self, question, run, guess, guess_history):
        nlp = spacy.load("en_core_web_sm")
        doc_guess = nlp(guess)
        doc_question = nlp(run)

        entities_question = [ents.label_ for ents in doc_question.ents]
        entities_guess = [ents.label_ for ents in doc_guess.ents]

        res = False

        for ent in entities_guess:
            if ent in entities_question:
                res = True
                break
        
        yield("true", res)

class VerbFeature(Feature):
    def __call__(self, question, run, guess, guess_history):
        nlp = spacy.load("en_core_web_sm")
        doc_guess = nlp(guess)
        POS_guess = [token.pos_ for token in doc_guess]
        print(POS_guess)
        
        yield("true", "VERB" in POS_guess)


if __name__ == "__main__":
    """

    Script to write out features for inspection or for data for the 470
    logistic regression homework.

    """
    import argparse
    
    from params import add_general_params, add_question_params, \
        add_buzzer_params, add_guesser_params, setup_logging, \
        load_guesser, load_questions, load_buzzer

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_guess_output', type=str)
    add_general_params(parser)    
    add_guesser_params(parser)
    add_buzzer_params(parser)    
    add_question_params(parser)

    flags = parser.parse_args()

    setup_logging(flags)

    guesser = load_guesser(flags)
    buzzer = load_buzzer(flags)
    questions = load_questions(flags)

    buzzer.add_data(questions)
    buzzer.build_features(flags.buzzer_history_length,
                          flags.buzzer_history_depth)

    vocab = buzzer.write_json(flags.json_guess_output)
    with open("data/small_guess.vocab", 'w') as outfile:
        for ii in vocab:
            outfile.write("%s\n" % ii)
