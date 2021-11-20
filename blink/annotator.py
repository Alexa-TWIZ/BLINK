from flair.data import Sentence
from flair.models import SequenceTagger


class Annotator:

    def __init__(self):
        self.tagger = SequenceTagger.load("flair/pos-english")

    def extract_name(self, text: str):
        sentence = Sentence(text)
        # predict NER tags
        self.tagger.predict(sentence)
        sentences_gen = self._identify_sequences(sentence)
        return [y for y in sentences_gen]

    def _get_name(self, composed_name: list):
        data = {}
        words = [x.text for x in composed_name]
        data["entity"] = " ".join(words)
        data["start_pos"] = composed_name[0].start_pos
        data["end_pos"] = composed_name[-1].end_pos
        return data

    def _identify_sequences(self, sentence):
        out = True
        in_candidate = False
        in_name = False

        composed_name = []

        for token in sentence.tokens:
            if out:
                if token.labels[0].value in ["NN", "NNS", "NNP"]:
                    composed_name.append(token)
                    out = False
                    in_name = True
                elif token.labels[0].value in ["JJ", "JJS"]:
                    composed_name.append(token)
                    out = False
                    in_candidate = True

            elif in_candidate:
                if token.labels[0].value in ["NN", "NNS", "NNP"]:
                    composed_name.append(token)
                    in_candidate = False
                    in_name = True
                elif token.labels[0].value in ["JJ", "JJS"]:
                    composed_name.append(token)
                else:
                    yield self._get_name(composed_name)
                    out = True
                    in_candidate = False
                    in_name = False
                    composed_name.clear()

            elif in_name:
                if token.labels[0].value in ["NN", "NNS", "NNP"]:
                    composed_name.append(token)
                else:
                    yield self._get_name(composed_name)
                    out = True
                    in_candidate = False
                    in_name = False
                    composed_name.clear()

        if in_name:
            yield self._get_name(composed_name)


if __name__ == "__main__":
    annotator = Annotator()
    x = annotator.extract_name(
        "If you must replace the window motor, proceed with highly extreme caution. The window motor has a lot of "
        "torque and can remove a finger if it is activated and your fingers are in the gears. To safely remove a "
        "window motor, the linkage arms should be secured in a vice before the springs and motor are removed.")
    print(x)
