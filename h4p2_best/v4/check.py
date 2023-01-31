# ARPABET PHONEME MAPPING
# DO NOT CHANGE
# This overwrites the phonetics.py file.

CMUdict_ARPAbet = {
    "[PAD]": "[PAD]", "[SOS]": "[SOS]", "[EOS]": "[EOS]", "": " ",
    "[SIL]": "-", "NG": "G", "F": "f", "M": "m", "AE": "@",
    "R": "r", "UW": "u", "N": "n", "IY": "i", "AW": "W",
    "V": "v", "UH": "U", "OW": "o", "AA": "a", "ER": "R",
    "HH": "h", "Z": "z", "K": "k", "CH": "C", "W": "w",
    "EY": "e", "ZH": "Z", "T": "t", "EH": "E", "Y": "y",
    "AH": "A", "B": "b", "P": "p", "TH": "T", "DH": "D",
    "AO": "c", "G": "g", "L": "l", "JH": "j", "OY": "O",
    "SH": "S", "D": "d", "AY": "Y", "S": "s", "IH": "I",
}

CMUdict = list(CMUdict_ARPAbet.keys())
ARPAbet = list(CMUdict_ARPAbet.values())


PHONEMES = CMUdict
mapping = CMUdict_ARPAbet
LABELS = ARPAbet

# These are the various characters in the transcripts of the datasetW
VOCAB = ['<pad>', '<sos>', '<eos>',
         'A',   'B',    'C',    'D',
         'E',   'F',    'G',    'H',
         'I',   'J',    'K',    'L',
         'M',   'N',    'O',    'P',
         'Q',   'R',    'S',    'T',
         'U',   'V',    'W',    'X',
         'Y',   'Z',    "'",    ' ',
         ]


VOCAB_MAP = {CMUdict[i]: i for i in range(0, len(CMUdict))}

# SOS_TOKEN = VOCAB_MAP["<sos>"]
# EOS_TOKEN = VOCAB_MAP["<eos>"]
