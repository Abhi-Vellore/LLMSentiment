import chardet

def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    print(result)

find_encoding('data/Sentences_75Agree.txt')
