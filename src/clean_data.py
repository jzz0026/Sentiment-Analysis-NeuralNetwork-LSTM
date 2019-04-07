from gensim.models import Word2Vec
import re
import json
import sys
import argparse
import glob
import os



root_path = Path(__file__).parents[2]
files = os.path.abspath(os.path.join(root_path, 'data/..'))  
cleaned_files=os.path.abspath(os.path.join(root_path, 'data/..'))  
word_embed=os.path.abspath(os.path.join(root_path, 'data/word2embed.txt'))


def read_files():
    #Parsing for the user arguments
    parser = argparse.ArgumentParser(word_embed)
    #Required input file
    parser.add_argument("input", files )
    #Optional arguments (room for further extending the script's capabilities)
    parser.add_argument("-o", "--output", default="vector.json",word_embed)
    args = parser.parse_args()
    #Using the arguments from the arg dictionary
    output_text_file = args.output
    listOfFiles = []
    if os.path.isdir(args.input):
    # Make a list with all txt in the folder
        listOfFiles = glob.glob(args.input + '/*.txt')
    else:
        # use a single file
        listOfFiles.append(args.input)
    return listOfFiles

listOfFiles = read_files()
def clean_data():
    final_sentences = []
    for file in listOfFiles:
        text = open(file).read().lower().replace("\n", " ") # Remove lineabreaks
        # Split into sentences 
        sentences = re.split("[.?!]", text)
        # Split each sentence into words!
        for sentence in sentences:
            words = re.split(r'\W+', sentence)
            final_sentences.append(words)
    return final_sentences


# Create the Word2Vec model
output_text_file = word_embed.open()
final_sentences = clean_data()
model = Word2Vec(final_sentences, size=100, window=5, min_count=5, workers=4)
# Save the vectors to a text file
model.wv.save_word2vec_format(output_text_file, binary=False)

# Open up that text file and convert to JSON
f = open(output_text_file)
v = {"vectors": {}}
for line in f:
    w, n = line.split(" ", 1)
    v["vectors"][w] = list(map(float, n.split()))

# Save to a JSON file
# Could make this an optional argument to specify output file
with open(output_text_file[:-4] + "json", "w") as out:
    json.dump(v, out)


def add_label():
    '''Add a label of either 0 (genative) or 1 (positive) to each review and write it to a new .txt-file'''
    
    cleaned_files_path=[cleaned_files]
    labels=[0,1]

    with open(cleaned_files_path, 'w') as writer:
        for file, label in zip(cleaned_files_path, labels):
            for line in open(file):
                line=line.rstrip('\n') + '| ' + str(label)+'\n'
                writer.write(line)   


   