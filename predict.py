# ===================================================================
#                           PREDICT LABEL FUNCTION
# ===================================================================

def getLabel(input_text):
    """
    Predict phrase for setting the reminder using Spacy's NLP techniques.

    Parameters
    ----------
    input_text : str
        Input sentence to be fed in.

    Returns
    -------
    str
        If phrase has been extracted then it will return phrase.
        Else returns 'Not Found'.

    """
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')

        predicted_label = '' # Empty string just to clean memory to be on a safer side.
        subj_classes = ['csubj', 'nsubj', 'csubjpass', 'nsubjpass', 'sb', 'sbp', 'sp'] # Spacy classes for sentence subjects
        root_class = ['ROOT'] # Spacy classes for sentence root words
        obj_classes = ['iobj', 'obj', 'dobj', 'oprd', 'obj', 'pobj', 'oa', 'oa2', 'oc', 'og', 'op'] # Spacy classes for sentence objects
        objects = [] # List to store objects detected in sentence
        subjects = [] # List to store subjects detected in sentence
        roots = [] # List to store root words detected in sentence
        noun_chunks = [] # List to store noun chunks (similar to clauses) detected in sentence

        # Identify the subjects, root words, objects in a sentence
        doc = nlp(input_text) # CLassify the sentence

        for text in doc:
            if str(text.dep_) in subj_classes:
                subjects.append(text.text)

            if str(text.dep_) in obj_classes:
                objects.append(text.text)

            if str(text.dep_) in root_class:
                roots.append(text.text)

        # Get the noun chunks from sentence
        for nc in doc.noun_chunks:
            noun_chunks.append(nc.text)
            clause_analysis = nlp(nc.text)


        # Check if the noun chunk has an object in it to affirm that it
        # is a sentence predicate
        for chunk in noun_chunks:
            if len(chunk.strip().split()) > 0:
                for word in chunk.strip().split():
                    if word in objects:
                        predicted_label = chunk

        # Return the predicted label/phrase
        if predicted_label == '':
            return "Not Found"

        else:
            return predicted_label

    except Exception as e:
        raise e

# ===================================================================
#                           MAIN FUNCTION
# ===================================================================
import pandas as pd

def main():

    # Load the evaluation dataset
    df2 = pd.read_csv('./Assignment_phrase_extractor/test_data.csv')
    labels_predicted = [] # List to store labels

    # Get labels for each sentence in evaluation dataset
    for sentence in df2['Message']:
        labels_predicted.append(getLabel(sentence))

    # Save the predicted labels as a .csv file
    labels = pd.DataFrame(labels_predicted, columns=['Labels'])
    labels.to_csv('labels.csv', index=False)


if __name__ == '__main__':
    main()
