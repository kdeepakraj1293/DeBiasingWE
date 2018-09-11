'''
 DeBiasingWE:
 -------------
 Debiasing Word Embeddings and removing gender biases from pretrained GloVe Vectors.
'''

import numpy as np
from w2v_utils import *

# load the glove vectors
words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')

# 1 - Cosine similarity
# To measure how similar two words are,
# we need a way to measure the degree of similarity between two embedding vectors for the two words.

def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """

    distance = 0.0

    # Compute the dot product between u and v
    dot = np.dot(u,v)
    # Compute the L2 norm of u
    norm_u = np.linalg.norm(u)

    # Compute the L2 norm of v
    norm_v = np.linalg.norm(v)
    # Compute the cosine similarity
    cosine_similarity = dot / (norm_u*norm_v)

    return cosine_similarity

# 2 - Word analogy

# In the word analogy, we complete the sentence "a is to b as c is to ____"
# An example is 'man is to woman as king is to queen'.
# In detail, we are trying to find a word d,
# such that the associated word vectors e_a, e_b, e_c, e_d are related
# in the following manner:
# e_b - e_a ~> e_d - e_c.
# We will measure the similarity between
# e_b - e_a and e_d - e_c using cosine similarity.

def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """

    # convert words to lower case
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # Get the word embeddings v_a, v_b and v_c
    e_a, e_b, e_c = word_to_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]

    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c] :
            continue

        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)
        cosine_sim = cosine_similarity((e_b - e_a),word_to_vec_map[w]-e_c)

        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word

# 3 - Debiasing word vectors
# We can examine gender biases that can be reflected in a word embedding,
# and explore algorithms for reducing the bias.
# Lets first see how the word embeddings relate to gender.
# First let us compute a vector g = e_{woman}-e_{man},
# The resulting vector g roughly encodes the concept of "gender".
# (we can get a more accurate representation if we compute
# g_1 = e_{mother}-e_{father},
# g_2 = e_{girl}-e_{boy}, etc.
# and average over them.
# But just using e_{woman}-e_{man} will give good enough results.)

g = word_to_vec_map['woman'] - word_to_vec_map['man']
print("Vector represinding Gender is : ", g)


# 3.1 - Neutralize bias for non-gender specific words

def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    # Select word vector representation of "word". Use word_to_vec_map.
    e = word_to_vec_map[word]

    # Compute e_biascomponent using the formula give above.
    e_biascomponent = np.dot(e,g)*g/(np.linalg.norm(g)**2)

    # Neutralize e by substracting e_biascomponent from it
    # e_debiased should be equal to its orthogonal projection.
    e_debiased = e - e_biascomponent

    return e_debiased
    
e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))

# 3.2 - Equalization algorithm for gender-specific words
def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    # Step 1: Select word vector representation of "word". Use word_to_vec_map.
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1],word_to_vec_map[w2]

    # Step 2: Compute the mean of e_w1 and e_w2
    mu = (e_w1+e_w2)/2

    # Step 3: Compute the projections of mu over the bias axis and the orthogonal axis
    mu_B = np.dot(mu,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2
    mu_orth = mu-mu_B

    # Step 4: Use equations (7) and (8) to compute e_w1B and e_w2B
    e_w1B = np.dot(e_w1,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2
    e_w2B = np.dot(e_w2,bias_axis)*bias_axis/np.linalg.norm(bias_axis)**2

    # Step 5: Adjust the Bias part of e_w1B and e_w2B using the formulas (9) and (10) given above
    corrected_e_w1B = np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))*(e_w1B-mu_B)/(abs(e_w1-mu_orth-mu_B))
    corrected_e_w2B = np.sqrt(abs(1-np.linalg.norm(mu_orth)**2))*(e_w2B-mu_B)/(abs(e_w2-mu_orth-mu_B))

    # Step 6: Debias by equalizing e1 and e2 to the sum of their corrected projections
    e1 = corrected_e_w1B+mu_orth
    e2 = corrected_e_w2B+mu_orth

    return e1, e2

print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
