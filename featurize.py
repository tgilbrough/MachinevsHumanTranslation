import nltk
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.tree import Tree
from nltk.corpus import brown
from InterpolationModel import InterpolationModel

k = 0.001 # To deal with division by 0
n = 2
lambdas = [0.1, 0.9]

itm = InterpolationModel(n)
itm.train(brown.sents())

def add_features(line):

    # Tokenizer
    o_sent = nltk.word_tokenize(line["chinese"])
    h_tran = nltk.word_tokenize(line["human"])
    q_tran = nltk.word_tokenize(line["?"])

    h_log_prob = itm.evaluate([h_tran], lambdas, k)[0]
    q_log_prob = itm.evaluate([q_tran], lambdas, k)[0]

    # POS Tagger
    h_pos = [word[1] for word in nltk.pos_tag(h_tran)]
    q_pos = [word[1] for word in nltk.pos_tag(q_tran)]

    h_tree = Tree.fromstring(line["h_tree"])
    q_tree = Tree.fromstring(line["q_tree"])

    # NP subtrees
    h_np = [st for st in h_tree.subtrees() if st.label() == "NP"]
    q_np = [st for st in q_tree.subtrees() if st.label() == "NP"]

    # VP subtrees
    h_vp = [st for st in h_tree.subtrees() if st.label() == "VP"]
    q_vp = [st for st in q_tree.subtrees() if st.label() == "VP"]

    # PP subtrees
    h_pp = [st for st in h_tree.subtrees() if st.label() == "PP"]
    q_pp = [st for st in q_tree.subtrees() if st.label() == "PP"]
    
    # Features
    line["human_to_trans_len_ratio"] = human_to_trans_ratio(line["human"], line["?"])
    line["original_to_trans_len_ratio"] = original_to_trans_ratio(line["chinese"], line["?"])
    line["token_ratio"] = token_ratio(h_tran, q_tran)
    line["tree_height_ratio"] = tree_height_ratio(h_tree, q_tree)
    # line["common_words_ratio"] = common_words_ratio(h_tran, q_tran)
    line["gleu"] = gleu(h_tran, q_tran)
    line["chrf"] = chrf(h_tran, q_tran)
    # line["np_height_ratio"] = np_height_ratio(h_np, q_np)
    # line["vp_height_ratio"] = vp_height_ratio(h_vp, q_vp)
    # line["np_density_ratio"] = np_density_ratio(h_np, q_np, h_tran, q_tran)
    # line["vp_density_ratio"] = vp_density_ratio(h_vp, q_vp, h_tran, q_tran)
    # line["pp_height_ratio"] = pp_height_ratio(h_pp, q_pp)
    # line["pp_density_ratio"] = pp_density_ratio(h_pp, q_pp, h_tran, q_tran)
    # line["perplexity_ratio"] = perplexity_ratio(h_tran, q_tran, h_log_prob, q_log_prob)

    return line

# Feature functions
def human_to_trans_ratio(h_tran, q_tran):
    return len(h_tran) / len(q_tran)

def original_to_trans_ratio(o_sent, q_tran):
    return len(o_sent) / len(q_tran)

def token_ratio(h_tran, q_tran):
    return (sum([len(token) for token in h_tran]) / len([len(token) for token in h_tran])) / (sum([len(token) for token in q_tran]) / len([len(token) for token in q_tran]))

def tree_height_ratio(h_tree, q_tree):
    return h_tree.height() / q_tree.height()

def common_words_ratio(h_tran, q_tran):
    return sum([1 for word in q_tran if word in h_tran]) / len(q_tran)

def gleu(h_tran, q_tran):
    return sentence_gleu(h_tran, q_tran)

def chrf(h_tran, q_tran):
    return sentence_chrf(h_tran, q_tran)

def np_height_ratio(h_np, q_np):
    h_avg = (sum([np.height() for np in h_np]) + k) / (len(h_np) + k)
    q_avg = (sum([np.height() for np in q_np]) + k) / (len(q_np) + k)
    return (h_avg + k) / (q_avg + k)

def vp_height_ratio(h_vp, q_vp):
    h_avg = (sum([vp.height() for vp in h_vp]) + k) / (len(h_vp) + k)
    q_avg = (sum([vp.height() for vp in q_vp]) + k) / (len(q_vp) + k)
    return (h_avg + k) / (q_avg + k)

def np_density_ratio(h_np, q_np, h_tran, q_tran):
    h_avg = (len(h_np) + k) / (len(h_tran) + k)
    q_avg = (len(q_np) + k) / (len(q_tran) + k)
    return (h_avg + k) / (q_avg + k)

def vp_density_ratio(h_vp, q_vp, h_tran, q_tran):
    h_avg = (len(h_vp) + k) / (len(h_tran) + k)
    q_avg = (len(q_vp) + k) / (len(q_tran) + k)
    return (h_avg + k) / (q_avg + k)

def pp_height_ratio(h_pp, q_pp):
    h_avg = (sum([pp.height() for pp in h_pp]) + k) / (len(h_pp) + k)
    q_avg = (sum([pp.height() for pp in q_pp]) + k) / (len(q_pp) + k)
    return (h_avg + k) / (q_avg + k)

def pp_density_ratio(h_pp, q_pp, h_tran, q_tran):
    h_avg = (len(h_pp) + k) / (len(h_tran) + k)
    q_avg = (len(q_pp) + k) / (len(q_tran) + k)
    return (h_avg + k) / (q_avg + k)

def perplexity_ratio(h_tran, q_tran, h_log_prob, q_log_prob):
    return (2 ** -(h_log_prob / len(h_tran))) / (2 ** -(q_log_prob / len(q_tran)))