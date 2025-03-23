
import numpy as np
import pybliometrics
pybliometrics.scopus.init()


from pybliometrics.scopus import AbstractRetrieval
from pybliometrics.scopus import AuthorRetrieval
from pybliometrics.scopus import ScopusSearch
from pybliometrics.scopus import AuthorSearch

import pandas as pd
import matplotlib.pyplot as plt

def split_authors(author_list):
    split_arr = []
    for author in author_list:
        split = author.split(',')
        split = [x.strip() for x in split]
        split = [x.split(' ') for x in split]
        split_arr.append(np.concatenate(split))
    return split_arr

def get_initials(authors_splitted):
    return [[x[0] for x in author_sp] for author_sp in authors_splitted]

def get_similar_authors(author_set):
    author_splitted  = split_authors(author_set)
    author_group_arr = [] 
    author_check = np.zeros(len(author_set))   # 0: Author not checked, 1: Author is already conidered

    for i in range(len(author_splitted)):
    
        if author_check[i]==0:
            author_group = []
            author_1 = author_splitted[i]
            author_group.append(list(author_1))

            len_1 = [len(x) for x in author_1]
            args_1 = np.argsort(len_1)[::-1]
            
            author_1 = np.array(author_1)[args_1]
            
            for j in range(i+1, len(author_splitted)):
                if author_check[j] ==0:
                    author_2 = list(author_splitted[j].copy())

                    match1 = np.zeros(len(author_1))
                    match2 = np.zeros(len(author_2))
                    match_init1 = np.zeros(len(author_1))
                    match_init2 = np.zeros(len(author_2))
                    
                    for k in range(len(author_1)):
                        for l in range(len(author_2)):
                            if (author_1[k] == author_2[l]) and ('.' not in author_1[k]) and ('.' not in author_2[l]):
                                match1[k]  = 1
                                match2[l]  = 1
                                break

                    if np.sum(match1)>0:
                        for k in range(len(author_1)):
                            for l in range(len(author_2)):
                                if ((author_1[k][0] == author_2[l][0]) and (match1[k] ==0) and (match2[l]==0) and 
                                    (('.' in author_1[k]) or (('.' in author_2[l])) or (len(author_1[k])<=2) or (len(author_2[l])<=2)) ):
                                    match_init1[k]  = 1
                                    match_init2[l]  = 1
                                    break
                            if np.sum(match_init1)>0:
                                break
                    
                    if (np.sum(match1) == 1 and np.sum(match_init1)>=1) or (np.sum(match1) > 1):
                        author_group.append(author_2)
                        author_check[i] = 1
                        author_check[j] = 1

            author_group_arr.append(author_group)
    return author_group_arr

#============================
#  Improved data management
#============================

def get_scopus_data(query, is_name=False):
    '''
    Eg.
    is_name=True  => query='AUTHLAST(Sujith) and AUTHFIRST(R. I.)'
    is_name=False => query="7004078700"
    '''
    if is_name:
        s = AuthorSearch(query)
        print("Found", len(s.authors), "author/authors.")
        author_scopus_id = s.authors[0].eid
    else:
        author_scopus_id = query
    author_search = AuthorRetrieval(author_scopus_id)
    author_pubs = author_search.get_documents()

    return author_pubs

def create_author_scopus_data_dictionary(author_pubs):
    author_sid_all   = []
    author_name_all = []

    for i in range(len(author_pubs)):
        author_sid_all.extend(  author_pubs[i].author_ids.split(";"))
        author_name_all.extend(author_pubs[i].author_names.split(";"))

    author_sid_to_name_dict_all = {}
    for author_id, author_name in zip(author_sid_all, author_name_all):
        if author_id not in author_sid_to_name_dict_all:
            author_sid_to_name_dict_all[author_id] = {}
            author_sid_to_name_dict_all[author_id][author_name] = 1
        else:
            if author_name not in author_sid_to_name_dict_all[author_id]:
                author_sid_to_name_dict_all[author_id][author_name] = 1
            else:
                author_sid_to_name_dict_all[author_id][author_name] = author_sid_to_name_dict_all[author_id][author_name] + 1

    author_name_to_sid_dict_all = {}
    for author_id, author_name in zip(author_sid_all, author_name_all):
        author_name_to_sid_dict_all[author_name] = author_id

    return author_sid_all, author_name_all, author_sid_to_name_dict_all, author_name_to_sid_dict_all

def create_scopus_id_and_name_dictionary(author_sid_to_name_dict_all):
    author_sid_to_name_dict = {}
    author_name_to_sid_dict = {}
    for author_id, author_names in author_sid_to_name_dict_all.items():
        if len(author_names)>0:
            author_sid_to_name_dict[author_id] = author_names
            for name in author_names:
                author_name_to_sid_dict[name] = author_id
        else:
            print("Empty: ", author_id, author_names)
    return author_sid_to_name_dict, author_name_to_sid_dict

def get_author_sid_number_of_publications_dictionary_with_sorted(author_sid_to_name_dict):
    author_sid_to_npub_dict = {}
    for sid, names in author_sid_to_name_dict.items():
        author_sid_to_npub_dict[sid] = np.sum(list(names.values()))

    npub = list(author_sid_to_npub_dict.values())
    idx_sort_npub = np.argsort(npub)[::-1]

    author_sid_to_npub_sorted = {k: v for k, v in sorted(author_sid_to_npub_dict.items(), key=lambda item: -item[1])}

    return author_sid_to_npub_dict, author_sid_to_npub_sorted

def get_scopus_id_to_local_id_dictionary(author_sid_to_name_dict):
    author_sid_to_lid_dict = dict() # author scopus id to local id dictionary
    author_lid_to_sid_dict = dict() # author local id dictionary tp scopus id 
    count = 0 

    for author_sid, author_names in author_sid_to_name_dict.items():
        if len(author_names)>0:
            author_sid_to_lid_dict[author_sid] = count
            author_lid_to_sid_dict[count]      = author_sid
            count = count +  1

    return author_sid_to_lid_dict, author_lid_to_sid_dict

def get_coauthor_network_with_local_index_of_row_col(author_pubs, author_sid_to_name_dict_all, author_sid_to_lid_dict):
    coauthor_network = np.zeros((len(author_sid_to_lid_dict), len(author_sid_to_lid_dict)))
    for author_p in author_pubs:
        author_ids = author_p.author_ids.split(";")
        for i in range(len(author_ids)):
            if len(author_sid_to_name_dict_all[author_ids[i]])>0: # Check if the length of list of author names with this id is non-zero
                for j in range(i+1, len(author_ids)):
                    if len(author_sid_to_name_dict_all[author_ids[j]])>0: # Check if the length of list of author names with this id is non-zero
                        author_lid_i = author_sid_to_lid_dict[author_ids[i]]
                        author_lid_j = author_sid_to_lid_dict[author_ids[j]]

                        coauthor_network[author_lid_i, author_lid_j] += 1
                        coauthor_network[author_lid_j, author_lid_i] += 1
    return coauthor_network

def get_sequence(Nmax,power=1):
    ''' 
    Outputs a sequece [1^p, 2^p, 3^p,...] whose sum is Nmax
    '''
    n_arr = []
    cs    = 0 
    i = 0
    while cs < Nmax:
        n_int = int((i+1)**power)
        n_arr.append(n_int)
        cs = cs + n_int
        i = i + 1

    if cs > Nmax:
        n_arr[-1] = Nmax - (cs - n_int)
    return np.array(n_arr)

def get_author_distribution_over_circles(Na, power=1):
    # Circular layout
    Na_per_cir = get_sequence(Na,power=power)
    Na_cir_cum = np.cumsum(Na_per_cir)
    Na_ids = np.concatenate([[0], Na_cir_cum[:-1]]) # Starting index of each circle
    Na_ide = Na_cir_cum # Ending index of each circle
    return Na_per_cir, Na_ids, Na_ide

def get_circle_rradius(N_circles, R_max, R1_offset = 0):
    radius = (np.linspace(0,R_max, N_circles))**(1.5) # Radius of each circles
    radius[1:] = radius[1:] + R1_offset
    return radius

def get_author_scopus_index_to_xy(radius_arr, Na_ids, Na_ide, author_sid_to_npub_sorted):
    author_sid_to_xy = {}
    x_arr, y_arr = [], []
    theta_shift = np.linspace(np.pi*0.113, 2.2*np.pi, len(radius_arr))
    for i in range(len(Na_ids)):
        Na_cir = Na_ide[i] - Na_ids[i]
        
        theta = np.linspace(0, 2*np.pi, int(Na_cir)+1)[0:-1] + theta_shift[i]
        
        x = radius_arr[i] * np.cos(theta) * (1+np.random.rand(Na_cir)*0.0)
        y = radius_arr[i] * np.sin(theta) * (1+np.random.rand(Na_cir)*0.0)
        
        x_arr = np.concatenate([x_arr, x])
        y_arr = np.concatenate([y_arr, y])

    for i, [sid,_] in enumerate(author_sid_to_npub_sorted.items()):
        author_sid_to_xy[sid] = [x_arr[i], y_arr[i]]     

    return x_arr, y_arr, author_sid_to_xy

def update_authors_to_emphasize(author_sid_to_xy, R_emph, authors_to_emphasize, author_name_to_sid_dict):
    theta_emph = np.linspace(0, 2*np.pi, len(authors_to_emphasize)+1)
    for i, author_name in enumerate(authors_to_emphasize):
            sid = author_name_to_sid_dict[author_name]
            author_sid_to_xy[sid] = [R_emph*np.cos(theta_emph[i]), R_emph*np.sin(theta_emph[i])]
    return author_sid_to_xy

def update_authors_special(author_sid_to_xy, author_name_to_sid_dict, authors_special, x_shift, y_shift):
    for i, author_name in enumerate(authors_special):
            sid = author_name_to_sid_dict[author_name]
            xt, yt = author_sid_to_xy[sid]
            author_sid_to_xy[sid] = [xt+x_shift, yt+y_shift]
    return author_sid_to_xy

def get_name_for_author(auth_data):
    #auth_data = author_sid_to_name_dict['56266798600']
    name_max, papers_max, paper_total = None, 0, 0
    for name, papers in auth_data.items():
        if papers>papers_max:
            name_max = name
            papers_max = papers
        paper_total += papers
    return name_max, papers_max, paper_total

def abbriviate_author_name(auth_name):
    parts = auth_name.split(',')
    name = parts[0:1] + [txt[0]+'.' for txt in parts[1].strip().split(' ')]
    name = ' '.join(name) 
    return name

class pyconet:
    def __init__(self, query, is_query_name=False):
        '''
            Eg.
            is_name=True  => query='AUTHLAST(Sujith) and AUTHFIRST(R. I.)'
            is_name=False => query="7004078700"
        '''
        self.query = query
        self.is_query_name = False
        self.author_pubs = None
        self.author_sid_all = [] 
        self.author_name_all = []
        self.author_sid_to_name_dict_all = {}
        self.author_name_to_sid_dict_all = {}
        self.author_sid_to_name_dict = {}
        self.author_name_to_sid_dict = {}
        self.author_sid_to_npub_dict = {}
        self.author_sid_to_npub_sorted = None
        self.author_sid_to_lid_dict = {}       # author scopus id to local id dictionary
        self.author_lid_to_sid_dict = {}       # author local id dictionary tp scopus id 
        self.coauthor_network = None
        self.number_of_collaborations = None

        self.get_scopus_data()
        self.create_author_scopus_data_dictionary()
        self.create_scopus_id_and_name_dictionary()
        self.get_author_sid_number_of_publications_dictionary_with_sorted()
        self.get_scopus_id_to_local_id_dictionary()
        self.get_coauthor_network_with_local_index_of_row_col()

        self.number_of_collaborations = np.sum(self.coauthor_network, axis=1)

    def get_scopus_data(self):
        '''
        Eg.
        is_name=True  => query='AUTHLAST(Sujith) and AUTHFIRST(R. I.)'
        is_name=False => query="7004078700"
        '''
        if self.is_query_name:
            s = AuthorSearch(self.query)
            print("Found", len(s.authors), "author/authors.")
            author_scopus_id = s.authors[0].eid
        else:
            author_scopus_id = self.query
        author_search = AuthorRetrieval(author_scopus_id)
        self.author_pubs = author_search.get_documents()

    def create_author_scopus_data_dictionary(self):

        for i in range(len(self.author_pubs)):
            self.author_sid_all.extend(  self.author_pubs[i].author_ids.split(";"))
            self.author_name_all.extend( self.author_pubs[i].author_names.split(";"))

        for author_id, author_name in zip(self.author_sid_all, self.author_name_all):
            if author_id not in self.author_sid_to_name_dict_all:
                self.author_sid_to_name_dict_all[author_id] = {}
                self.author_sid_to_name_dict_all[author_id][author_name] = 1
            else:
                if author_name not in self.author_sid_to_name_dict_all[author_id]:
                    self.author_sid_to_name_dict_all[author_id][author_name] = 1
                else:
                    self.author_sid_to_name_dict_all[author_id][author_name] = self.author_sid_to_name_dict_all[author_id][author_name] + 1

        for author_id, author_name in zip(self.author_sid_all, self.author_name_all):
            self.author_name_to_sid_dict_all[author_name] = author_id

    def create_scopus_id_and_name_dictionary(self):
        for author_id, author_names in self.author_sid_to_name_dict_all.items():
            if len(author_names)>0:
                self.author_sid_to_name_dict[author_id] = author_names
                for name in author_names:
                    self.author_name_to_sid_dict[name] = author_id
            else:
                print("Empty: ", author_id, author_names)

    def get_author_sid_number_of_publications_dictionary_with_sorted(self):
        for sid, names in self.author_sid_to_name_dict.items():
            self.author_sid_to_npub_dict[sid] = np.sum(list(names.values()))

        npub = list(self.author_sid_to_npub_dict.values())
        idx_sort_npub = np.argsort(npub)[::-1]

        self.author_sid_to_npub_sorted = {k: v for k, v in sorted(self.author_sid_to_npub_dict.items(), key=lambda item: -item[1])}

    def get_scopus_id_to_local_id_dictionary(self):
        count = 0 
        for author_sid, author_names in self.author_sid_to_name_dict.items():
            if len(author_names)>0:
                self.author_sid_to_lid_dict[author_sid] = count
                self.author_lid_to_sid_dict[count]      = author_sid
                count = count +  1

    def get_coauthor_network_with_local_index_of_row_col(self):
        self.coauthor_network = np.zeros((len(self.author_sid_to_lid_dict), len(self.author_sid_to_lid_dict)))
        for author_p in self.author_pubs:
            author_ids = author_p.author_ids.split(";")
            for i in range(len(author_ids)):
                if len(self.author_sid_to_name_dict_all[author_ids[i]])>0: # Check if the length of list of author names with this id is non-zero
                    for j in range(i+1, len(author_ids)):
                        if len(self.author_sid_to_name_dict_all[author_ids[j]])>0: # Check if the length of list of author names with this id is non-zero
                            author_lid_i = self.author_sid_to_lid_dict[author_ids[i]]
                            author_lid_j = self.author_sid_to_lid_dict[author_ids[j]]

                            self.coauthor_network[author_lid_i, author_lid_j] += 1
                            self.coauthor_network[author_lid_j, author_lid_i] += 1

def plot_network(pcn_obj, fig, axs, authors_to_emphasize, authors_special, authors_to_exclude):
    Na_per_cir, Na_ids, Na_ide = get_author_distribution_over_circles(len(pcn_obj.author_sid_to_lid_dict), power=4)

    radius_arr = get_circle_rradius(N_circles= len(Na_per_cir), R_max=10, R1_offset = 20) 

    x_arr, y_arr, author_sid_to_xy = get_author_scopus_index_to_xy(radius_arr, Na_ids, Na_ide, pcn_obj.author_sid_to_npub_sorted)

    author_sid_to_xy = update_authors_to_emphasize(author_sid_to_xy, R_emph=20, authors_to_emphasize= authors_to_emphasize, author_name_to_sid_dict= pcn_obj.author_name_to_sid_dict)

    author_sid_to_xy = update_authors_special(author_sid_to_xy, pcn_obj.author_name_to_sid_dict, authors_special, x_shift=0, y_shift=10)

    axs.axis('off')
    axs.plot([0,0], [1,1], 'w')
    axs.set_aspect(aspect=.7)

    #color_var_arr = number_of_collaborations[args]
    color_var_arr = (x_arr**2 + y_arr**2)
    vmin = min(color_var_arr) - 0.2*(max(color_var_arr) - min(color_var_arr))
    vmax = max(color_var_arr) + 0.1*(max(color_var_arr) - min(color_var_arr))
    cmap = 'Paired'#'tab10'#'jet'

    #for xa, ya, author, n_colabs, color_var, arg in zip(x_arr, y_arr, author_set[args], number_of_collaborations[args], color_var_arr, args):
    for lid, sid in pcn_obj.author_lid_to_sid_dict.items():
            xa, ya = author_sid_to_xy[sid]
            auth_data = pcn_obj.author_sid_to_name_dict[sid]
            author, _, papers_total =  get_name_for_author(auth_data=auth_data)
            author_txt = abbriviate_author_name(auth_name=author)
            color_var = (xa**2 + ya**2 )


            text_rot = np.arctan2(ya,xa)*180/np.pi
            if text_rot > 90:
                text_rot = text_rot - 180
            if text_rot < -90:
                text_rot = text_rot + 180 

            if author not in authors_to_exclude:
                if author in authors_special:
                    axs.scatter(xa, ya, s= papers_total, 
                                        c=color_var, 
                                        cmap=cmap, vmin=vmin, vmax=vmax)   
                    
                    axs.text(   xa, ya-6, s= author_txt, 
                                        fontsize= 16,
                                        horizontalalignment='center', 
                                        verticalalignment='bottom',
                                        alpha= 1,
                                        rotation=text_rot*0, 
                                        color = "darkslateblue",
                                        weight='bold')

                    for coauth_lid, coauthor_conn in enumerate(pcn_obj.coauthor_network[lid]):
                        names_conn = pcn_obj.author_sid_to_name_dict[pcn_obj.author_lid_to_sid_dict[coauth_lid]]
                        flag_conn = 1
                        for name_c in names_conn:
                            if name_c in authors_to_exclude:
                                flag_conn = 0
                                break

                        if coauthor_conn > 0 and flag_conn:
                            xb, yb = author_sid_to_xy[pcn_obj.author_lid_to_sid_dict[coauth_lid]]
                            axs.plot([xa, xb], [ya, yb], 'k', alpha=0.2, zorder=-1, linewidth=0.1)

                elif (author in authors_to_emphasize):

                    axs.scatter(xa, ya, s= 10+papers_total*1, 
                                        c=color_var, 
                                        cmap=cmap, vmin=vmin, vmax=vmax)   
                    
                    axs.text(   xa, ya, s= author_txt, 
                                        fontsize= 12,
                                        horizontalalignment='center', 
                                        verticalalignment='bottom',
                                        alpha= 1,
                                        rotation=text_rot,
                                        color = 'brown')
                    for coauth_lid, coauthor_conn in enumerate(pcn_obj.coauthor_network[lid]):
                        names_conn = pcn_obj.author_sid_to_name_dict[pcn_obj.author_lid_to_sid_dict[coauth_lid]]
                        flag_conn = 1
                        for name_c in names_conn:
                            if name_c in authors_to_exclude:
                                flag_conn = 0
                                break

                        if coauthor_conn > 0 and flag_conn:
                            xb, yb = author_sid_to_xy[pcn_obj.author_lid_to_sid_dict[coauth_lid]]
                            axs.plot([xa, xb], [ya, yb], 'k', alpha=0.2, zorder=-1, linewidth=0.1)
                else:
                    axs.scatter(xa, ya, s=papers_total*4, 
                                        c=color_var, cmap=cmap, vmin=vmin, vmax=vmax)
                    axs.text(xa, ya, s=author_txt, 
                                        fontsize= 9,#4+ np.log(n_colabs)*1.,
                                        horizontalalignment='center', 
                                        verticalalignment='bottom',
                                        alpha=0.9, rotation=text_rot)
                    
                    for coauth_lid, coauthor_conn in enumerate(pcn_obj.coauthor_network[lid]):
                        names_conn = pcn_obj.author_sid_to_name_dict[pcn_obj.author_lid_to_sid_dict[coauth_lid]]
                        flag_conn = 1
                        for name_c in names_conn:
                            if name_c in authors_to_exclude:
                                flag_conn = 0
                                break

                        if coauthor_conn > 0 and flag_conn:
                            xb, yb = author_sid_to_xy[pcn_obj.author_lid_to_sid_dict[coauth_lid]]
                            axs.plot([xa, xb], [ya, yb], 'k', alpha=0.2, zorder=-1, linewidth=0.1)
    fig.tight_layout()


