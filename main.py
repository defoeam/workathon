import networkx as nx
import ast
import json
import matplotlib.pyplot as plt

import llm_agents.llm_agent as agent

open_information_extraction_sysprompt = """
Given a piece of text, extract relational triplets in the form of [Subject, Relation, Object] from it.
Relational triplets should have three items maximum. 
Here are some examples: Example 1: Text: The 17068.8 millimeter long ALCO RS-3 has a diesel-electric transmission. 
Example Output: [['ALCO RS-3', 'powerType', 'Diesel-electric transmission'], ['ALCO RS-3', 'length', '17068.8 (millimetres)']]"
"""
schema_definition_sysprompt = """
Given a piece of text and a list of relational triplets extracted from it, write a definition for each relation present. 
Example 1: Text: The 17068.8 millimeter long ALCO RS-3 has a diesel-electric transmission. 
Triplets: [[’ALCO RS-3’, ’powerType’, ’Diesel-electric transmission’], 
[’ALCO RS-3’, ’length’, ’17068.8 (millimetres)’]] 
Definitions:
powerType: The subject entity uses the type of power or energy source specified by the object entity
length: The measurement or extent of something from end to end; the greater of two or the greatest of three dimensions of a body.
"""
#todo
schema_canonicalization_sysprompt = """

"""

input_text = """
The Gutenberg Bible, also known as the 42-line Bible, the Mazarin Bible or the B42, was the earliest major book printed in Europe using mass-produced metal movable type. It marked the start of the "Gutenberg Revolution" and the age of printed books in the West. The book is valued and revered for its high aesthetic and artistic qualities[1] and its historical significance.

The Gutenberg Bible is an edition of the Latin Vulgate printed in the 1450s by Johannes Gutenberg in Mainz, in present-day Germany. Forty-nine copies (or substantial portions of copies) have survived. They are thought to be among the world's most valuable books, although no complete copy has been sold since 1978.[2][3] In March 1455, the future Pope Pius II wrote that he had seen pages from the Gutenberg Bible displayed in Frankfurt to promote the edition, and that either 158 or 180 copies had been printed.

The 36-line Bible, said to be the second printed Bible, is also sometimes referred to as a Gutenberg Bible, but may be the work of another printer.[4]
"""

relation_text = agent.run_agent(open_information_extraction_sysprompt, input_text)
schema_text = agent.run_agent(schema_definition_sysprompt, "Text: {}, Triplets: {}".format(input_text, relation_text))

# Convert the relation_text string to a list of tuples manually
relation_tuples = []
relation_text = relation_text.strip('[]')
for item in relation_text.split('], ['):
    item = item.strip('[]').replace("'", "").split(', ')
    relation_tuples.append((item[0], item[1], item[2]))

# Create a directed graph
graph = nx.DiGraph()

# Add edges to the graph
for subj, rel, obj in relation_tuples:
    graph.add_edge(subj, obj, relation=rel)

def visualize_graph(G):
    pos = nx.spring_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.show()

visualize_graph(graph)