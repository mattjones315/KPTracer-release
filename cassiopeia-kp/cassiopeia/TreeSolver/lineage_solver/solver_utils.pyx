import networkx as nx
import numpy as np
from collections import OrderedDict
import sys

def node_parent(x, y):
	"""
	Given two nodes, finds the latest common ancestor
	:param x:
		Sample x in string format no identifier
	:param y:
		Sample x in string format no identifier
	:return:
		Returns latest common ancestor of x and y
	"""

	parr = []
	if '_' in x:
		x = ''.join(x.split("_")[:-1])
	if '_' in y:
		y = ''.join(y.split("_")[:-1])
	x_list = x.split('|')
	y_list = y.split('|')
	for i in range(0,len(x_list)):
		if x_list[i] == y_list[i]:
			parr.append(x_list[i])
		elif x_list[i] == '-':
			parr.append(y_list[i])
		elif y_list[i] == '-':
			parr.append(x_list[i])
		else:
			parr.append('0')

	return '|'.join(parr)

def get_edge_length(x,y,priors=None, weighted=False):
	"""
	Given two nodes, if x is a parent of y, returns the edge length between x and y, else -1
	:param x:
		Sample x in string format no identifier
	:param y:
		Sample x in string format no identifier
	:param priors:
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:return:
		Length of edge if valid transition, else -1
	"""
	count = 0
	if '_' in x:
		x = ''.join(x.split("_")[:-1])
	if '_' in y:
		y = ''.join(y.split("_")[:-1])
	x_list = x.split('|')
	y_list = y.split('|')

	for i in range(0, len(x_list)):
			if x_list[i] == y_list[i]:
					pass
			elif y_list[i] == "-":
					count += 0

			elif x_list[i] == '0':
				if not weighted:
					count += 1
				else:
					count += -np.log(priors[i][str(y_list[i])])
			else:
				return -1
	return count

def mutations_from_parent_to_child(parent, child):
	"""
	Creates a string label describing the mutations taken from  a parent to a child
	:param parent: A node in the form 'Ch1|Ch2|....|Chn'
	:param child: A node in the form 'Ch1|Ch2|....|Chn'
	:return: A comma seperated string in the form Ch1: 0-> S1, Ch2: 0-> S2....
	where Ch1 is the character, and S1 is the state that Ch1 mutaated into
	"""
	if '_' in parent:
		parent = ''.join(parent.split("_")[:-1])
	if '_' in child:
		child = ''.join(child.split("_")[:-1])

	parent_list = parent.split("_")[0].split('|')
	child_list = child.split("_")[0].split('|')
	mutations = []
	for i in range(0, len(parent_list)):
		if parent_list[i] != child_list[i] and child_list[i] != '-':
			mutations.append(str(i) + ": " + str(parent_list[i]) + "->" + str(child_list[i]))

	return " , ".join(mutations)

def root_finder(target_nodes):
	"""
	Given a list of targets_nodes, return the least common ancestor of all nodes
	:param target_nodes:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:return:
		The least common ancestor of all target nodes, in the form 'Ch1|Ch2|....|Chn'
	"""
	np = target_nodes[0]
	for sample in target_nodes:
		np = node_parent(sample, np)

	return np

def build_potential_graph_from_base_graph(samples, root, max_neighborhood_size = 10000, priors=None, pid=-1, weighted = False, lca_dist = None):
	"""
	Given a series of samples, or target nodes, creates a tree which contains potential
	ancestors for the given samples.
	First, a directed graph is constructed, by considering all pairs of samples, and checking
	if a sample can be a possible parent of another sample
	Then we all pairs of nodes with in-degree 0 and < a certain edit distance away
	from one another, and add their least common ancestor as a parent to these two nodes. This is done
	until only one possible ancestor remains
	:param samples:
		A list of target nodes, where each node is in the form 'Ch1|Ch2|....|Chn'
	:param priors
		A nested dictionary containing prior probabilities for [character][state] mappings
		where characters are in the form of integers, and states are in the form of strings,
		and values are the probability of mutation from the '0' state.
	:return:
		A graph, which contains a tree which explains the data with minimal parsimony
	"""
		#print "Initial Sample Size:", len(set(samples))

	cdef int neighbor_mod
	cdef int max_neighbor_dist


	neighbor_mod = 0
	prev_network = None
	flag = False

	potential_graph_diagnostic = {}
	prev_widths = []

	if lca_dist is None:
		lca_dist = 13

	print("Estimating potential graph with maximum neighborhood size of " + str(max_neighborhood_size) + " with lca distance of " + str(lca_dist) + " (pid: " + str(pid) + ")")
	sys.stdout.flush()

	max_neighbor_dist = 0
	while max_neighbor_dist < (lca_dist+1):	 
		initial_network = nx.DiGraph()
		samples = np.unique((samples))
		for sample in samples:
			initial_network.add_node(sample)

		source_nodes = samples
		neighbor_mod = max_neighbor_dist
		max_width = 0

		while len(source_nodes) != 1:

			if len(source_nodes) > int(max_neighborhood_size):
				print("Max Neighborhood Exceeded, Returning Network (pid: " + str(pid) + ")")
				return prev_network, max_neighbor_dist - 1, potential_graph_diagnostic

			temp_source_nodes = list()
			for i in range(0, len(source_nodes)-1):
				sample = source_nodes[i]
				top_parents = []
				p_to_s1_lengths, p_to_s2_lengths = {}, {}
				muts_to_s1, muts_to_s2 = {}, {}
				for j in range(i + 1, len(source_nodes)):
					sample_2 = source_nodes[j]
					if sample != sample_2:

						parent = node_parent(sample, sample_2)
						edge_length_p_s1 = get_edge_length(parent, sample)
						edge_length_p_s2 = get_edge_length(parent, sample_2)
						top_parents.append((edge_length_p_s1 + edge_length_p_s2, parent, sample_2))

						muts_to_s1[(parent, sample)] = mutations_from_parent_to_child(parent, sample)
						muts_to_s2[(parent, sample_2)] = mutations_from_parent_to_child(parent, sample_2)

						p_to_s1_lengths[(parent, sample)] = edge_length_p_s1
						p_to_s2_lengths[(parent, sample_2)] = edge_length_p_s2

						#Check this cutoff
						if edge_length_p_s1 + edge_length_p_s2 < neighbor_mod:

							edge_length_p_s1_priors, edge_length_p_s2_priors = get_edge_length(parent, sample, priors, weighted), get_edge_length(parent, sample_2, priors, weighted)

							initial_network.add_edge(parent, sample_2, weight=edge_length_p_s2_priors, label=muts_to_s2[(parent, sample_2)])
							initial_network.add_edge(parent, sample, weight=edge_length_p_s1_priors, label=muts_to_s1[(parent, sample)])
							temp_source_nodes.append(parent)

							p_to_s1_lengths[(parent, sample)] = edge_length_p_s1_priors
							p_to_s2_lengths[(parent, sample_2)] = edge_length_p_s2_priors

				min_distance = min(top_parents, key = lambda k: k[0])[0]
				lst = [(s[1], s[2]) for s in top_parents if s[0] <= min_distance]

				for parent, sample_2 in lst:
					initial_network.add_edge(parent, sample_2, weight=p_to_s2_lengths[(parent, sample_2)], label=muts_to_s2[(parent, sample_2)])
					initial_network.add_edge(parent, sample, weight=p_to_s1_lengths[(parent, sample)], label=muts_to_s1[(parent, sample)])
					temp_source_nodes.append(parent)

				temp_source_nodes = list(np.unique(temp_source_nodes))
				if len(temp_source_nodes) > int(max_neighborhood_size) and prev_network != None:
					return prev_network, max_neighbor_dist - 1, potential_graph_diagnostic

			if len(source_nodes) > len(temp_source_nodes):
				if neighbor_mod == max_neighbor_dist:
					neighbor_mod *= 3

			source_nodes = temp_source_nodes
			max_width = max(max_width, len(source_nodes))
		
		max_width = max(max_width, len(source_nodes))
		print("LCA Distance " + str(max_neighbor_dist) + " completed with a neighborhood size of " + str(max_width) + " (pid: " + str(pid) + ")")
		sys.stdout.flush()

		if len(prev_widths) > 2 and max_width == prev_widths[-1] and max_width == prev_widths[-2]:
			max_neighbor_dist += 5
		elif len(prev_widths) > 1 and max_width == prev_widths[-1]:
			max_neighbor_dist += 3
		else:
			max_neighbor_dist += 1
		
		potential_graph_diagnostic[max_neighbor_dist] = max_width
		prev_widths.append(max_width)
		
		prev_network = initial_network
		if flag:
			return prev_network, max_neighbor_dist - 1, potential_graph_diagnostic

	return initial_network, max_neighbor_dist, potential_graph_diagnostic


def get_sources_of_graph(tree):
	"""
	Returns all nodes with in-degree zero
	:param tree:
		networkx tree
	:return:
		Leaves of the corresponding Tree
	"""
	return [x for x in tree.nodes() if tree.in_degree(x)==0 ]