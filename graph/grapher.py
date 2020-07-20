import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from process_txt import code_sentence

# for making adjacency matrix
import networkx as nx
import re


class ObjectTree:	
	'''
		Description:
		-----------
			This class is used to generate a dictionary of lists that contain
			the graph structure:
				{src_id: [dest_id1, dest_id2, ..]}
			and return the list of text entities in the input document
		
		Example use:
		-----------
			>> connector = ObjectTree(label_column='label')
			>> connector.read(object_map_df, img)
			>> df, obj_list = connector.connect(plot=False, export_df=False)
	'''

	def __init__(self, label_column='label'):
		self.label_column = label_column
		self.df = None
		self.img = None
		self.file_name = None
	
	def read(self, object_map, image, save_name):
		
		'''
			Function to ensure the data is in correct format and saves the 
			dataframe and image as class properties

			Args:
				object_map: pd.DataFrame, having coordinates of bounding boxes,
										  text object and label
				
				image: np.array, black and white cv2 image

			Returns:
				None

		'''

		assert type(object_map) == pd.DataFrame,f'object_map should be of type \
			{pd.DataFrame}. Received {type(object_map)}'
		assert type(image) == np.ndarray,f'image should be of type {np.ndarray} \
			. Received {type(image)}'

		assert 'xmin' in object_map.columns, '"xmin" not in object map'
		assert 'xmax' in object_map.columns, '"xmax" not in object map'
		assert 'ymin' in object_map.columns, '"ymin" not in object map'
		assert 'ymax' in object_map.columns, '"ymax" not in object map'
		assert 'Object' in object_map.columns, '"Object" column not in object map'
		assert self.label_column in object_map.columns, \
						f'"{self.label_column}" does not exist in the object map'

		# check if image is greyscale
		assert image.ndim == 2, 'Check if the read image is greyscale.'

		# drop unneeded columns
		required_cols = {'xmin', 'xmax', 'ymin', 'ymax', 'Object', 
							self.label_column}
		un_required_cols = set(object_map.columns) - required_cols
		object_map.drop(columns=un_required_cols, inplace=True)
		
		self.df = object_map
		self.img = image
		self.file_name = save_name
		return 


	def connect(self, plot=False, export_df=False):
		
		'''
			This method implements the logic to generate a graph based on
			visibility. If a horizontal/vertical line can be drawn from one
			node to another, the two nodes are connected.

			Args:
				plot (default=False):
					bool, whether to plot the graph;
					the graph is plotted in at path ./grapher_outputs/plots

				export_df (default=False):
					bool, whether to export the dataframe containing graph 
					information;
					the dataframe is exported as csv to path 
					./grapher_outputs/connections
		'''
		df, img = self.df, self.img

		# check if object map was successfully read by .read() method
		try:
			if len(df) == 0:
				return
		except:
			return
		
		# initialize empty df to store plotting coordinates
		df_plot = pd.DataFrame()
		
		# initialize empty lists to store coordinates and distances
		# ================== vertical======================================== #
		distances, nearest_dest_ids_vert = [], []
		
		x_src_coords_vert, y_src_coords_vert, x_dest_coords_vert, \
		y_dest_coords_vert = [], [], [], []

		# ======================= horizontal ================================ #
		lengths, nearest_dest_ids_hori = [], []
		
		x_src_coords_hori, y_src_coords_hori, x_dest_coords_hori, \
		y_dest_coords_hori = [], [], [], []

		# 遍历源块
		for src_idx, src_row in df.iterrows():
			
			# ================= vertical ======================= #
			src_range_x = (src_row['xmin'], src_row['xmax'])
			src_center_y = (src_row['ymin'] + src_row['ymax'])/2

			dest_attr_vert = []

			# ================= horizontal ===================== #
			src_range_y = (src_row['ymin'], src_row['ymax'])
			src_center_x = (src_row['xmin'] + src_row['xmax'])/2
			
			dest_attr_hori = []

			################ iterate over destination objects #################
			# 遍历目标块
			for dest_idx, dest_row in df.iterrows():
				# flag to signal whether the destination object is below source
				# 1 先判断当前目标文字块是不是属于下方
				# is_beneath = False
				if not src_idx == dest_idx: # 排除自己和自己作比较
					is_beneath = False
					# ==================== vertical ==========================#
					dest_range_x = (dest_row['xmin'], dest_row['xmax'])
					dest_center_y = (dest_row['ymin'] + dest_row['ymax'])/2
					
					height = dest_center_y - src_center_y

					# consider only the cases where destination object lies 
					# below source
					if dest_center_y > src_center_y:
						# check if horizontal range of dest lies within range of source
						# 如果想考虑左下方、右下方（水平无重叠）的连接，增加2个case即可，并设置阈值，不要偏移太远

						# case 1
						if dest_range_x[0] <= src_range_x[0] and \
							dest_range_x[1] >= src_range_x[1]:
							
							x_common = (src_range_x[0] + src_range_x[1])/2
							
							line_src = (x_common , src_center_y)
							line_dest = (x_common, dest_center_y)

							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)
							
							is_beneath = True

						# case 2
						elif dest_range_x[0] >= src_range_x[0] and \
							dest_range_x[1] <= src_range_x[1]:
							
							x_common = (dest_range_x[0] + dest_range_x[1])/2
							
							line_src = (x_common, src_center_y)
							line_dest = (x_common, dest_center_y)
							
							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)
							
							is_beneath = True

						# case 3
						elif dest_range_x[0] <= src_range_x[0] and \
							dest_range_x[1] >= src_range_x[0] and \
								dest_range_x[1] < src_range_x[1]:

							x_common = (src_range_x[0] + dest_range_x[1])/2

							line_src = (x_common , src_center_y)
							line_dest = (x_common, dest_center_y)

							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)

							is_beneath = True

						# case 4
						elif dest_range_x[0] <= src_range_x[1] and \
							dest_range_x[1] >= src_range_x[1] and \
								dest_range_x[0] > src_range_x[0]:
							
							x_common = (dest_range_x[0] + src_range_x[1])/2
							
							line_src = (x_common , src_center_y)
							line_dest = (x_common, dest_center_y)

							attributes = (dest_idx, line_src, line_dest, height)
							dest_attr_vert.append(attributes)

							is_beneath = True

					# 2 如果当前文字块不是属于下方，再判断它是不是属于右方
					if not is_beneath:
						# ======================= horizontal ==================== #
							dest_range_y = (dest_row['ymin'], dest_row['ymax'])
							# get center of destination NOTE: not used
							dest_center_x = (dest_row['xmin'] + dest_row['xmax'])/2

							# get length from destination center to source center
							if dest_center_x > src_center_x:
								length = dest_center_x - src_center_x
							else:
								length = 0

							# consider only the cases where the destination object
							# lies to the right of source
							if dest_center_x > src_center_x:
								#check if vertical range of dest lies within range of source
								# 如果想考虑右上方、右下方（垂直无重叠）的连接，增加2个case即可，并设置阈值

								# case 1
								if dest_range_y[0] >= src_range_y[0] and \
									dest_range_y[1] <= src_range_y[1]:

									y_common = (dest_range_y[0] + dest_range_y[1])/2

									line_src = (src_center_x, y_common)
									line_dest = (dest_center_x, y_common)

									attributes = (dest_idx, line_src, line_dest, length)
									dest_attr_hori.append(attributes)

								# case 2
								elif dest_range_y[0] <= src_range_y[0] and \
									dest_range_y[1] <= src_range_y[1] and \
										dest_range_y[1] > src_range_y[0]:

									y_common = (src_range_y[0] + dest_range_y[1])/2

									line_src = (src_center_x, y_common)
									line_dest = (dest_center_x, y_common)

									attributes = (dest_idx, line_src, line_dest, length)
									dest_attr_hori.append(attributes)

								# case 3
								elif dest_range_y[0] >= src_range_y[0] and \
									dest_range_y[1] >= src_range_y[1] and \
										dest_range_y[0] < src_range_y[1]:

									y_common = (dest_range_y[0] + src_range_y[1])/2

									line_src = (src_center_x, y_common)
									line_dest = (dest_center_x, y_common)

									attributes = (dest_idx, line_src, line_dest, length)
									dest_attr_hori.append(attributes)

								# case 4
								elif dest_range_y[0] <= src_range_y[0] \
									and dest_range_y[1] >= src_range_y[1]:

									y_common = (src_range_y[0] + src_range_y[1])/2

									line_src = (src_center_x, y_common)
									line_dest = (dest_center_x, y_common)

									attributes = (dest_idx, line_src, line_dest, length)
									dest_attr_hori.append(attributes)

			# sort list of destination attributes by height/length at position 3 in tuple
			dest_attr_vert_sorted = sorted(dest_attr_vert, key = lambda x: x[3])
			dest_attr_hori_sorted = sorted(dest_attr_hori, key = lambda x: x[3])
			
			# append the index and source and destination coords to draw line 
			# ==================== vertical ================================= #
			if len(dest_attr_vert_sorted) == 0:
				nearest_dest_ids_vert.append(-1)
				x_src_coords_vert.append(-1)
				y_src_coords_vert.append(-1)
				x_dest_coords_vert.append(-1)
				y_dest_coords_vert.append(-1)
				distances.append(0)
			else:
				# 只添加距离最近的那个
				nearest_dest_ids_vert.append(dest_attr_vert_sorted[0][0])
				x_src_coords_vert.append(dest_attr_vert_sorted[0][1][0])
				y_src_coords_vert.append(dest_attr_vert_sorted[0][1][1])
				x_dest_coords_vert.append(dest_attr_vert_sorted[0][2][0])
				y_dest_coords_vert.append(dest_attr_vert_sorted[0][2][1])
				distances.append(dest_attr_vert_sorted[0][3])

			'''if len(dest_attr_vert_sorted) > 0:
				for i in dest_attr_vert_sorted:
					print(src_row['Object'], "@@",df['Object'][i[0]])
				print(len(dest_attr_vert_sorted),dest_attr_vert_sorted)
				print(nearest_dest_ids_vert[-1])
				print(df['Object'][nearest_dest_ids_vert[-1]])'''


			# ========================== horizontal ========================= #
			if len(dest_attr_hori_sorted) == 0:
				nearest_dest_ids_hori.append(-1)
				x_src_coords_hori.append(-1)
				y_src_coords_hori.append(-1)
				x_dest_coords_hori.append(-1)
				y_dest_coords_hori.append(-1)
				lengths.append(0)			
			
			else:
			# try and except for the cases where there are vertical connections
			# still to be made but all horizontal connections are accounted for
				try:
					nearest_dest_ids_hori.append(dest_attr_hori_sorted[0][0])
				except:
					nearest_dest_ids_hori.append(-1)
				
				try:
					x_src_coords_hori.append(dest_attr_hori_sorted[0][1][0])
				except:
					x_src_coords_hori.append(-1)

				try:
					y_src_coords_hori.append(dest_attr_hori_sorted[0][1][1])
				except:
					y_src_coords_hori.append(-1)

				try:
					x_dest_coords_hori.append(dest_attr_hori_sorted[0][2][0])
				except:
					x_dest_coords_hori.append(-1)

				try:
					y_dest_coords_hori.append(dest_attr_hori_sorted[0][2][1])
				except:
					y_dest_coords_hori.append(-1)

				try:
					lengths.append(dest_attr_hori_sorted[0][3])
				except:
					lengths.append(0)

			'''if len(dest_attr_hori_sorted) > 0:
				for i in dest_attr_hori_sorted:
					print(src_row['Object'], "@@", df['Object'][i[0]])
				print(len(dest_attr_hori_sorted),dest_attr_hori_sorted)
				print(nearest_dest_ids_hori[-1])
				print(df['Object'][nearest_dest_ids_hori[-1]])
			print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")'''

		# print(nearest_dest_ids_vert)
		# print(nearest_dest_ids_hori)

		# ==================== vertical ===================================== #
		# create df for plotting lines
		# df['below_object'] = df.loc[nearest_dest_ids_vert, 'Object'].values
		df['below_object'] = df.reindex(nearest_dest_ids_vert, columns=['Object']).values # 索引值为-1的地方为NaN

		# add distances column
		df['below_dist'] = distances
		
		# add column containing index of destination object
		df['below_obj_index'] = nearest_dest_ids_vert

		# add coordinates for plotting
		df_plot['x_src_vert'] = x_src_coords_vert
		df_plot['y_src_vert'] = y_src_coords_vert
		df_plot['x_dest_vert'] = x_dest_coords_vert
		df_plot['y_dest_vert'] = y_dest_coords_vert
		
		# df.fillna('NULL', inplace = True)

		# ==================== horizontal =================================== #
		# create df for plotting lines
		# df['side_object'] = df.loc[nearest_dest_ids_hori, 'Object'].values
		df['side_object'] = df.reindex(nearest_dest_ids_hori, columns = ['Object']).values

		# add lengths column
		df['side_length'] = lengths

		# add column containing index of destination object
		df['side_obj_index'] = nearest_dest_ids_hori

		# add coordinates for plotting
		df_plot['x_src_hori'] = x_src_coords_hori
		df_plot['y_src_hori'] = y_src_coords_hori
		df_plot['x_dest_hori'] = x_dest_coords_hori
		df_plot['y_dest_hori'] = y_dest_coords_hori

		########################## concat df and df_plot ######################
		
		df_merged = pd.concat([df, df_plot], axis=1)

		# if an object has more than one parent above it, only the connection 
		# with the smallest distance is retained and the other distances are 
		# replaced by '-1' to get such objects, group by 'below_object' column 
		# and use minimum of 'below_dist'
		#  下面的操作是为了：
		#   当一个文字块有多个垂直连接时，只保留距离最短的一个连接；同理水平方向也是。
		# ======================= vertical ================================== #
		groups_vert = df_merged.groupby('below_obj_index')['below_dist'].min()
		# 用below_obj_index分组，最后取below_dist,再取该列分组中的最小值的行
		# groups.index gives a list of the below_object text and groups.values 
		# gives the corresponding minimum distance
		# print(groups_vert) # 索引是组名（below_obj_index），值是距离
		groups_dict_vert = dict(zip(groups_vert.index, groups_vert.values))

		# ======================= horizontal ================================ #
		groups_hori = df_merged.groupby('side_obj_index')['side_length'].min()
		# groups.index gives a list of the below_object text and groups.values 
		# gives the corresponding minimum distance
		groups_dict_hori = dict(zip(groups_hori.index, groups_hori.values))


		revised_distances_vert = []
		revised_distances_hori = []

		rev_x_src_vert, rev_y_src_vert, rev_x_dest_vert, rev_y_dest_vert = \
																[], [], [], []
		rev_x_src_hori, rev_y_src_hori, rev_x_dest_hori, rev_y_dest_hori = \
																[], [], [], []
		
		for idx, row in df_merged.iterrows():
			below_idx = row['below_obj_index']
			side_idx = row['side_obj_index']

			# ======================== vertical ============================= #
			# 修正距离远的连接
			if row['below_dist'] > groups_dict_vert[below_idx]:
				# revised_distances_vert.append(-1)
				revised_distances_vert.append(0) # changed by me
				rev_x_src_vert.append(-1)
				rev_y_src_vert.append(-1)
				rev_x_dest_vert.append(-1)
				rev_y_dest_vert.append(-1)

			else:
				revised_distances_vert.append(row['below_dist'])
				rev_x_src_vert.append(row['x_src_vert'])
				rev_y_src_vert.append(row['y_src_vert'])
				rev_x_dest_vert.append(row['x_dest_vert'])
				rev_y_dest_vert.append(row['y_dest_vert'])

			# ========================== horizontal ========================= #
			if row['side_length'] > groups_dict_hori[side_idx]:
				# revised_distances_hori.append(-1)
				revised_distances_hori.append(0) # changed by me
				rev_x_src_hori.append(-1)
				rev_y_src_hori.append(-1)
				rev_x_dest_hori.append(-1)
				rev_y_dest_hori.append(-1)

			else:
				revised_distances_hori.append(row['side_length'])
				rev_x_src_hori.append(row['x_src_hori'])
				rev_y_src_hori.append(row['y_src_hori'])
				rev_x_dest_hori.append(row['x_dest_hori'])
				rev_y_dest_hori.append(row['y_dest_hori'])

		# store in dataframe
		# ============================ vertical ============================= #
		df['revised_distances_vert'] = revised_distances_vert
		df_merged['x_src_vert'] = rev_x_src_vert
		df_merged['y_src_vert'] = rev_y_src_vert
		df_merged['x_dest_vert'] = rev_x_dest_vert
		df_merged['y_dest_vert'] = rev_y_dest_vert

		# ======================== horizontal =============================== #
		df['revised_distances_hori'] = revised_distances_hori
		df_merged['x_src_hori'] = rev_x_src_hori
		df_merged['y_src_hori'] = rev_y_src_hori
		df_merged['x_dest_hori'] = rev_x_dest_hori
		df_merged['y_dest_hori'] = rev_y_dest_hori
		
		
		# plot image if plot==True 
		if plot == True:
		
			# make folder to store output
			if not os.path.exists('grapher_outputs'):
				os.makedirs('grapher_outputs')	
			
			# subdirectory to store plots
			if not os.path.exists('./grapher_outputs/plots'):
				os.makedirs('./grapher_outputs/plots')
			
			# check if image exists in folder
			try: 
				if len(img) == None:
					pass
			except: 
				pass
			
			# plot if image exists
			else:
				img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # add by me !
				for idx, row in df_merged.iterrows():
					# print(int(row['x_src_vert']),  type(row['y_src_vert']))
					if row['x_dest_vert'] != -1:
						cv2.line(img,
								(int(row['x_src_vert']), int(row['y_src_vert'])),
								# (int(row['x_src_vert']), int(row['ymax'])),
								(int(row['x_dest_vert']), int(row['y_dest_vert'])),
								# (int(row['x_dest_vert']), int(df_merged['ymin'][row['below_obj_index']])),
								(0,0,255), 2)

					if row['x_dest_hori'] != -1:
						cv2.line(img,
								# (int(row['x_src_hori']), int(row['y_src_hori'])),
								(int(row['xmax']), int(row['y_src_hori'])),
								# (int(row['x_dest_hori']), int(row['y_dest_hori'])),
								(int(df_merged['xmin'][row['side_obj_index']]), int(row['y_dest_hori'])),
								(0,255,0), 2)

				# write image in same folder
				PLOT_PATH = \
					'./grapher_outputs/plots/' + 'object_tree_' + str(self.file_name) + '.jpg'
				cv2.imwrite(PLOT_PATH, img)


		# export dataframe with destination objects to csv in same folder
		if export_df == True:
			
			# make folder to store output
			if not os.path.exists('grapher_outputs'):
				os.makedirs('grapher_outputs')			
			
			# subdirectory to store plots
			if not os.path.exists('./grapher_outputs/connections'):
				os.makedirs('./grapher_outputs/connections')

			CSV_PATH = \
				'./grapher_outputs/connections/' + 'connections_' + str(self.file_name) + '.csv'
			df.to_csv(CSV_PATH, index = None)
			 
		# convert dataframe to dict:
		# {src_id: dest_1, dest_2, ..}

		graph_dict = {}
		for src_id, row in df.iterrows():
			if row['below_obj_index'] != -1:
				if src_id in graph_dict:
					graph_dict[src_id].append(row['below_obj_index'])
				else:
					graph_dict[src_id] = [row['below_obj_index']]

			if row['side_obj_index'] != -1:
				if src_id in graph_dict:
					graph_dict[src_id].append(row['side_obj_index'])
				else:
					graph_dict[src_id] = [row['side_obj_index']]

		return graph_dict, df['Object'].tolist(),  df['label'].tolist()


class Graph:
	'''
		This class generates a padded adjacency matrix and a feature matrix
	'''
	def __init__(self, max_nodes=50, resize=False):
		self.max_nodes = max_nodes
		self.resize = resize
		return

	# def make_graph(self, graph_dict):
	# 	'''
	# 		Function to make networkx graph

	# 		Args:
	# 			graph_dict: dict of lists, 
	# 						{src_id: [dest_id]}

				
	# 		Returns:
	# 			G: 
	# 				Padded adjacency matrix of size (max_nodes, max_nodes)

	# 			feats:
	# 				Padded feature matrix of size (max_nodes, m)
	# 				(m: dimension of node text vector)
	# 	'''
	# 	G = nx.from_dict_of_lists(graph_dict)

	# 	return G

	def _get_text_features(self, data): 
    
		'''
			Args:
				str, input data
				
			Returns: 
				np.array, shape=(22,);
				an array of the text converted to features
				
		'''
		
		assert type(data) == str, f'Expected type {str}. Received {type(data)}.'
		lenth = len(data)
		n_upper = 0
		n_lower = 0
		n_alpha = 0
		n_digits = 0
		n_spaces = 0
		n_numeric = 0
		n_special = 0
		number = 0
		special_chars = {'&': 0, '@': 1, '#': 2, '(': 3, ')': 4, '-': 5, '+': 6, 
						'=': 7, '*': 8, '%': 9, '.':10, ',': 11, '\\': 12,'/': 13, 
						'|': 14, ':': 15}
		
		special_chars_arr = np.zeros(shape=len(special_chars))    
		
		# character wise
		for char in data: 
			
			# for lower letters 
			if char.islower(): 
				n_lower += 1
	
			# for upper letters 
			if char.isupper(): 
				n_upper += 1
			
			# for white spaces
			if char.isspace():
				n_spaces += 1
			
			# for alphabetic chars
			if char.isalpha():
				n_alpha += 1
			
			# for numeric chars
			if char.isnumeric():
				n_numeric += 1
			
			# array for special chars
			if char in special_chars.keys():
				char_idx = special_chars[char]
				# put 1 at index
				special_chars_arr[char_idx] += 1
				
		# word wise # 统计字符串中数字的个数
		# for word in data.split():
		for word in re.split('[/-]',data):

			# if digit is integer 
			try:
				number = int(word) # word表示的不是整数数据字符串时候会报错
				n_digits += 1
			except:
				pass

			# if digit is float
			if n_digits == 0:
				try:
					number = float(word) # word表示的不是浮点数据字符串时候回报错
					n_digits += 1
				except:
					pass
		
		features = []
		features.append([n_lower/lenth, n_upper/lenth, n_spaces, n_alpha/lenth, n_numeric/lenth, n_digits])
		features = np.array(features)
		features = np.append(features, np.array(special_chars_arr))
		vec_arr = code_sentence._generate_txt_vec(data)
		features = np.append(features, vec_arr)

		return features

	def _get_text_labels(self, data):
		label_classes = ['buyer', 'date', 'no', 'amount', 'o']
		if self.resize :
			label_classes.append('virtual')
		mlb = MultiLabelBinarizer(classes=label_classes)
		la = mlb.fit_transform([[data]])

		return la[0]

	def _pad_adj(self, adj):
		'''
			This method resizes the input Adjacency matrix to shape 
			(self.max_nodes, self.max_nodes)

			adj: 
				2d numpy array
				adjacency matrix
		'''
		
		assert adj.shape[0] == adj.shape[1], f'The input adjacency matrix is \
			not square and has shape {adj.shape}'
		
		# get n of nxn matrix
		n = adj.shape[0]
		
		if n < self.max_nodes:
			target = np.zeros(shape=(self.max_nodes, self.max_nodes))

			# fill in the target matrix with the adjacency
			target[:adj.shape[0], :adj.shape[1]] = adj
			
		elif n > self.max_nodes:
			# cut away the excess rows and columns of adj
			target = adj[:self.max_nodes, :self.max_nodes]
			
		else:
			# do nothing
			target = adj
			
		return target
	
	def _pad_text_features(self, feat_arr):
		'''
			This method pads the feature matrix to size 
			(self.max_nodes, feat_arr.shape[1])
		'''
		target = np.zeros(shape=(self.max_nodes, feat_arr.shape[1]))

		if self.max_nodes > feat_arr.shape[0]:
			target[:feat_arr.shape[0], :feat_arr.shape[1]] = feat_arr

		elif self.max_nodes < feat_arr.shape[0]:
			# target = feat_arr[:self.max_nodes, feat_arr.shape[1]] # error
			target = feat_arr[:self.max_nodes, :feat_arr.shape[1]]

		else: 
			target = feat_arr

		return target

	def _pad_text_labels(self, label_arr):
		'''
			This method pads the feature matrix to size
			(self.max_nodes, label_arr.shape[1])
		'''
		target = np.zeros(shape=(self.max_nodes, label_arr.shape[1]))

		if self.max_nodes > label_arr.shape[0]:
			target[:label_arr.shape[0], :label_arr.shape[1]] = label_arr
			target[label_arr.shape[0]:, -1] = 1 # 即one-hot label 最后一个标签位为1 代表是增加的虚拟节点

		elif self.max_nodes < label_arr.shape[0]:
			target = label_arr[:self.max_nodes, :label_arr.shape[1]]

		else:
			target = label_arr

		return target


	def make_graph_data(self, graph_dict, text_list, label_list):
		'''
			Function to make an adjacency matrix from a networkx graph object
			as well as padded feature matrix

			Args:
				G: networkx graph object
				
				text_list: list,
							of text entities:
							['Tax Invoice', '1/2/2019', ...]

			Returns:
				A: Adjacency matrix as np.array

				X: Feature matrix as numpy array for input graph
		'''
		G = nx.from_dict_of_lists(graph_dict)
		adj_sparse = nx.adjacency_matrix(G)

		# preprocess the sparse adjacency matrix returned by networkx function
		adj_arr = np.array(adj_sparse.todense())
		# print(len(adj_arr), adj_arr)
		if self.resize:
			adj_arr = self._pad_adj(adj_arr)
			# print(adj_arr.dtype) # int32

		# preprocess the list of text entities
		# 节点初始向量表示生成模块，每个节点维度要一致，即句子编码长度相同，
		# 后期可以要改成句子的词典one-hot表达或者其他句子级别的词向量，统一长度，作为节点初始输入
		feat_list = list(map(self._get_text_features, text_list))
		feat_arr = np.array(feat_list)
		# print(len(feat_arr),feat_arr)
		if self.resize:
			feat_arr = self._pad_text_features(feat_arr)
			# print(feat_arr.dtype) # float64

		# preprocess the list of text labels
		la_list = list(map(self._get_text_labels, label_list))
		la_arr = np.array(la_list)
		# print(len(la_arr), la_arr)
		if self.resize:
			la_arr = self._pad_text_labels(la_arr)
			# print(la_arr.dtype) #int32

		return adj_arr, feat_arr, la_arr

if __name__ == "__main__":
	print(os.getcwd())
	map_file = './data/object_map.csv'
	img_file = './data/object_map.jpg'
	file_name = os.path.basename(map_file)[:-4]

	df = pd.read_csv(map_file)
	img = cv2.imread(img_file, 0)
	
	tree = ObjectTree()
	# tree.read(df, img)
	tree.read(df, img, file_name)

	graph_dict, text_list, label_list = tree.connect(plot=True, export_df=True)
	
	print(graph_dict)
	print(text_list)
	print(label_list)
	print('\n--------------------------------------------------------------\n')

	graph = Graph(max_nodes=50, resize=False)
	A, X, L = graph.make_graph_data(graph_dict, text_list, label_list)
	np.save("./npy_data/a.npy", A)
	np.save("./npy_data/x.npy", X)
	np.save("./npy_data/l.npy", L)

	print(A.shape)
	# print(A)
	print('-----------------------------------------------------------------\n')
	print(X.shape)
	# print(X)
	print('-----------------------------------------------------------------\n')
	print(L.shape)
	# print(L)